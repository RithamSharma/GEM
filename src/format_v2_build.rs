// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Production heterogeneous formatter: scheduled macro nodes -> validated,
//! field/bit-major, 64-bit, warp-coalesced V2 program + a coalesced state-gather
//! plan + a live-range shared-slot allocator.
//!
//! Completes the Host-Side Macro Memory Formatter plan:
//!
//! * **Phase 1** -- [`resolve_source`] turns a macro operand pin-with-invert
//!   into a typed [`SourceSel`] using *the scheduled edge*, never a raw net id:
//!   constant -> `Constant`; primary input / DFF Q / SRAM RD / DSP `P`
//!   (old state) -> `PreviousState`; producer in an earlier local wave ->
//!   `LocalShared`; producer in an earlier major stage -> `CurrentStage`.
//! * **Phase 3** -- every selector grid is laid out field/bit-major through
//!   [`crate::format_v2::transpose_selectors`], so lanes `0..32` at a fixed
//!   `(field, bit)` read consecutive `u64` words.
//! * **Phase 4** -- [`build_gather_plan`] dedups the non-local source words a
//!   wave needs, groups them into coalesced read rounds, and hands back a
//!   preamble plan; [`allocate_shared_slots`] assigns same-cycle macro values
//!   to shared bit-slots by live range so the peak is the true simultaneous
//!   maximum, not the sum of every operand and output.
//! * **Phase 6 (Rust half)** -- [`build_resolved_program`] returns the flat
//!   `Vec<u64>` + [`ProgramLayoutV2`]; `format_v2::decode_selector_section`
//!   reads it back independently, and `format_v2_cpu` evaluates it.
//!
//! [`build_partitioned_program`] consumes the versioned heterogeneous
//! placement, promotes cross-partition same-cycle values to `CurrentStage`,
//! and emits the partition count used by the cooperative-grid executor.

use indexmap::IndexMap;

use crate::aig::{DriverType, AIG};
use crate::aigpdk::AIGPDK_SRAM_SIZE;
use crate::format_v2::{
    assemble_macro_program, content_hash, decode_destination, decode_source, encode_destination,
    encode_source, validate, DestinationSel, DestinationSpace, FormatError, MacroSelSpec,
    ProgramLayoutV2, Section, SectionKind, SelField, SourceSel, SourceSpace, V2_HEADER_WORDS,
};
use crate::hetero_parts::HeteroPlacementV2;
use crate::schedule::{HeteroSchedule, MacroKind};

/// Conservative dynamic-shared-memory ceiling for the portable V2 launch.
/// 4096 u64 words = 32 KiB, below the default per-block limit of every CUDA
/// device targeted by this project. Larger graphs must be partitioned instead
/// of failing later with an opaque CUDA launch error.
pub const V2_MAX_SHARED_WORDS: u32 = 4096;

// ===========================================================================
// canonical, contiguous field tables (must match csrc/format_v2_decode.cuh)
// ===========================================================================

pub const DSP_SRC_FIELDS: &[SelField] = &[
    SelField {
        name: "A",
        width: 27,
    },
    SelField {
        name: "D",
        width: 27,
    },
    SelField {
        name: "B",
        width: 18,
    },
    SelField {
        name: "C",
        width: 48,
    },
    SelField {
        name: "OPMODE",
        width: 9,
    },
    SelField {
        name: "ALUMODE",
        width: 4,
    },
    SelField {
        name: "INMODE",
        width: 5,
    },
    SelField {
        name: "CEP",
        width: 1,
    },
    SelField {
        name: "RSTP",
        width: 1,
    },
    SelField {
        name: "CLK_EDGE",
        width: 1,
    },
];
pub const DSP_DST_FIELDS: &[SelField] = &[SelField {
    name: "P",
    width: 48,
}];

pub const CARRY4_SRC_FIELDS: &[SelField] = &[
    SelField {
        name: "S",
        width: 4,
    },
    SelField {
        name: "DI",
        width: 4,
    },
    SelField {
        name: "CIN",
        width: 1,
    },
    SelField {
        name: "CYINIT",
        width: 1,
    },
];
pub const CARRY4_DST_FIELDS: &[SelField] = &[
    SelField {
        name: "O",
        width: 4,
    },
    SelField {
        name: "CO",
        width: 4,
    },
];

pub const SRLC_SRC_FIELDS: &[SelField] = &[
    SelField {
        name: "D",
        width: 1,
    },
    SelField {
        name: "CE",
        width: 1,
    },
    SelField {
        name: "A",
        width: 5,
    },
    SelField {
        name: "CLK_EDGE",
        width: 1,
    },
];
pub const SRLC_DST_FIELDS: &[SelField] = &[
    SelField {
        name: "Q",
        width: 1,
    },
    SelField {
        name: "Q31",
        width: 1,
    },
];

/// The ordered operand pins-with-invert of one CARRY4, in `CARRY4_SRC_FIELDS`
/// flat-bit order.
fn carry4_operand_pins(blk: &crate::aig::Carry4Block) -> Vec<usize> {
    let mut v = Vec::with_capacity(10);
    v.extend_from_slice(&blk.s_iv);
    v.extend_from_slice(&blk.di_iv);
    v.push(blk.cin_iv);
    v.push(blk.cyinit_iv);
    v
}
fn carry4_result_pins(blk: &crate::aig::Carry4Block) -> Vec<usize> {
    let mut v = Vec::with_capacity(8);
    v.extend(blk.o_out.iter().map(|&p| if p == 0 { 0 } else { p << 1 }));
    v.extend(blk.co_out.iter().map(|&p| if p == 0 { 0 } else { p << 1 }));
    v
}
fn dsp_operand_pins(blk: &crate::aig::DSPBlock) -> Vec<usize> {
    let mut v = Vec::with_capacity(140);
    v.extend_from_slice(&blk.a_iv);
    v.extend_from_slice(&blk.d_iv);
    v.extend_from_slice(&blk.b_iv);
    v.extend_from_slice(&blk.c_iv);
    v.extend_from_slice(&blk.opmode_iv);
    v.extend_from_slice(&blk.alumode_iv);
    v.extend_from_slice(&blk.inmode_iv);
    v.push(blk.cep_iv);
    v.push(blk.rstp_iv);
    v.push(blk.clk_iv);
    v
}
fn dsp_result_pins(blk: &crate::aig::DSPBlock) -> Vec<usize> {
    blk.p_out
        .iter()
        .map(|&p| if p == 0 { 0 } else { p << 1 })
        .collect()
}
fn srlc_operand_pins(blk: &crate::aig::Srlc32eBlock) -> Vec<usize> {
    let mut v = Vec::with_capacity(7);
    v.push(blk.d_iv);
    v.push(blk.ce_iv);
    v.extend_from_slice(&blk.a_iv);
    v.push(blk.clk_iv);
    v
}
fn srlc_result_pins(blk: &crate::aig::Srlc32eBlock) -> Vec<usize> {
    vec![
        if blk.q_out == 0 { 0 } else { blk.q_out << 1 },
        if blk.q31_out == 0 {
            0
        } else {
            blk.q31_out << 1
        },
    ]
}

// ===========================================================================
// Phase 0 -- frozen state layout
// ===========================================================================

/// The previous-cycle ("old state") image plus the persistent macro registers.
/// `PreviousState` selectors index this by `(u32 word, bit 0..=31)`.
///
/// Layout (all `u32` words, matching GEM's `states_noninteractive`):
///
/// ```text
/// [ 0 .. prev_words )                primary inputs, DFF Q, SRAM RD, DSP P bits
/// [ prev_words .. + 2*n_dsp )        per-DSP 48-bit P (2 words each)  -- also aliased above
/// [ .. + n_srlc )                    per-SRLC 32-bit shift storage
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateLayout {
    /// old-state aigpin -> (word, bit) in the persistent `u32` image.
    pub prev: IndexMap<usize, (u32, u32)>,
    /// primary-output pin-with-invert -> destination in the cycle state image.
    pub outputs: IndexMap<usize, (u32, u32)>,
    pub prev_words: u32,
    /// DSP dense instance -> first `u32` word of its 48-bit `P`.
    pub dsp_p_word: Vec<u32>,
    /// SRLC dense instance -> `u32` word of its 32-bit storage.
    pub srlc_storage_word: Vec<u32>,
    pub persistent_words: u32,
}

impl StateLayout {
    /// Build from the AIG and the schedule's instance ordering.
    pub fn from_schedule(aig: &AIG, sched: &HeteroSchedule) -> Self {
        let mut prev: IndexMap<usize, (u32, u32)> = IndexMap::new();
        let mut next_bit: u32 = 0;
        let put = |prev: &mut IndexMap<usize, (u32, u32)>, next_bit: &mut u32, p: usize| {
            if p == 0 || prev.contains_key(&p) {
                return;
            }
            prev.insert(p, (*next_bit / 32, *next_bit % 32));
            *next_bit += 1;
        };
        for p in 1..=aig.num_aigpins {
            match &aig.drivers[p] {
                DriverType::InputPort(_)
                | DriverType::InputClockFlag(..)
                | DriverType::DFF(_)
                | DriverType::SRAM(_) => put(&mut prev, &mut next_bit, p),
                _ => {}
            }
        }
        let mut outputs = IndexMap::new();
        for &pin_iv in &aig.primary_outputs {
            outputs.insert(pin_iv, (next_bit / 32, next_bit % 32));
            next_bit += 1;
        }
        let prev_words = (next_bit + 31) / 32;

        // dense macro instance order from the schedule.
        let (mut dsp_cells, mut srlc_cells) = (Vec::new(), Vec::new());
        for w in &sched.waves {
            for &ni in &w.dsp48e2 {
                dsp_cells.push(sched.nodes[ni].cell_id);
            }
            for &ni in &w.srlc32e {
                srlc_cells.push(sched.nodes[ni].cell_id);
            }
        }

        let mut word = prev_words;
        let mut dsp_p_word = Vec::with_capacity(dsp_cells.len());
        for &cell in &dsp_cells {
            dsp_p_word.push(word);
            // alias each connected P bit into `prev` so a same-cycle reader of
            // this DSP's registered P resolves to PreviousState here.
            let blk = &aig.dsps[&cell];
            for (k, &p) in blk.p_out.iter().enumerate() {
                if p != 0 {
                    prev.entry(p)
                        .or_insert((word + (k as u32) / 32, (k as u32) % 32));
                }
            }
            word += 2;
        }
        let mut srlc_storage_word = Vec::with_capacity(srlc_cells.len());
        for _ in &srlc_cells {
            srlc_storage_word.push(word);
            word += 1;
        }

        StateLayout {
            prev,
            outputs,
            prev_words,
            dsp_p_word,
            srlc_storage_word,
            persistent_words: word,
        }
    }
}

// ===========================================================================
// Phase 1 -- resolve a scheduled edge into a typed selector
// ===========================================================================

/// Per-partition context the resolver needs. For the single-stage fixture path
/// `stage_of_node` is all-zero and `node_stage_out_word` is empty.
pub struct ResolveCtx<'a> {
    pub sched: &'a HeteroSchedule,
    pub state: &'a StateLayout,
    /// same-cycle net (aigpin) -> shared bit-slot (u64-word * 64 + bit).
    pub shared_slot: &'a IndexMap<usize, u32>,
    /// major stage of each schedule node (0 for the single-stage path).
    pub stage_of_node: &'a [u32],
    /// net -> (stage word, bit) for a producer in an earlier major stage.
    pub stage_word: &'a IndexMap<usize, (u32, u32)>,
    /// the stage of the consumer macro currently being resolved.
    pub consumer_stage: u32,
    /// the wave of the consumer macro currently being resolved.
    pub consumer_wave: usize,
}

/// Resolve one operand pin-with-invert. `pin_iv == 0` / `1` is the constant
/// literal `0` / `1`.
pub fn resolve_source(ctx: &ResolveCtx<'_>, pin_iv: usize) -> Result<SourceSel, FormatError> {
    let p = pin_iv >> 1;
    let invert = pin_iv & 1 == 1;
    if p == 0 {
        return Ok(SourceSel {
            space: SourceSpace::Constant,
            index: 0,
            bit: 0,
            invert,
        });
    }
    if let Some(&node) = ctx.sched.producer.get(&p) {
        let prod_stage = ctx.stage_of_node.get(node).copied().unwrap_or(0);
        if prod_stage < ctx.consumer_stage {
            let &(word, bit) = ctx
                .stage_word
                .get(&p)
                .ok_or(FormatError::ArithmeticOverflow {
                    what: "stage producer without a stage word",
                })?;
            return Ok(SourceSel {
                space: SourceSpace::CurrentStage,
                index: word,
                bit: bit as u8,
                invert,
            });
        }
        let prod_wave = ctx.sched.nodes[node].level;
        if prod_wave >= ctx.consumer_wave {
            return Err(FormatError::SameCycleEdgeMisordered {
                producer_wave: prod_wave as u32,
                consumer_wave: ctx.consumer_wave as u32,
            });
        }
        let slot = *ctx
            .shared_slot
            .get(&p)
            .ok_or(FormatError::ArithmeticOverflow {
                what: "same-cycle producer without a shared slot",
            })?;
        return Ok(SourceSel {
            space: SourceSpace::LocalShared,
            index: slot / 64,
            bit: (slot % 64) as u8,
            invert,
        });
    }
    // not produced this cycle -> old state.
    let &(word, bit) = ctx
        .state
        .prev
        .get(&p)
        .ok_or(FormatError::ArithmeticOverflow {
            what: "operand net is neither scheduled nor old-state",
        })?;
    Ok(SourceSel {
        space: SourceSpace::PreviousState,
        index: word,
        bit: bit as u8,
        invert,
    })
}

/// Resolve an AND-gate input. Gates inside one AIG region are deliberately
/// kept in topological pin order, so an input produced by an earlier gate in
/// the same region is a legal local dependency even though both operations
/// share the region's outer heterogeneous wave.
fn resolve_aig_source(
    ctx: &ResolveCtx<'_>,
    aig: &AIG,
    pin_iv: usize,
) -> Result<SourceSel, FormatError> {
    let pin = pin_iv >> 1;
    if pin != 0 && matches!(aig.drivers[pin], DriverType::AndGate(..)) {
        let slot = *ctx
            .shared_slot
            .get(&pin)
            .ok_or(FormatError::ArithmeticOverflow {
                what: "AIG producer without a shared slot",
            })?;
        return Ok(SourceSel {
            space: SourceSpace::LocalShared,
            index: slot / 64,
            bit: (slot % 64) as u8,
            invert: pin_iv & 1 != 0,
        });
    }
    resolve_source(ctx, pin_iv)
}

// ===========================================================================
// Phase 4 -- shared-slot live-range allocator
// ===========================================================================

/// Assign packed shared words to same-cycle producers that a later wave reads.
/// All outputs of one producer occupy one or more private words. This matters
/// on CUDA: the thread evaluating a macro can publish its complete packed
/// result without racing another thread that writes a different bit of the
/// same word.
///
/// Returns `(net -> slot, peak_u64_words)`. `slot` is `u64_word * 64 + bit`;
/// here every value is 1 bit so slots are packed 64 per word.
pub fn allocate_shared_slots(aig: &AIG, sched: &HeteroSchedule) -> (IndexMap<usize, u32>, u32) {
    // AIG gates are bit-packed. They remain live for the cycle so a later AIG
    // region, macro, or output commit can consume them. CUDA uses shared-memory
    // atomics for these packed writes, preventing two regions from racing on
    // different bits of the same word.
    let mut slot_of: IndexMap<usize, u32> = IndexMap::new();
    let mut next_bit = 0u32;
    for node in &sched.nodes {
        if matches!(node.kind, crate::schedule::NodeKind::AigRegion) {
            for &gate in &node.region_gates {
                slot_of.insert(gate, next_bit);
                next_bit += 1;
            }
        }
    }

    // Macro producers own whole u64 words, but their ranges are recycled after
    // the final same-cycle consumer. Include both macro consumers and AIG
    // regions so macro->AIG routing cannot disappear from the live-range map.
    let mut last_use: IndexMap<usize, usize> = IndexMap::new();
    let mut used_outputs: IndexMap<usize, Vec<usize>> = IndexMap::new();
    for (wave, typed) in sched.waves.iter().enumerate() {
        let mut inputs = Vec::new();
        for &node_id in typed
            .carry4
            .iter()
            .chain(&typed.dsp48e2)
            .chain(&typed.srlc32e)
        {
            let node = &sched.nodes[node_id];
            inputs.extend(operand_pins(aig, node.macro_kind().unwrap(), node.cell_id));
        }
        for &node_id in &typed.aig_regions {
            for &gate in &sched.nodes[node_id].region_gates {
                if let DriverType::AndGate(a, b) = aig.drivers[gate] {
                    inputs.extend([a, b]);
                }
            }
        }
        for pin_iv in inputs {
            let pin = pin_iv >> 1;
            let Some(&producer) = sched.producer.get(&pin) else {
                continue;
            };
            if sched.nodes[producer].macro_kind().is_none() || sched.nodes[producer].level >= wave {
                continue;
            }
            last_use
                .entry(producer)
                .and_modify(|last| *last = (*last).max(wave))
                .or_insert(wave);
            let outputs = used_outputs.entry(producer).or_default();
            if !outputs.contains(&pin) {
                outputs.push(pin);
            }
        }
    }
    // Values committed to primary outputs or DFFs are consumed after the final
    // heterogeneous wave and must remain live until that commit phase.
    let commit_wave = sched.waves.len();
    let mut endpoint_inputs: Vec<usize> = aig.primary_outputs.iter().copied().collect();
    for dff in aig.dffs.values() {
        endpoint_inputs.extend([dff.d_iv, dff.en_iv]);
    }
    for pin_iv in endpoint_inputs {
        let pin = pin_iv >> 1;
        let Some(&producer) = sched.producer.get(&pin) else {
            continue;
        };
        if sched.nodes[producer].macro_kind().is_none() {
            continue;
        }
        last_use
            .entry(producer)
            .and_modify(|last| *last = (*last).max(commit_wave))
            .or_insert(commit_wave);
        let outputs = used_outputs.entry(producer).or_default();
        if !outputs.contains(&pin) {
            outputs.push(pin);
        }
    }

    let aig_words = (next_bit + 63) / 64;
    let mut next_word = aig_words;
    let mut free: Vec<(u32, u32)> = Vec::new();
    let mut live: Vec<(usize, u32, u32)> = Vec::new();
    for wave in 0..sched.waves.len() {
        live.retain(|&(producer, base, len)| {
            if last_use[&producer] < wave {
                free.push((base, len));
                false
            } else {
                true
            }
        });
        for (&producer, outputs) in &used_outputs {
            if sched.nodes[producer].level != wave {
                continue;
            }
            let node_outputs = &sched.nodes[producer].outputs;
            let max_bit = outputs
                .iter()
                .filter_map(|pin| node_outputs.iter().position(|candidate| candidate == pin))
                .max()
                .unwrap_or(0) as u32;
            let need = max_bit / 64 + 1;
            let base = if let Some(index) = free.iter().position(|&(_, len)| len >= need) {
                let (base, len) = free.swap_remove(index);
                if len > need {
                    free.push((base + need, len - need));
                }
                base
            } else {
                let base = next_word;
                next_word += need;
                base
            };
            for &net in outputs {
                let bit = node_outputs
                    .iter()
                    .position(|candidate| *candidate == net)
                    .unwrap() as u32;
                slot_of.insert(net, base * 64 + bit);
            }
            live.push((producer, base, need));
        }
    }
    (slot_of, next_word)
}

/// One coalesced global read round of the level preamble: up to 32 `u32`
/// old-state words, lane `i` loading `words[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherRound {
    pub words: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GatherPlan {
    pub rounds: Vec<GatherRound>,
    /// old-state `u32` word -> the `u64` shared-arena word the preamble drops it
    /// into (low 32 bits). Consumers then read `LocalShared`.
    pub staged_word_slot: IndexMap<u32, u32>,
    pub peak_shared_words: u32,
}

/// Phase 4: dedup the `PreviousState` words every macro reads and group them
/// into <=32-word coalesced rounds. `shared_base_word` is where the staged
/// old-state words live in the shared arena (after the same-cycle value slots).
pub fn build_gather_plan(resolved_sources: &[SourceSel], shared_base_word: u32) -> GatherPlan {
    let mut words: Vec<u32> = resolved_sources
        .iter()
        .filter(|s| s.space == SourceSpace::PreviousState)
        .map(|s| s.index)
        .collect();
    words.sort_unstable();
    words.dedup();

    let mut staged_word_slot = IndexMap::new();
    for (i, &w) in words.iter().enumerate() {
        staged_word_slot.insert(w, shared_base_word + i as u32);
    }
    let rounds = words
        .chunks(32)
        .map(|c| GatherRound { words: c.to_vec() })
        .collect();

    GatherPlan {
        rounds,
        peak_shared_words: shared_base_word + words.len() as u32,
        staged_word_slot,
    }
}

/// After the gather, every macro operand reads from `LocalShared`. Rewrite a
/// pre-gather selector into its post-gather form.
pub fn apply_gather(sel: SourceSel, plan: &GatherPlan) -> SourceSel {
    match sel.space {
        SourceSpace::PreviousState => {
            let staged = plan.staged_word_slot[&sel.index];
            SourceSel {
                space: SourceSpace::LocalShared,
                index: staged,
                bit: sel.bit, // 0..=31 within the staged u32
                invert: sel.invert,
            }
        }
        SourceSpace::CurrentStage => {
            // cross-stage words are also staged into shared by the preamble;
            // callers that use >1 stage extend `staged_word_slot` accordingly.
            sel
        }
        _ => sel,
    }
}

// ===========================================================================
// Phase 1+3+4 -- assemble the resolved program
// ===========================================================================

#[derive(Clone, Debug)]
pub struct MacroQueue {
    pub kind: MacroKind,
    /// dense cell ids, schedule-wave order.
    pub cells: Vec<usize>,
    /// wave (schedule level) of each cell, same order as `cells`.
    pub waves: Vec<usize>,
}

/// Dense ranges in each type queue belonging to one topological wave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaveDescriptor {
    pub aig_start: u32,
    pub aig_count: u32,
    pub aig_depths: u32,
    pub carry4_start: u32,
    pub carry4_count: u32,
    pub dsp_start: u32,
    pub dsp_count: u32,
    pub srlc_start: u32,
    pub srlc_count: u32,
}

/// One topologically ordered AND operation in the unified V2 stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AigOperation {
    pub wave: usize,
    pub depth: u32,
    pub output_pin: usize,
    pub a: SourceSel,
    pub b: SourceSel,
    pub dst: DestinationSel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointKind {
    PrimaryOutput = 0,
    Dff = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndpointOperation {
    pub kind: EndpointKind,
    pub data: SourceSel,
    pub enable: SourceSel,
    pub dst: DestinationSel,
    /// Primary-output pin-with-invert, or DFF Q pin for diagnostics.
    pub logical_pin: usize,
}

pub const SRAM_SOURCE_BITS: usize = 1 + 13 + 13 + 32 + 32;
pub const SRAM_OPERATION_WORDS: usize = 1 + SRAM_SOURCE_BITS + 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SramOperation {
    pub storage_base: u32,
    /// read-enable, read-address, write-address, write-enables, write-data.
    pub sources: Vec<SourceSel>,
    pub read_destinations: Vec<Option<DestinationSel>>,
}

#[derive(Clone, Debug)]
pub struct ResolvedProgram {
    pub program: Vec<u64>,
    pub layout: ProgramLayoutV2,
    pub state: StateLayout,
    pub gather: GatherPlan,
    pub shared_slot: IndexMap<usize, u32>,
    pub queues: Vec<MacroQueue>,
    pub waves: Vec<WaveDescriptor>,
    pub aig_operations: Vec<AigOperation>,
    pub endpoint_operations: Vec<EndpointOperation>,
    pub sram_operations: Vec<SramOperation>,
    pub sram_storage_words: u32,
    /// pre-gather source selectors per macro kind, `[flat_bit][instance]`,
    /// kept for tests / the CPU interpreter / diagnostics.
    pub sources_pre_gather: Vec<(MacroKind, Vec<Vec<SourceSel>>)>,
    pub initial_state: Vec<u32>,
    /// Shared u64 words containing same-cycle values before gathered old-state
    /// words begin. Partitioned mode remaps precisely this prefix to global
    /// CurrentStage storage.
    pub same_cycle_shared_words: u32,
    /// Global CurrentStage u32 words required by the selected placement.
    pub current_stage_words: u32,
    pub num_partitions: u32,
}

/// The production entry point named in the plan's Phase 0 output contract,
/// specialised to what this tree actually has (`AIG` + `HeteroSchedule`, one
/// major stage).
pub fn build_resolved_program(
    aig: &AIG,
    sched: &HeteroSchedule,
) -> Result<ResolvedProgram, FormatError> {
    let state = StateLayout::from_schedule(aig, sched);
    let (shared_slot, same_cycle_words) = allocate_shared_slots(aig, sched);

    // queues, dense order.
    let mut queues: Vec<MacroQueue> = Vec::new();
    for kind in [MacroKind::Dsp48e2, MacroKind::Carry4, MacroKind::Srlc32e] {
        let mut cells = Vec::new();
        let mut waves = Vec::new();
        for (wi, w) in sched.waves.iter().enumerate() {
            let q = match kind {
                MacroKind::Dsp48e2 => &w.dsp48e2,
                MacroKind::Carry4 => &w.carry4,
                MacroKind::Srlc32e => &w.srlc32e,
            };
            for &ni in q {
                cells.push(sched.nodes[ni].cell_id);
                waves.push(wi);
            }
        }
        if !cells.is_empty() {
            queues.push(MacroQueue { kind, cells, waves });
        }
    }

    let stage_of_node = vec![0u32; sched.nodes.len()];
    let stage_word: IndexMap<usize, (u32, u32)> = IndexMap::new();

    // Resolve every AIG operation first. Gates within one region are emitted
    // in AIG pin order, which is guaranteed topological by AIG construction.
    let mut aig_operations_pre: Vec<AigOperation> = Vec::new();
    let mut all_sources_flat: Vec<SourceSel> = Vec::new();
    for (wave, typed) in sched.waves.iter().enumerate() {
        for &node_id in &typed.aig_regions {
            let node = &sched.nodes[node_id];
            for &gate in &node.region_gates {
                let (a_iv, b_iv) = match aig.drivers[gate] {
                    DriverType::AndGate(a, b) => (a, b),
                    _ => unreachable!("AIG region contains a non-AND node"),
                };
                let ctx = ResolveCtx {
                    sched,
                    state: &state,
                    shared_slot: &shared_slot,
                    stage_of_node: &stage_of_node,
                    stage_word: &stage_word,
                    consumer_stage: 0,
                    consumer_wave: wave,
                };
                let a = resolve_aig_source(&ctx, aig, a_iv)?;
                let b = resolve_aig_source(&ctx, aig, b_iv)?;
                let slot = shared_slot[&gate];
                let dst = DestinationSel {
                    space: DestinationSpace::LocalShared,
                    index: slot / 64,
                    bit: (slot % 64) as u8,
                };
                let input_depth = |pin_iv: usize| {
                    let pin = pin_iv >> 1;
                    aig_operations_pre
                        .iter()
                        .rev()
                        .find(|op| op.wave == wave && op.output_pin == pin)
                        .map_or(0, |op| op.depth + 1)
                };
                let depth = input_depth(a_iv).max(input_depth(b_iv));
                all_sources_flat.extend([a, b]);
                aig_operations_pre.push(AigOperation {
                    wave,
                    depth,
                    output_pin: gate,
                    a,
                    b,
                    dst,
                });
            }
        }
    }

    // resolve every macro operand selector (pre-gather).
    let mut sources_pre_gather: Vec<(MacroKind, Vec<Vec<SourceSel>>)> = Vec::new();
    for q in &queues {
        let fields = src_fields(q.kind);
        let total_bits: usize = fields.iter().map(|f| f.width).sum();
        let mut grid = vec![vec![SourceSel::CONST0; q.cells.len()]; total_bits];
        for (inst, (&cell, &wi)) in q.cells.iter().zip(&q.waves).enumerate() {
            let pins = operand_pins(aig, q.kind, cell);
            debug_assert_eq!(pins.len(), total_bits);
            let ctx = ResolveCtx {
                sched,
                state: &state,
                shared_slot: &shared_slot,
                stage_of_node: &stage_of_node,
                stage_word: &stage_word,
                consumer_stage: 0,
                consumer_wave: wi,
            };
            for (fb, pin_iv) in pins.into_iter().enumerate() {
                let sel = resolve_source(&ctx, pin_iv)?;
                grid[fb][inst] = sel;
                all_sources_flat.push(sel);
            }
        }
        sources_pre_gather.push((q.kind, grid));
    }

    // End-of-cycle primary-output and DFF commits consume the settled result
    // after every heterogeneous wave.
    let endpoint_ctx = ResolveCtx {
        sched,
        state: &state,
        shared_slot: &shared_slot,
        stage_of_node: &stage_of_node,
        stage_word: &stage_word,
        consumer_stage: 0,
        consumer_wave: sched.waves.len(),
    };
    let mut endpoint_operations_pre = Vec::new();
    for &pin_iv in &aig.primary_outputs {
        let pin = pin_iv >> 1;
        // A registered DSP P is an old-state source while computing the next
        // P, but a primary output at this timestamp must observe the P value
        // committed by the just-completed edge. The DSP destination publishes
        // that value to current_stage at this exact state address.
        let data = if matches!(aig.drivers[pin], DriverType::DSP(..)) {
            let (word, bit) = state.prev[&pin];
            SourceSel {
                space: SourceSpace::CurrentStage,
                index: word,
                bit: bit as u8,
                invert: pin_iv & 1 != 0,
            }
        } else {
            resolve_aig_source(&endpoint_ctx, aig, pin_iv)?
        };
        let (word, bit) = state.outputs[&pin_iv];
        endpoint_operations_pre.push(EndpointOperation {
            kind: EndpointKind::PrimaryOutput,
            data,
            enable: SourceSel::CONST1,
            dst: DestinationSel {
                space: DestinationSpace::NextState,
                index: word,
                bit: bit as u8,
            },
            logical_pin: pin_iv,
        });
        all_sources_flat.push(data);
    }
    for dff in aig.dffs.values() {
        let data = resolve_aig_source(&endpoint_ctx, aig, dff.d_iv)?;
        let enable = resolve_aig_source(&endpoint_ctx, aig, dff.en_iv)?;
        let (word, bit) = state.prev[&dff.q];
        endpoint_operations_pre.push(EndpointOperation {
            kind: EndpointKind::Dff,
            data,
            enable,
            dst: DestinationSel {
                space: DestinationSpace::NextState,
                index: word,
                bit: bit as u8,
            },
            logical_pin: dff.q,
        });
        all_sources_flat.extend([data, enable]);
    }

    let mut sram_operations_pre = Vec::with_capacity(aig.srams.len());
    for (instance, ram) in aig.srams.values().enumerate() {
        let mut sources = Vec::with_capacity(SRAM_SOURCE_BITS);
        sources.push(resolve_aig_source(&endpoint_ctx, aig, ram.port_r_en_iv)?);
        for &pin in &ram.port_r_addr_iv {
            sources.push(resolve_aig_source(&endpoint_ctx, aig, pin)?);
        }
        for &pin in &ram.port_w_addr_iv {
            sources.push(resolve_aig_source(&endpoint_ctx, aig, pin)?);
        }
        for &pin in &ram.port_w_wr_en_iv {
            sources.push(resolve_aig_source(&endpoint_ctx, aig, pin)?);
        }
        for &pin in &ram.port_w_wr_data_iv {
            sources.push(resolve_aig_source(&endpoint_ctx, aig, pin)?);
        }
        debug_assert_eq!(sources.len(), SRAM_SOURCE_BITS);
        all_sources_flat.extend(sources.iter().copied());
        let read_destinations = ram
            .port_r_rd_data
            .iter()
            .map(|&pin| {
                state.prev.get(&pin).map(|&(word, bit)| DestinationSel {
                    space: DestinationSpace::NextState,
                    index: word,
                    bit: bit as u8,
                })
            })
            .collect();
        sram_operations_pre.push(SramOperation {
            storage_base: (instance * AIGPDK_SRAM_SIZE) as u32,
            sources,
            read_destinations,
        });
    }

    // gather plan for the PreviousState words, staged right after the
    // same-cycle value slots.
    let gather = build_gather_plan(&all_sources_flat, same_cycle_words);
    if gather.peak_shared_words > V2_MAX_SHARED_WORDS {
        return Err(FormatError::SharedBudgetExceeded {
            needed: gather.peak_shared_words,
            limit: V2_MAX_SHARED_WORDS,
        });
    }

    let aig_operations: Vec<AigOperation> = aig_operations_pre
        .iter()
        .map(|op| AigOperation {
            wave: op.wave,
            depth: op.depth,
            output_pin: op.output_pin,
            a: apply_gather(op.a, &gather),
            b: apply_gather(op.b, &gather),
            dst: op.dst,
        })
        .collect();
    let endpoint_operations: Vec<EndpointOperation> = endpoint_operations_pre
        .iter()
        .map(|op| EndpointOperation {
            kind: op.kind,
            data: apply_gather(op.data, &gather),
            enable: apply_gather(op.enable, &gather),
            dst: op.dst,
            logical_pin: op.logical_pin,
        })
        .collect();
    let sram_operations: Vec<SramOperation> = sram_operations_pre
        .iter()
        .map(|op| SramOperation {
            storage_base: op.storage_base,
            sources: op
                .sources
                .iter()
                .map(|&source| apply_gather(source, &gather))
                .collect(),
            read_destinations: op.read_destinations.clone(),
        })
        .collect();

    // post-gather selectors + destinations, transposed and assembled.
    let mut specs: Vec<MacroSelSpec<'_>> = Vec::new();
    for q in &queues {
        specs.push(MacroSelSpec {
            source_kind: src_section_kind(q.kind),
            dest_kind: dst_section_kind(q.kind),
            source_fields: src_fields(q.kind),
            dest_fields: dst_fields(q.kind),
            n_instances: q.cells.len(),
        });
    }

    // scoped so the selector closures release their borrows of `sources_pre_gather`,
    // `gather`, `queues`, `shared_slot` and `state` before we move them into the
    // returned `ResolvedProgram`.
    let (mut program, mut layout) = {
        let src_closure = |si: usize, inst: usize, fb: usize| -> u64 {
            let (_, grid) = &sources_pre_gather[si];
            let sel = apply_gather(grid[fb][inst], &gather);
            encode_source(&sel).unwrap_or(0)
        };
        let dst_closure = |si: usize, inst: usize, fb: usize| -> u64 {
            let q = &queues[si];
            let cell = q.cells[inst];
            let res_pins = result_pins(aig, q.kind, cell);
            let pin_iv = res_pins.get(fb).copied().unwrap_or(0);
            if pin_iv == 0 {
                return 0; // unconnected result bit -> padding word
            }
            let net = pin_iv >> 1;
            let sel = if let Some(&slot) = shared_slot.get(&net) {
                DestinationSel {
                    space: DestinationSpace::LocalShared,
                    index: slot / 64,
                    bit: (slot % 64) as u8,
                }
            } else if let Some(&(word, bit)) = state.prev.get(&net) {
                // A registered/output-state position already assigned by the
                // state layout. Unobserved outputs remain invalid padding; they
                // must never alias state word zero.
                DestinationSel {
                    space: DestinationSpace::CurrentStage,
                    index: word,
                    bit: bit as u8,
                }
            } else {
                return 0;
            };
            encode_destination(&sel).unwrap_or(0)
        };

        assemble_macro_program(
            &specs,
            1,
            1,
            sched.waves.len().max(1) as u32,
            gather.peak_shared_words,
            src_closure,
            dst_closure,
        )?
    };

    let queue_range = |kind: MacroKind, wave: usize| -> (u32, u32) {
        let Some(q) = queues.iter().find(|q| q.kind == kind) else {
            return (0, 0);
        };
        let start = q.waves.partition_point(|w| *w < wave);
        let end = q.waves.partition_point(|w| *w <= wave);
        (start as u32, (end - start) as u32)
    };
    let waves: Vec<WaveDescriptor> = (0..sched.waves.len())
        .map(|wave| {
            let aig_start = aig_operations.partition_point(|op| op.wave < wave);
            let aig_end = aig_operations.partition_point(|op| op.wave <= wave);
            let aig_depths = aig_operations[aig_start..aig_end]
                .iter()
                .map(|op| op.depth)
                .max()
                .map_or(0, |depth| depth + 1);
            let (carry4_start, carry4_count) = queue_range(MacroKind::Carry4, wave);
            let (dsp_start, dsp_count) = queue_range(MacroKind::Dsp48e2, wave);
            let (srlc_start, srlc_count) = queue_range(MacroKind::Srlc32e, wave);
            WaveDescriptor {
                aig_start: aig_start as u32,
                aig_count: (aig_end - aig_start) as u32,
                aig_depths,
                carry4_start,
                carry4_count,
                dsp_start,
                dsp_count,
                srlc_start,
                srlc_count,
            }
        })
        .collect();

    // Four words per AIG operation: depth plus three selectors. These use the same selector encoding as
    // macro sections and are decoded independently by CPU and CUDA.
    if !aig_operations.is_empty() {
        if program.len() % 2 != 0 {
            program.push(0);
        }
        let aig_start = program.len() as u64;
        for op in &aig_operations {
            program.push(u64::from(op.depth));
            program.push(encode_source(&op.a)?);
            program.push(encode_source(&op.b)?);
            program.push(encode_destination(&op.dst)?);
        }
        layout.sections.push(Section {
            kind: SectionKind::AigOperations,
            start: aig_start,
            end: program.len() as u64,
            align_bytes: 16,
        });
    }
    if !endpoint_operations.is_empty() {
        if program.len() % 2 != 0 {
            program.push(0);
        }
        let endpoint_start = program.len() as u64;
        for op in &endpoint_operations {
            program.push(op.kind as u64);
            program.push(encode_source(&op.data)?);
            program.push(encode_source(&op.enable)?);
            program.push(encode_destination(&op.dst)?);
        }
        layout.sections.push(Section {
            kind: SectionKind::EndpointOperations,
            start: endpoint_start,
            end: program.len() as u64,
            align_bytes: 16,
        });
    }
    if !sram_operations.is_empty() {
        if program.len() % 2 != 0 {
            program.push(0);
        }
        let sram_start = program.len() as u64;
        for op in &sram_operations {
            program.push(u64::from(op.storage_base));
            for source in &op.sources {
                program.push(encode_source(source)?);
            }
            for destination in &op.read_destinations {
                program.push(match destination {
                    Some(destination) => encode_destination(destination)?,
                    None => 0,
                });
            }
        }
        layout.sections.push(Section {
            kind: SectionKind::SramOperations,
            start: sram_start,
            end: program.len() as u64,
            align_bytes: 16,
        });
    }

    // The GPU needs the actual schedule, not only a wave count. Each wave is
    // five u64 words: AIG `(start,count)`, AIG depth count, then the three
    // macro `(start,count)` pairs.
    if program.len() % 2 != 0 {
        program.push(0);
    }
    let wave_start = program.len() as u64;
    for w in &waves {
        program.push(u64::from(w.aig_start) | (u64::from(w.aig_count) << 32));
        program.push(u64::from(w.aig_depths));
        program.push(u64::from(w.carry4_start) | (u64::from(w.carry4_count) << 32));
        program.push(u64::from(w.dsp_start) | (u64::from(w.dsp_count) << 32));
        program.push(u64::from(w.srlc_start) | (u64::from(w.srlc_count) << 32));
    }
    layout.sections.push(Section {
        kind: SectionKind::WaveDescriptors,
        start: wave_start,
        end: program.len() as u64,
        align_bytes: 16,
    });
    layout.header.total_words = program.len() as u64;
    layout.header.feature_flags |= 1;
    layout.header.content_hash = content_hash(&program, &layout.sections);
    let header = layout.header_words();
    program[..V2_HEADER_WORDS].copy_from_slice(&header);
    validate(&layout, program.len() as u64)?;

    let mut initial_state = vec![0u32; state.persistent_words as usize];
    for (instance, &word) in state.srlc_storage_word.iter().enumerate() {
        let cell = queues
            .iter()
            .find(|q| q.kind == MacroKind::Srlc32e)
            .and_then(|q| q.cells.get(instance))
            .copied()
            .expect("SRLC state word must have a dense queue cell");
        initial_state[word as usize] = aig.srlc32es[&cell].init;
    }

    let persistent_words = state.persistent_words;
    Ok(ResolvedProgram {
        program,
        layout,
        state,
        gather,
        shared_slot,
        queues,
        waves,
        aig_operations,
        endpoint_operations,
        sram_operations,
        sram_storage_words: (aig.srams.len() * AIGPDK_SRAM_SIZE) as u32,
        sources_pre_gather,
        initial_state,
        same_cycle_shared_words: same_cycle_words,
        current_stage_words: persistent_words,
        num_partitions: 1,
    })
}

fn globalize_source(sel: SourceSel, same_cycle_words: u32, stage_base: u32) -> SourceSel {
    if sel.space != SourceSpace::LocalShared || sel.index >= same_cycle_words {
        return sel;
    }
    SourceSel {
        space: SourceSpace::CurrentStage,
        index: stage_base + sel.index * 2 + u32::from(sel.bit >= 32),
        bit: sel.bit % 32,
        invert: sel.invert,
    }
}

fn globalize_destination(
    sel: DestinationSel,
    same_cycle_words: u32,
    stage_base: u32,
) -> DestinationSel {
    if sel.space != DestinationSpace::LocalShared || sel.index >= same_cycle_words {
        return sel;
    }
    DestinationSel {
        space: DestinationSpace::CurrentStage,
        index: stage_base + sel.index * 2 + u32::from(sel.bit >= 32),
        bit: sel.bit % 32,
    }
}

/// Convert the verified one-block program into the cross-block form. Every
/// same-cycle producer/consumer is routed through a packed global CurrentStage
/// arena, while gathered previous-state values remain block-local. A
/// cooperative grid barrier between dependency depths/waves makes publication
/// visible before any consumer runs.
pub fn build_partitioned_program(
    aig: &AIG,
    sched: &HeteroSchedule,
    placement: &HeteroPlacementV2,
) -> Result<ResolvedProgram, FormatError> {
    placement
        .validate(sched)
        .map_err(|_| FormatError::ArithmeticOverflow {
            what: "invalid heterogeneous placement",
        })?;
    let mut rp = build_resolved_program(aig, sched)?;
    let same = rp.same_cycle_shared_words;
    let stage_base = rp.state.persistent_words;

    let source_sections = [
        SectionKind::DspSourceSel,
        SectionKind::Carry4SourceSel,
        SectionKind::Srlc32eSourceSel,
    ];
    let destination_sections = [
        SectionKind::DspDestSel,
        SectionKind::Carry4DestSel,
        SectionKind::Srlc32eDestSel,
    ];
    for kind in source_sections {
        if let Some(section) = rp.layout.section(kind).copied() {
            for word in &mut rp.program[section.start as usize..section.end as usize] {
                if let Some(sel) = decode_source(*word)? {
                    *word = encode_source(&globalize_source(sel, same, stage_base))?;
                }
            }
        }
    }
    for kind in destination_sections {
        if let Some(section) = rp.layout.section(kind).copied() {
            for word in &mut rp.program[section.start as usize..section.end as usize] {
                if let Some(sel) = decode_destination(*word)? {
                    *word = encode_destination(&globalize_destination(sel, same, stage_base))?;
                }
            }
        }
    }
    if let Some(section) = rp.layout.section(SectionKind::AigOperations).copied() {
        for op in rp.program[section.start as usize..section.end as usize].chunks_exact_mut(4) {
            if let Some(sel) = decode_source(op[1])? {
                op[1] = encode_source(&globalize_source(sel, same, stage_base))?;
            }
            if let Some(sel) = decode_source(op[2])? {
                op[2] = encode_source(&globalize_source(sel, same, stage_base))?;
            }
            if let Some(sel) = decode_destination(op[3])? {
                op[3] = encode_destination(&globalize_destination(sel, same, stage_base))?;
            }
        }
    }
    if let Some(section) = rp.layout.section(SectionKind::EndpointOperations).copied() {
        for op in rp.program[section.start as usize..section.end as usize].chunks_exact_mut(4) {
            if let Some(sel) = decode_source(op[1])? {
                op[1] = encode_source(&globalize_source(sel, same, stage_base))?;
            }
            if let Some(sel) = decode_source(op[2])? {
                op[2] = encode_source(&globalize_source(sel, same, stage_base))?;
            }
        }
    }
    if let Some(section) = rp.layout.section(SectionKind::SramOperations).copied() {
        for op in rp.program[section.start as usize..section.end as usize]
            .chunks_exact_mut(SRAM_OPERATION_WORDS)
        {
            for word in &mut op[1..1 + SRAM_SOURCE_BITS] {
                if let Some(sel) = decode_source(*word)? {
                    *word = encode_source(&globalize_source(sel, same, stage_base))?;
                }
            }
        }
    }

    rp.num_partitions = placement.num_partitions;
    rp.current_stage_words = stage_base
        .checked_add(same.checked_mul(2).ok_or(FormatError::ArithmeticOverflow {
            what: "same-cycle CurrentStage words",
        })?)
        .ok_or(FormatError::ArithmeticOverflow {
            what: "CurrentStage words",
        })?;
    rp.layout.header.num_partitions = placement.num_partitions;
    rp.layout.header.feature_flags |= 1 << 1; // cooperative cross-block mode
    rp.layout.header.content_hash = content_hash(&rp.program, &rp.layout.sections);
    let header = rp.layout.header_words();
    rp.program[..V2_HEADER_WORDS].copy_from_slice(&header);
    validate(&rp.layout, rp.program.len() as u64)?;
    Ok(rp)
}

// ---- small dispatch helpers -----------------------------------------------

pub fn src_fields(k: MacroKind) -> &'static [SelField] {
    match k {
        MacroKind::Dsp48e2 => DSP_SRC_FIELDS,
        MacroKind::Carry4 => CARRY4_SRC_FIELDS,
        MacroKind::Srlc32e => SRLC_SRC_FIELDS,
    }
}
pub fn dst_fields(k: MacroKind) -> &'static [SelField] {
    match k {
        MacroKind::Dsp48e2 => DSP_DST_FIELDS,
        MacroKind::Carry4 => CARRY4_DST_FIELDS,
        MacroKind::Srlc32e => SRLC_DST_FIELDS,
    }
}
fn src_section_kind(k: MacroKind) -> SectionKind {
    match k {
        MacroKind::Dsp48e2 => SectionKind::DspSourceSel,
        MacroKind::Carry4 => SectionKind::Carry4SourceSel,
        MacroKind::Srlc32e => SectionKind::Srlc32eSourceSel,
    }
}
fn dst_section_kind(k: MacroKind) -> SectionKind {
    match k {
        MacroKind::Dsp48e2 => SectionKind::DspDestSel,
        MacroKind::Carry4 => SectionKind::Carry4DestSel,
        MacroKind::Srlc32e => SectionKind::Srlc32eDestSel,
    }
}
pub fn operand_pins(aig: &AIG, k: MacroKind, cell: usize) -> Vec<usize> {
    match k {
        MacroKind::Dsp48e2 => dsp_operand_pins(&aig.dsps[&cell]),
        MacroKind::Carry4 => carry4_operand_pins(&aig.carry4s[&cell]),
        MacroKind::Srlc32e => srlc_operand_pins(&aig.srlc32es[&cell]),
    }
}
pub fn result_pins(aig: &AIG, k: MacroKind, cell: usize) -> Vec<usize> {
    match k {
        MacroKind::Dsp48e2 => dsp_result_pins(&aig.dsps[&cell]),
        MacroKind::Carry4 => carry4_result_pins(&aig.carry4s[&cell]),
        MacroKind::Srlc32e => srlc_result_pins(&aig.srlc32es[&cell]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{Carry4Block, DSPBlock, DriverType, AIG};
    use crate::format_v2::{decode_selector_section, validate};
    use crate::schedule::build_schedule;

    fn base_aig() -> AIG {
        let mut aig = AIG::default();
        aig.num_aigpins = 1;
        aig.drivers = vec![DriverType::Tie0, DriverType::InputPort(0)];
        aig
    }
    fn iv(p: usize, i: usize) -> usize {
        p << 1 | i
    }
    fn add_carry4(
        aig: &mut AIG,
        cell: usize,
        di: [usize; 4],
        s: [usize; 4],
        cin_iv: usize,
    ) -> ([usize; 4], [usize; 4]) {
        let mut o_out = [0usize; 4];
        let mut co_out = [0usize; 4];
        for k in 0..4 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::CARRY4(cell, k));
            o_out[k] = aig.num_aigpins;
        }
        for k in 0..4 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::CARRY4(cell, k + 4));
            co_out[k] = aig.num_aigpins;
        }
        aig.carry4s.insert(
            cell,
            Carry4Block {
                di_iv: di,
                s_iv: s,
                cin_iv,
                cyinit_iv: 0,
                o_out,
                co_out,
            },
        );
        (o_out, co_out)
    }
    fn finish(mut aig: AIG) -> AIG {
        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        aig
    }

    #[test]
    fn primary_input_operand_resolves_to_previous_state() {
        let mut aig = base_aig();
        add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();
        // pre-gather: S[0] of carry 1 reads primary input pin 1 -> PreviousState.
        let (_, grid) = &rp.sources_pre_gather[0];
        assert_eq!(grid[0][0].space, SourceSpace::PreviousState);
        // CIN is constant 0.
        assert_eq!(grid[8][0].space, SourceSpace::Constant);
    }

    #[test]
    fn chained_carry_ci_resolves_to_local_shared_and_producer_gets_a_slot() {
        let mut aig = base_aig();
        let (_o1, co1) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0);
        add_carry4(&mut aig, 2, [iv(1, 0); 4], [iv(1, 0); 4], iv(co1[3], 0));
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        // carry 1's CO[3] net must have a shared slot.
        assert!(rp.shared_slot.contains_key(&co1[3]));
        // carry 2 is instance 1 of the CARRY4 queue; CIN is flat bit 8.
        let (_, grid) = &rp.sources_pre_gather[0];
        assert_eq!(grid[8][1].space, SourceSpace::LocalShared);
        let slot = rp.shared_slot[&co1[3]];
        assert_eq!(grid[8][1].index, slot / 64);
    }

    #[test]
    fn program_validates_and_round_trips_through_the_independent_decoder() {
        let mut aig = base_aig();
        let (_o1, co1) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0);
        add_carry4(&mut aig, 2, [iv(1, 0); 4], [iv(1, 0); 4], iv(co1[3], 0));
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        assert_eq!(validate(&rp.layout, rp.program.len() as u64), Ok(()));
        let ssec = rp.layout.section(SectionKind::Carry4SourceSel).unwrap();
        let decoded =
            decode_selector_section(&rp.program, ssec, CARRY4_SRC_FIELDS, 2, true).unwrap();
        // 10 flat bits (S4 DI4 CIN1 CYINIT1), 2 instances.
        assert_eq!(decoded.sources.len(), 10);
        // instance 1 CIN (flat 8) is LocalShared post-gather too.
        assert_eq!(
            decoded.sources[8][1].unwrap().space,
            SourceSpace::LocalShared
        );
        // instance 0 S[0] was PreviousState -> LocalShared after the gather.
        assert_eq!(
            decoded.sources[0][0].unwrap().space,
            SourceSpace::LocalShared
        );
        let waves = rp.layout.section(SectionKind::WaveDescriptors).unwrap();
        assert_eq!(waves.end - waves.start, 10); // two waves, five words each
        let w0 = rp.program[waves.start as usize + 2]; // CARRY4 descriptor
        let w1 = rp.program[waves.start as usize + 7];
        assert_eq!((w0 as u32, (w0 >> 32) as u32), (0, 1));
        assert_eq!((w1 as u32, (w1 >> 32) as u32), (1, 1));
    }

    #[test]
    fn gather_dedups_one_word_used_by_many_operands() {
        // 20 CARRY4s all reading primary input pin 1 for every S/DI bit.
        let mut aig = base_aig();
        for cell in 1..=20 {
            add_carry4(&mut aig, cell, [iv(1, 0); 4], [iv(1, 0); 4], 0);
        }
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();
        // pin 1 lives in old-state word 0; the gather stages exactly one word.
        assert_eq!(rp.gather.rounds.len(), 1);
        assert_eq!(rp.gather.rounds[0].words, vec![0]);
    }

    #[test]
    fn shared_slot_liveness_reuses_after_last_consumer() {
        // chain: c1 -> c2 -> c3 -> c4 (each CO[3] feeds the next CIN).
        let mut aig = base_aig();
        let (_o, mut co) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0);
        for cell in 2..=4 {
            let (_o2, co2) = add_carry4(&mut aig, cell, [iv(1, 0); 4], [iv(1, 0); 4], iv(co[3], 0));
            co = co2;
        }
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let (slots, peak_words) = allocate_shared_slots(&aig, &sched);
        // Three producer words are required over time. Two are simultaneously
        // reserved at a boundary: the current wave must finish every read of
        // c_i before the same storage may be reused for c_{i+1}. This is the
        // race-safe ping-pong requirement for parallel consumers.
        assert_eq!(slots.len(), 3);
        assert_eq!(peak_words, 2);
    }

    #[test]
    fn dsp_prev_p_resolves_to_previous_state_in_its_persistent_word() {
        let mut aig = base_aig();
        // one DSP whose C operand reads its own P[0] (MAC-style feedback via P).
        let mut p_out = [0usize; 48];
        for k in 0..48 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::DSP(1, k));
            p_out[k] = aig.num_aigpins;
        }
        let mut blk = DSPBlock::default();
        blk.a_iv = [iv(1, 0); 27];
        blk.d_iv = [iv(1, 0); 27];
        blk.b_iv = [iv(1, 0); 18];
        blk.c_iv = [0; 48];
        blk.c_iv[0] = iv(p_out[0], 0); // C[0] <- P[0] (old state)
        blk.p_out = p_out;
        aig.dsps.insert(1, blk);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();
        let (_, grid) = &rp.sources_pre_gather[0];
        // C is field index 3; flat bit base = 27+27+18 = 72. C[0] = flat 72.
        assert_eq!(grid[72][0].space, SourceSpace::PreviousState);
        assert_eq!(grid[72][0].index, rp.state.dsp_p_word[0]);
    }
}
