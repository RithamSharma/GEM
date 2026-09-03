// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Authoritative CPU interpreter of the V2 macro program (Host-Side Macro Memory
//! Formatter plan, Phase 6). Decodes the *identical* `u64` bytes the GPU will
//! read (via [`crate::format_v2::decode_selector_section`], not the encoder
//! helpers), runs the level preamble + type-homogeneous waves, and evaluates
//! every macro through [`crate::primitive_models`] -- the same fixed-width
//! models `cuda_test --check-with-cpu` already uses.
//!
//! ```text
//! logical AIG + schedule
//!   -> format_v2_build::build_resolved_program   (encoder)
//!   -> format_v2::decode_selector_section        (independent decoder)
//!   -> this interpreter                          (CPU V2)
//!   -> csrc/kernel_v2_impl.cuh                   (CUDA V2, uncompiled here)
//! ```
//!
//! The interpreter proves the ABI is self-consistent and that a same-cycle
//! macro chain (`CO[3] -> CI`) produces the correct value in one pass -- the
//! exact defect the batched V1 kernel has.

use crate::format_v2::{
    content_hash, decode_destination, decode_selector_section, decode_source, DestinationSel,
    DestinationSpace, SectionKind, SourceSel, SourceSpace,
};
use crate::format_v2_build::{
    dst_fields, src_fields, ResolvedProgram, SRAM_OPERATION_WORDS, SRAM_SOURCE_BITS,
};
use crate::primitive_models::{carry4, decode_dsp_controls, dsp48e2_next, srlc32e_step};
use crate::schedule::MacroKind;

/// Result of one simulated cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleResult {
    /// full persistent `u32` image after the cycle, ready to feed back as
    /// `prev` next cycle.
    pub next_state: Vec<u32>,
    /// per-CARRY4 `(O, CO)`, dense instance order.
    pub carry4_out: Vec<(u8, u8)>,
    /// per-DSP 48-bit `P_next`.
    pub dsp_p: Vec<u64>,
    /// per-SRLC `(Q, Q31, storage_next)`.
    pub srlc_out: Vec<(bool, bool, u32)>,
    /// `(AIG output pin, value)` in V2 execution order; primarily used by the
    /// independent mixed-path oracle and output-commit integration.
    pub aig_values: Vec<(usize, bool)>,
    /// primary-output pin-with-invert and its settled value for this cycle.
    pub primary_outputs: Vec<(usize, bool)>,
}

fn read_src(sel: &Option<SourceSel>, shared: &[u64], prev: &[u32], current_stage: &[u32]) -> bool {
    match sel {
        None => false,
        Some(s) => {
            let raw = match s.space {
                SourceSpace::Constant => false,
                SourceSpace::LocalShared => (shared[s.index as usize] >> s.bit) & 1 == 1,
                SourceSpace::PreviousState => (prev[s.index as usize] >> s.bit) & 1 == 1,
                SourceSpace::CurrentStage => (current_stage[s.index as usize] >> s.bit) & 1 == 1,
            };
            raw ^ s.invert
        }
    }
}

fn read_endpoint_src(sel: &Option<SourceSel>, shared: &[u64], current_stage: &[u32]) -> bool {
    match sel {
        Some(s) if s.space == SourceSpace::CurrentStage => {
            (((current_stage[s.index as usize] >> s.bit) & 1) != 0) ^ s.invert
        }
        _ => read_src(sel, shared, current_stage, current_stage),
    }
}

fn pack(get: &[bool], lo: usize, hi: usize) -> u64 {
    let mut v = 0u64;
    for (i, b) in (lo..hi).enumerate() {
        if get[b] {
            v |= 1u64 << i;
        }
    }
    v
}

fn write_result(
    dst: &[Vec<Option<DestinationSel>>],
    inst: usize,
    bits: &[(usize, bool)],
    shared: &mut [u64],
    current_stage: &mut [u32],
    next_state: &mut [u32],
) {
    for &(flat_bit, val) in bits {
        let Some(sel) = dst
            .get(flat_bit)
            .and_then(|row| row.get(inst))
            .and_then(|o| *o)
        else {
            continue;
        };
        match sel.space {
            DestinationSpace::LocalShared => {
                let w = &mut shared[sel.index as usize];
                if val {
                    *w |= 1u64 << sel.bit;
                } else {
                    *w &= !(1u64 << sel.bit);
                }
            }
            DestinationSpace::CurrentStage | DestinationSpace::NextState => {
                let base = if sel.space == DestinationSpace::CurrentStage {
                    &mut *current_stage
                } else {
                    &mut *next_state
                };
                let w = &mut base[sel.index as usize];
                if val {
                    *w |= 1u32 << sel.bit;
                } else {
                    *w &= !(1u32 << sel.bit);
                }
            }
        }
    }
}

fn section_kinds(k: MacroKind) -> (SectionKind, SectionKind) {
    match k {
        MacroKind::Dsp48e2 => (SectionKind::DspSourceSel, SectionKind::DspDestSel),
        MacroKind::Carry4 => (SectionKind::Carry4SourceSel, SectionKind::Carry4DestSel),
        MacroKind::Srlc32e => (SectionKind::Srlc32eSourceSel, SectionKind::Srlc32eDestSel),
    }
}

struct QueueDecoded {
    kind: MacroKind,
    cells: Vec<usize>,
    waves: Vec<usize>,
    src: Vec<Vec<Option<SourceSel>>>,
    dst: Vec<Vec<Option<DestinationSel>>>,
}

/// Evaluate one cycle of the resolved program.
///
/// * `prev` -- persistent `u32` image from last cycle (length
///   `rp.state.persistent_words`).
/// * `rising_edge` -- the single global clock edge for this cycle.
pub fn interpret_cycle(
    rp: &ResolvedProgram,
    prev: &[u32],
    rising_edge: bool,
) -> Result<CycleResult, String> {
    let mut sram_storage = vec![0u32; rp.sram_storage_words as usize];
    interpret_cycle_with_sram(rp, prev, rising_edge, &mut sram_storage)
}

pub fn interpret_cycle_with_sram(
    rp: &ResolvedProgram,
    prev: &[u32],
    rising_edge: bool,
    sram_storage: &mut [u32],
) -> Result<CycleResult, String> {
    if prev.len() != rp.state.persistent_words as usize {
        return Err(format!(
            "prev image is {} words, expected {}",
            prev.len(),
            rp.state.persistent_words
        ));
    }
    // ABI self-check: the program the interpreter is about to read must pass the
    // same validator the GPU loader runs.
    crate::format_v2::validate(&rp.layout, rp.program.len() as u64)
        .map_err(|e| format!("{e:?}"))?;
    if rp.layout.header.content_hash != content_hash(&rp.program, &rp.layout.sections) {
        return Err("content hash mismatch".into());
    }

    let shared_words = rp.layout.header.shared_words_per_block as usize;
    let mut shared = vec![0u64; shared_words.max(1)];

    // ---- level preamble: stage the gathered old-state words ----
    for (&orig_word, &shared_word) in &rp.gather.staged_word_slot {
        shared[shared_word as usize] = u64::from(prev[orig_word as usize]);
    }

    // decode every selector section once, straight from the bytes.
    let mut qd: Vec<QueueDecoded> = Vec::new();
    for q in &rp.queues {
        let (skind, dkind) = section_kinds(q.kind);
        let ssec = rp
            .layout
            .section(skind)
            .ok_or_else(|| format!("missing source section for {:?}", q.kind))?;
        let dsec = rp
            .layout
            .section(dkind)
            .ok_or_else(|| format!("missing dest section for {:?}", q.kind))?;
        let n = q.cells.len();
        let sd = decode_selector_section(&rp.program, ssec, src_fields(q.kind), n, true)
            .map_err(|e| format!("{e:?}"))?;
        let dd = decode_selector_section(&rp.program, dsec, dst_fields(q.kind), n, false)
            .map_err(|e| format!("{e:?}"))?;
        qd.push(QueueDecoded {
            kind: q.kind,
            cells: q.cells.clone(),
            waves: q.waves.clone(),
            src: sd.sources,
            dst: dd.dests,
        });
    }

    let mut next_state = prev.to_vec();
    let mut current_stage = vec![0u32; rp.current_stage_words.max(1) as usize];
    current_stage[..prev.len()].copy_from_slice(prev);
    let mut carry4_by_inst: Vec<(u8, u8)> = Vec::new();
    let mut dsp_by_inst: Vec<u64> = Vec::new();
    let mut srlc_by_inst: Vec<(bool, bool, u32)> = Vec::new();
    let mut aig_values: Vec<(usize, bool)> = rp
        .aig_operations
        .iter()
        .map(|op| (op.output_pin, false))
        .collect();

    let num_waves = rp.layout.header.num_waves as usize;

    for wave in 0..num_waves {
        if let Some(section) = rp.layout.section(SectionKind::AigOperations) {
            let desc = rp.waves[wave];
            for depth in 0..desc.aig_depths {
                for op_index in desc.aig_start..desc.aig_start + desc.aig_count {
                    let base = section.start as usize + op_index as usize * 4;
                    if rp.program[base] as u32 != depth {
                        continue;
                    }
                    let a = decode_source(rp.program[base + 1]).map_err(|e| format!("{e:?}"))?;
                    let b = decode_source(rp.program[base + 2]).map_err(|e| format!("{e:?}"))?;
                    let dst =
                        decode_destination(rp.program[base + 3]).map_err(|e| format!("{e:?}"))?;
                    let value = read_src(&a, &shared, prev, &current_stage)
                        && read_src(&b, &shared, prev, &current_stage);
                    if let Some(d) = dst {
                        match d.space {
                            DestinationSpace::LocalShared => {
                                let word = &mut shared[d.index as usize];
                                if value {
                                    *word |= 1u64 << d.bit;
                                } else {
                                    *word &= !(1u64 << d.bit);
                                }
                            }
                            DestinationSpace::CurrentStage | DestinationSpace::NextState => {
                                let base = if d.space == DestinationSpace::CurrentStage {
                                    &mut current_stage
                                } else {
                                    &mut next_state
                                };
                                let word = &mut base[d.index as usize];
                                if value {
                                    *word |= 1u32 << d.bit;
                                } else {
                                    *word &= !(1u32 << d.bit);
                                }
                            }
                        }
                    }
                    aig_values[op_index as usize].1 = value;
                }
            }
        }
        for q in &qd {
            for inst in 0..q.cells.len() {
                if q.waves[inst] != wave {
                    continue;
                }
                // read all operand bits, releasing the &shared borrow.
                let get: Vec<bool> = (0..q.src.len())
                    .map(|fb| read_src(&q.src[fb][inst], &shared, prev, &current_stage))
                    .collect();

                match q.kind {
                    MacroKind::Carry4 => {
                        let s = pack(&get, 0, 4) as u8;
                        let di = pack(&get, 4, 8) as u8;
                        let ci = get[8];
                        let cyinit = get[9];
                        let r = carry4(s, di, ci, cyinit);
                        let bits: Vec<(usize, bool)> = (0..4)
                            .map(|k| (k, (r.o >> k) & 1 != 0))
                            .chain((0..4).map(|k| (4 + k, (r.co >> k) & 1 != 0)))
                            .collect();
                        write_result(
                            &q.dst,
                            inst,
                            &bits,
                            &mut shared,
                            &mut current_stage,
                            &mut next_state,
                        );
                        ensure_len(&mut carry4_by_inst, inst, (0, 0));
                        carry4_by_inst[inst] = (r.o, r.co);
                    }
                    MacroKind::Srlc32e => {
                        let d = get[0];
                        let ce = get[1];
                        let a = pack(&get, 2, 7) as u8;
                        let clock_edge = get[7];
                        let storage_word = rp.state.srlc_storage_word[inst] as usize;
                        let cur = prev[storage_word];
                        let (outs, nxt) = srlc32e_step(cur, d, ce, clock_edge, a);
                        write_result(
                            &q.dst,
                            inst,
                            &[(0, outs.q), (1, outs.q31)],
                            &mut shared,
                            &mut current_stage,
                            &mut next_state,
                        );
                        next_state[storage_word] = nxt;
                        ensure_len(&mut srlc_by_inst, inst, (false, false, 0));
                        srlc_by_inst[inst] = (outs.q, outs.q31, nxt);
                    }
                    MacroKind::Dsp48e2 => {
                        let a = pack(&get, 0, 27) as u32;
                        let d = pack(&get, 27, 54) as u32;
                        let b = pack(&get, 54, 72) as u32;
                        let c = pack(&get, 72, 120);
                        let opmode = pack(&get, 120, 129) as u16;
                        let alumode = pack(&get, 129, 133) as u8;
                        let inmode = pack(&get, 133, 138) as u8;
                        let cep = get[138];
                        let rstp = get[139];
                        let clock_edge = get[140];
                        let pw = rp.state.dsp_p_word[inst] as usize;
                        let p_cur = u64::from(prev[pw]) | (u64::from(prev[pw + 1]) << 32);
                        let (mode, preadd) = decode_dsp_controls(opmode, alumode, inmode)
                            .map_err(|e| format!("DSP {inst} unsupported controls: {e:?}"))?;
                        let next_p = if !clock_edge {
                            p_cur & ((1u64 << 48) - 1)
                        } else if rstp {
                            0
                        } else if !cep {
                            p_cur & ((1u64 << 48) - 1)
                        } else {
                            dsp48e2_next(a, b, c, d, p_cur, mode, preadd)
                        };
                        let bits: Vec<(usize, bool)> =
                            (0..48).map(|k| (k, (next_p >> k) & 1 != 0)).collect();
                        write_result(
                            &q.dst,
                            inst,
                            &bits,
                            &mut shared,
                            &mut current_stage,
                            &mut next_state,
                        );
                        next_state[pw] = (next_p & 0xFFFF_FFFF) as u32;
                        next_state[pw + 1] = (next_p >> 32) as u32;
                        ensure_len(&mut dsp_by_inst, inst, 0u64);
                        dsp_by_inst[inst] = next_p;
                    }
                }
            }
        }
    }

    let mut primary_outputs = Vec::new();
    if let Some(section) = rp.layout.section(SectionKind::SramOperations) {
        if sram_storage.len() != rp.sram_storage_words as usize {
            return Err(format!(
                "SRAM image is {} words, expected {}",
                sram_storage.len(),
                rp.sram_storage_words
            ));
        }
        for (instance, op) in rp.program[section.start as usize..section.end as usize]
            .chunks_exact(SRAM_OPERATION_WORDS)
            .enumerate()
        {
            let storage_base = op[0] as usize;
            let sources: Result<Vec<_>, _> = op[1..1 + SRAM_SOURCE_BITS]
                .iter()
                .map(|&word| decode_source(word).map_err(|e| format!("{e:?}")))
                .collect();
            let sources = sources?;
            let get = |index: usize| read_src(&sources[index], &shared, prev, &current_stage);
            let read_enable = get(0);
            let mut read_address = 0usize;
            let mut write_address = 0usize;
            for bit in 0..13 {
                read_address |= usize::from(get(1 + bit)) << bit;
                write_address |= usize::from(get(14 + bit)) << bit;
            }
            let old_read = sram_storage[storage_base + read_address];
            let old_write = sram_storage[storage_base + write_address];
            let mut write_mask = 0u32;
            let mut write_data = 0u32;
            for bit in 0..32 {
                write_mask |= u32::from(get(27 + bit)) << bit;
                write_data |= u32::from(get(59 + bit)) << bit;
            }
            if read_enable {
                for bit in 0..32 {
                    if let Some(dst) = decode_destination(op[1 + SRAM_SOURCE_BITS + bit])
                        .map_err(|e| format!("{e:?}"))?
                    {
                        let word = &mut next_state[dst.index as usize];
                        let mask = 1u32 << dst.bit;
                        if (old_read >> bit) & 1 != 0 {
                            *word |= mask;
                        } else {
                            *word &= !mask;
                        }
                    }
                }
            }
            sram_storage[storage_base + write_address] =
                (old_write & !write_mask) | (write_data & write_mask);
            debug_assert_eq!(storage_base, instance * (1 << 13));
        }
    }
    if let Some(section) = rp.layout.section(SectionKind::EndpointOperations) {
        for (op_index, op) in rp.endpoint_operations.iter().enumerate() {
            let base = section.start as usize + op_index * 4;
            let kind = rp.program[base];
            let data = decode_source(rp.program[base + 1]).map_err(|e| format!("{e:?}"))?;
            let enable = decode_source(rp.program[base + 2]).map_err(|e| format!("{e:?}"))?;
            let dst = decode_destination(rp.program[base + 3]).map_err(|e| format!("{e:?}"))?;
            let value = read_endpoint_src(&data, &shared, &current_stage);
            if read_endpoint_src(&enable, &shared, &current_stage) {
                if let Some(d) = dst {
                    let word = &mut next_state[d.index as usize];
                    if value {
                        *word |= 1u32 << d.bit;
                    } else {
                        *word &= !(1u32 << d.bit);
                    }
                }
            }
            if kind == crate::format_v2_build::EndpointKind::PrimaryOutput as u64 {
                primary_outputs.push((op.logical_pin, value));
            }
        }
    }

    let _ = rising_edge; // retained in the public API for compatibility.

    Ok(CycleResult {
        next_state,
        carry4_out: carry4_by_inst,
        dsp_p: dsp_by_inst,
        srlc_out: srlc_by_inst,
        aig_values,
        primary_outputs,
    })
}

fn ensure_len<T: Clone>(v: &mut Vec<T>, idx: usize, fill: T) {
    if v.len() <= idx {
        v.resize(idx + 1, fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{Carry4Block, DriverType, RAMBlock, Srlc32eBlock, AIG, DFF};
    use crate::format_v2_build::{build_partitioned_program, build_resolved_program};
    use crate::hetero_parts::HeteroPlacementV2;
    use crate::primitive_models::{carry4 as ref_carry4, srlc32e_step as ref_srl};
    use crate::schedule::build_schedule;

    fn base_aig() -> AIG {
        let mut aig = AIG::default();
        aig.num_aigpins = 2;
        aig.drivers = vec![
            DriverType::Tie0,
            DriverType::InputPort(0),
            DriverType::InputPort(1),
        ];
        aig
    }
    fn iv(p: usize, i: usize) -> usize {
        p << 1 | i
    }
    fn add_carry4(
        aig: &mut AIG,
        cell: usize,
        s: [usize; 4],
        di: [usize; 4],
        cin_iv: usize,
        cyinit_iv: usize,
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
                cyinit_iv,
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

    fn add_and(aig: &mut AIG, a_iv: usize, b_iv: usize) -> usize {
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::AndGate(a_iv, b_iv));
        aig.num_aigpins
    }

    #[test]
    fn single_carry4_matches_the_reference_model() {
        let mut aig = base_aig();
        add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(2, 0); 4], 0, 0);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        for x in [false, true] {
            for y in [false, true] {
                let mut prev = vec![0u32; rp.state.persistent_words as usize];
                let (w1, b1) = rp.state.prev[&1];
                let (w2, b2) = rp.state.prev[&2];
                if x {
                    prev[w1 as usize] |= 1 << b1;
                }
                if y {
                    prev[w2 as usize] |= 1 << b2;
                }
                let got = interpret_cycle(&rp, &prev, true).unwrap();
                let s = if x { 0b1111 } else { 0 };
                let di = if y { 0b1111 } else { 0 };
                let want = ref_carry4(s, di, false, false);
                assert_eq!(got.carry4_out[0], (want.o, want.co), "x={x} y={y}");
            }
        }
    }

    #[test]
    fn chained_carry4_is_correct_in_one_pass() {
        let mut aig = base_aig();
        let (_o1, co1) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0, 0);
        add_carry4(&mut aig, 2, [iv(1, 0); 4], [iv(1, 0); 4], iv(co1[3], 0), 0);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        for x in [false, true] {
            let mut prev = vec![0u32; rp.state.persistent_words as usize];
            let (w1, b1) = rp.state.prev[&1];
            if x {
                prev[w1 as usize] |= 1 << b1;
            }
            let got = interpret_cycle(&rp, &prev, true).unwrap();

            let s = if x { 0b1111 } else { 0 };
            let c1 = ref_carry4(s, s, false, false);
            let c1_co3 = (c1.co >> 3) & 1 != 0;
            let c2 = ref_carry4(s, s, c1_co3, false);
            assert_eq!(got.carry4_out[0], (c1.o, c1.co), "c1 x={x}");
            assert_eq!(got.carry4_out[1], (c2.o, c2.co), "c2 chained x={x}");
        }
    }

    #[test]
    fn chained_carry4_cross_partition_uses_current_stage() {
        let mut aig = base_aig();
        let (_o1, co1) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0, 0);
        add_carry4(&mut aig, 2, [iv(1, 0); 4], [iv(1, 0); 4], iv(co1[3], 0), 0);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let placement = HeteroPlacementV2::build(&sched, 2).unwrap();
        let rp = build_partitioned_program(&aig, &sched, &placement).unwrap();
        assert_eq!(rp.num_partitions, 2);
        assert!(rp.current_stage_words > rp.state.persistent_words);

        let mut prev = vec![0u32; rp.state.persistent_words as usize];
        let (word, bit) = rp.state.prev[&1];
        prev[word as usize] |= 1 << bit;
        let got = interpret_cycle(&rp, &prev, true).unwrap();
        let first = ref_carry4(0b1111, 0b1111, false, false);
        let second = ref_carry4(0b1111, 0b1111, first.co & 8 != 0, false);
        assert_eq!(got.carry4_out[1], (second.o, second.co));
    }

    #[test]
    fn synchronous_sram_reads_old_word_then_commits_masked_write() {
        let mut aig = base_aig();
        let mut ram = RAMBlock::default();
        ram.port_r_en_iv = iv(1, 0);
        ram.port_w_wr_en_iv = [iv(1, 0); 32];
        ram.port_w_wr_data_iv = [iv(2, 0); 32];
        for bit in 0..32 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::SRAM(77));
            ram.port_r_rd_data[bit] = aig.num_aigpins;
        }
        aig.srams.insert(77, ram);
        let aig = finish(aig);
        let schedule = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &schedule).unwrap();
        let mut storage = vec![0u32; rp.sram_storage_words as usize];
        let mut prev = vec![0u32; rp.state.persistent_words as usize];
        for pin in [1usize, 2] {
            let (word, bit) = rp.state.prev[&pin];
            prev[word as usize] |= 1 << bit;
        }

        let first = interpret_cycle_with_sram(&rp, &prev, true, &mut storage).unwrap();
        assert_eq!(storage[0], u32::MAX);
        for pin in &aig.srams[&77].port_r_rd_data {
            let (word, bit) = rp.state.prev[pin];
            assert_eq!((first.next_state[word as usize] >> bit) & 1, 0);
        }

        let mut second_prev = first.next_state;
        let (data_word, data_bit) = rp.state.prev[&2];
        second_prev[data_word as usize] &= !(1 << data_bit);
        let second = interpret_cycle_with_sram(&rp, &second_prev, true, &mut storage).unwrap();
        assert_eq!(storage[0], 0);
        for pin in &aig.srams[&77].port_r_rd_data {
            let (word, bit) = rp.state.prev[pin];
            assert_eq!((second.next_state[word as usize] >> bit) & 1, 1);
        }
    }

    #[test]
    fn aig_to_carry_dependency_executes_in_the_same_v2_schedule() {
        // g = x & !y; every S bit of CARRY4 consumes g.  This test must fail
        // until AIG regions are serialized and executed by the V2 interpreter.
        let mut aig = base_aig();
        let g = add_and(&mut aig, iv(1, 0), iv(2, 1));
        add_carry4(&mut aig, 1, [iv(g, 0); 4], [0; 4], 0, 1);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        assert_eq!(
            sched
                .waves
                .iter()
                .map(|w| w.aig_regions.len())
                .sum::<usize>(),
            1
        );
        let rp = build_resolved_program(&aig, &sched).unwrap();

        for (x, y) in [(false, false), (true, false), (true, true)] {
            let mut prev = vec![0u32; rp.state.persistent_words as usize];
            for (pin, value) in [(1usize, x), (2usize, y)] {
                let (word, bit) = rp.state.prev[&pin];
                if value {
                    prev[word as usize] |= 1 << bit;
                }
            }
            let got = interpret_cycle(&rp, &prev, true).unwrap();
            let g_value = x && !y;
            let expected = ref_carry4(if g_value { 0b1111 } else { 0 }, 0, false, true);
            assert_eq!(got.carry4_out[0], (expected.o, expected.co), "x={x} y={y}");
        }
    }

    #[test]
    fn aig_carry_aig_chain_executes_all_three_dependency_waves() {
        let mut aig = base_aig();
        let g1 = add_and(&mut aig, iv(1, 0), iv(2, 1));
        let (o, _) = add_carry4(&mut aig, 1, [iv(g1, 0); 4], [0; 4], 0, 1);
        let g2 = add_and(&mut aig, iv(o[0], 0), iv(1, 0));
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();
        assert_eq!(sched.num_levels, 3);

        for (x, y) in [(false, false), (true, false), (true, true)] {
            let mut prev = vec![0u32; rp.state.persistent_words as usize];
            for (pin, value) in [(1usize, x), (2usize, y)] {
                let (word, bit) = rp.state.prev[&pin];
                if value {
                    prev[word as usize] |= 1 << bit;
                }
            }
            let got = interpret_cycle(&rp, &prev, true).unwrap();
            let first = x && !y;
            let carry = ref_carry4(if first { 0b1111 } else { 0 }, 0, false, true);
            let expected_g2 = (carry.o & 1 != 0) && x;
            assert_eq!(
                got.aig_values
                    .iter()
                    .find(|(pin, _)| *pin == g2)
                    .map(|(_, value)| *value),
                Some(expected_g2),
                "x={x} y={y}"
            );
        }
    }

    #[test]
    fn settled_aig_value_commits_to_primary_output_and_enabled_dff() {
        let mut aig = base_aig();
        let g = add_and(&mut aig, iv(1, 0), iv(2, 1));
        aig.primary_outputs.insert(iv(g, 0));
        aig.num_aigpins += 1;
        let q = aig.num_aigpins;
        aig.drivers.push(DriverType::DFF(7));
        aig.dffs.insert(
            7,
            DFF {
                d_iv: iv(g, 0),
                en_iv: iv(1, 0),
                q,
            },
        );
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        for (x, y, expected) in [
            (false, false, false),
            (true, false, true),
            (true, true, false),
        ] {
            let mut prev = vec![0u32; rp.state.persistent_words as usize];
            for (pin, value) in [(1usize, x), (2usize, y)] {
                let (word, bit) = rp.state.prev[&pin];
                if value {
                    prev[word as usize] |= 1 << bit;
                }
            }
            let got = interpret_cycle(&rp, &prev, true).unwrap();
            assert_eq!(got.primary_outputs, vec![(iv(g, 0), expected)]);
            let (qword, qbit) = rp.state.prev[&q];
            assert_eq!(
                (got.next_state[qword as usize] >> qbit) & 1 != 0,
                expected && x
            );
            let (oword, obit) = rp.state.outputs[&iv(g, 0)];
            assert_eq!((got.next_state[oword as usize] >> obit) & 1 != 0, expected);
        }
    }

    #[test]
    fn srlc32e_shifts_and_taps_settle_after_the_edge() {
        let mut aig = base_aig();
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(9, 0));
        let q = aig.num_aigpins;
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(9, 1));
        let q31 = aig.num_aigpins;
        aig.srlc32es.insert(
            9,
            Srlc32eBlock {
                d_iv: iv(1, 0),
                ce_iv: iv(2, 0),
                a_iv: [0; 5],
                clk_iv: 1,
                q_out: q,
                q31_out: q31,
                init: 0,
            },
        );
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        let mut prev = vec![0u32; rp.state.persistent_words as usize];
        let sw = rp.state.srlc_storage_word[0] as usize;
        prev[sw] = 0x8000_0001;
        let (w2, b2) = rp.state.prev[&2];
        prev[w2 as usize] |= 1 << b2; // CE = 1
        let got = interpret_cycle(&rp, &prev, true).unwrap();

        let (refouts, refnext) = ref_srl(0x8000_0001, false, true, true, 0);
        assert_eq!(got.srlc_out[0], (refouts.q, refouts.q31, refnext));
        assert_eq!(got.next_state[sw], refnext);
    }

    #[test]
    fn dsp_multiply_only_matches_the_reference_model() {
        use crate::aig::DSPBlock;
        use crate::primitive_models::{dsp48e2_next, DspMode};
        let mut aig = base_aig();
        let mut p_out = [0usize; 48];
        for k in 0..48 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::DSP(4, k));
            p_out[k] = aig.num_aigpins;
        }
        let mut dsp = DSPBlock::default();
        dsp.clk_iv = 1;
        // A = all-x, D = 0, B = all-y, C = 0, OPMODE = 9'h005 (bits 0 and 2).
        dsp.a_iv = [iv(1, 0); 27];
        dsp.d_iv = [0; 27];
        dsp.b_iv = [iv(2, 0); 18];
        dsp.c_iv = [0; 48];
        dsp.opmode_iv = [1, 0, 1, 0, 0, 0, 0, 0, 0]; // 1 == const-1 pin_iv
        dsp.p_out = p_out;
        aig.dsps.insert(4, dsp);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();

        let mut prev = vec![0u32; rp.state.persistent_words as usize];
        let (w1, b1) = rp.state.prev[&1];
        let (w2, b2) = rp.state.prev[&2];
        prev[w1 as usize] |= 1 << b1; // x = 1  -> A = 0x07FF_FFFF (27 ones)
        prev[w2 as usize] |= 1 << b2; // y = 1  -> B = 0x0003_FFFF (18 ones)
        let got = interpret_cycle(&rp, &prev, true).unwrap();

        let want = dsp48e2_next(0x07FF_FFFF, 0x0003_FFFF, 0, 0, 0, DspMode::Multiply, false);
        assert_eq!(got.dsp_p[0], want);
        // P_next also committed to the persistent image.
        let pw = rp.state.dsp_p_word[0] as usize;
        assert_eq!(
            u64::from(got.next_state[pw]) | (u64::from(got.next_state[pw + 1]) << 32),
            want
        );
    }

    #[test]
    fn all_three_macro_types_in_one_program_validate_and_interpret() {
        // the exact shape src/bin/formatter_gpu_test.rs formats.
        let mut aig = base_aig();
        let (_o1, co1) = add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(2, 0); 4], 0, 0);
        add_carry4(&mut aig, 2, [iv(1, 0); 4], [iv(2, 0); 4], iv(co1[3], 0), 0);
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(3, 0));
        let q = aig.num_aigpins;
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(3, 1));
        let q31 = aig.num_aigpins;
        aig.srlc32es.insert(
            3,
            Srlc32eBlock {
                d_iv: iv(1, 0),
                ce_iv: iv(2, 0),
                a_iv: [0; 5],
                clk_iv: 0,
                q_out: q,
                q31_out: q31,
                init: 0,
            },
        );
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let rp = build_resolved_program(&aig, &sched).unwrap();
        assert_eq!(
            crate::format_v2::validate(&rp.layout, rp.program.len() as u64),
            Ok(())
        );
        let prev = vec![0u32; rp.state.persistent_words as usize];
        let got = interpret_cycle(&rp, &prev, true).unwrap();
        assert_eq!(got.carry4_out.len(), 2);
        assert_eq!(got.srlc_out.len(), 1);
    }

    #[test]
    fn abi_self_check_rejects_a_corrupt_header() {
        let mut aig = base_aig();
        add_carry4(&mut aig, 1, [iv(1, 0); 4], [iv(1, 0); 4], 0, 0);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let mut rp = build_resolved_program(&aig, &sched).unwrap();
        rp.layout.header.magic = 0;
        let prev = vec![0u32; rp.state.persistent_words as usize];
        assert!(interpret_cycle(&rp, &prev, true).is_err());
    }
}
