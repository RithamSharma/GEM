// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Versioned, sectioned, 64-bit heterogeneous-program ABI.
//!
//! This module owns the pure, host-side ABI primitives. Resolution, shared-slot
//! allocation, upload, and execution are implemented by the neighbouring V2
//! modules and the CUDA V2 kernel.
//!
//! * **Phase 1** -- typed [`SourceSel`] / [`DestinationSel`] with one documented
//!   64-bit device encoding using named masks and checked conversions, plus an
//!   independent decoder.
//! * **Phase 2** -- [`ScriptV2Header`], [`Section`], [`ProgramLayoutV2`],
//!   monotonic / non-overlapping / aligned / in-bounds [`validate`], and a
//!   golden header.
//! * **Phase 3 address math** -- [`transpose_selectors`] lays a selector stream
//!   out field/bit-major with `padded_count = round_up(n, 32)` so lanes
//!   `0..32` at a fixed (field, bit) read consecutive `u64` words.
//! * **Phase 6 (host half)** -- [`decode_program`] reconstructs the logical
//!   selector records from raw bytes *without* calling the encoder's helpers.
//!
//! See [`crate::format_v2_build`], [`crate::format_v2_gpu`], and
//! `csrc/kernel_v2.cu` for the integrated path.

use std::collections::BTreeSet;

// ===========================================================================
// Phase 1 -- selectors and their 64-bit encoding
// ===========================================================================

/// Where a macro operand bit is read from, resolved from the scheduled edge:
/// constant edge -> [`Constant`](SourceSpace::Constant); primary-input or
/// registered old-state edge -> [`PreviousState`](SourceSpace::PreviousState);
/// producer in an earlier major stage -> [`CurrentStage`](SourceSpace::CurrentStage);
/// producer in an earlier local wave -> [`LocalShared`](SourceSpace::LocalShared).
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum SourceSpace {
    Constant = 0,
    PreviousState = 1,
    CurrentStage = 2,
    LocalShared = 3,
}

/// Where a macro result bit is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum DestinationSpace {
    LocalShared = 0,
    CurrentStage = 1,
    NextState = 2,
    // 3 is reserved.
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SourceSel {
    pub space: SourceSpace,
    /// state-word / shared-slot / stage-word index.
    pub index: u32,
    /// bit within that word, `0..=63`.
    pub bit: u8,
    pub invert: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DestinationSel {
    pub space: DestinationSpace,
    pub index: u32,
    pub bit: u8,
}

/// A constant source: the literal is `invert` (const 0, or its inverse).
impl SourceSel {
    pub const CONST0: SourceSel = SourceSel {
        space: SourceSpace::Constant,
        index: 0,
        bit: 0,
        invert: false,
    };
    pub const CONST1: SourceSel = SourceSel {
        space: SourceSpace::Constant,
        index: 0,
        bit: 0,
        invert: true,
    };
}

// ---- bit-field layout of the 64-bit selector word (shared with CUDA) ----
//   bits  0..=31 : index
//   bits 32..=37 : bit (6 bits, 0..=63)
//   bits 38..=39 : space
//   bit      40  : invert
//   bit      41  : valid
//   bits 42..=63 : reserved, must be zero
pub const SEL_INDEX_SHIFT: u32 = 0;
pub const SEL_INDEX_MASK: u64 = 0x0000_0000_FFFF_FFFF;
pub const SEL_BIT_SHIFT: u32 = 32;
pub const SEL_BIT_MASK: u64 = 0x0000_003F_0000_0000;
pub const SEL_SPACE_SHIFT: u32 = 38;
pub const SEL_SPACE_MASK: u64 = 0x0000_00C0_0000_0000;
pub const SEL_INVERT_BIT: u64 = 1 << 40;
pub const SEL_VALID_BIT: u64 = 1 << 41;
pub const SEL_RESERVED_MASK: u64 = 0xFFFF_FC00_0000_0000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    BitIndexOutOfRange {
        bit: u8,
    },
    ReservedBitsSet {
        word: u64,
    },
    InvalidSourceSpace {
        raw: u8,
    },
    InvalidDestinationSpace {
        raw: u8,
    },
    SelectorNotValid {
        word: u64,
    },
    BadMagic {
        got: u64,
    },
    UnsupportedVersion {
        got: u32,
    },
    SectionMisaligned {
        section: usize,
        start_byte: u64,
        align: u64,
    },
    SectionNotMonotonic {
        section: usize,
        start: u64,
        prev_end: u64,
    },
    SectionOutOfBounds {
        section: usize,
        end: u64,
        total: u64,
    },
    ArithmeticOverflow {
        what: &'static str,
    },
    NodeScheduledTwice {
        node: usize,
    },
    SameCycleEdgeMisordered {
        producer_wave: u32,
        consumer_wave: u32,
    },
    SharedBudgetExceeded {
        needed: u32,
        limit: u32,
    },
    PaddingNotZero {
        word_index: usize,
        word: u64,
    },
}

fn checked_bit(bit: u8) -> Result<u64, FormatError> {
    if bit > 63 {
        return Err(FormatError::BitIndexOutOfRange { bit });
    }
    Ok((u64::from(bit) << SEL_BIT_SHIFT) & SEL_BIT_MASK)
}

/// Encode a source selector. Infallible on `index` (any `u32` fits); fails only
/// on an out-of-range `bit`.
pub fn encode_source(sel: &SourceSel) -> Result<u64, FormatError> {
    let mut w = (u64::from(sel.index) << SEL_INDEX_SHIFT) & SEL_INDEX_MASK;
    w |= checked_bit(sel.bit)?;
    w |= (u64::from(sel.space as u8) << SEL_SPACE_SHIFT) & SEL_SPACE_MASK;
    if sel.invert {
        w |= SEL_INVERT_BIT;
    }
    w |= SEL_VALID_BIT;
    debug_assert_eq!(w & SEL_RESERVED_MASK, 0);
    Ok(w)
}

pub fn encode_destination(sel: &DestinationSel) -> Result<u64, FormatError> {
    let mut w = (u64::from(sel.index) << SEL_INDEX_SHIFT) & SEL_INDEX_MASK;
    w |= checked_bit(sel.bit)?;
    w |= (u64::from(sel.space as u8) << SEL_SPACE_SHIFT) & SEL_SPACE_MASK;
    w |= SEL_VALID_BIT;
    Ok(w)
}

/// The zero word is a valid *padding* lane (valid bit clear). Any word with
/// reserved bits set is rejected even when the valid bit is clear -- Phase 2
/// requires all padding to be deterministically zero.
pub fn decode_source(word: u64) -> Result<Option<SourceSel>, FormatError> {
    if word & SEL_RESERVED_MASK != 0 {
        return Err(FormatError::ReservedBitsSet { word });
    }
    if word & SEL_VALID_BIT == 0 {
        return Ok(None);
    }
    let raw_space = ((word & SEL_SPACE_MASK) >> SEL_SPACE_SHIFT) as u8;
    let space = match raw_space {
        0 => SourceSpace::Constant,
        1 => SourceSpace::PreviousState,
        2 => SourceSpace::CurrentStage,
        3 => SourceSpace::LocalShared,
        other => return Err(FormatError::InvalidSourceSpace { raw: other }),
    };
    Ok(Some(SourceSel {
        space,
        index: ((word & SEL_INDEX_MASK) >> SEL_INDEX_SHIFT) as u32,
        bit: ((word & SEL_BIT_MASK) >> SEL_BIT_SHIFT) as u8,
        invert: word & SEL_INVERT_BIT != 0,
    }))
}

pub fn decode_destination(word: u64) -> Result<Option<DestinationSel>, FormatError> {
    if word & SEL_RESERVED_MASK != 0 {
        return Err(FormatError::ReservedBitsSet { word });
    }
    if word & SEL_INVERT_BIT != 0 {
        // destinations never invert; a set invert bit is corruption.
        return Err(FormatError::ReservedBitsSet { word });
    }
    if word & SEL_VALID_BIT == 0 {
        return Ok(None);
    }
    let raw_space = ((word & SEL_SPACE_MASK) >> SEL_SPACE_SHIFT) as u8;
    let space = match raw_space {
        0 => DestinationSpace::LocalShared,
        1 => DestinationSpace::CurrentStage,
        2 => DestinationSpace::NextState,
        other => return Err(FormatError::InvalidDestinationSpace { raw: other }),
    };
    Ok(Some(DestinationSel {
        space,
        index: ((word & SEL_INDEX_MASK) >> SEL_INDEX_SHIFT) as u32,
        bit: ((word & SEL_BIT_MASK) >> SEL_BIT_SHIFT) as u8,
    }))
}

// ===========================================================================
// Phase 3 -- field/bit-major transpose
// ===========================================================================

/// `round_up(n, 32)` -- the warp-padded lane count.
pub fn padded_count(n: usize) -> usize {
    n.checked_add(31).map(|x| x / 32 * 32).unwrap_or(usize::MAX)
}

/// One logical field of a macro type (e.g. DSP `A`), `width` bits wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelField {
    pub name: &'static str,
    pub width: usize,
}

/// Transpose a per-instance, per-bit selector grid into a field/bit-major
/// `u64` stream: `word[field_base + bit * padded_count + instance]`.
///
/// `sel[instance][flat_bit]` where `flat_bit` runs over the concatenation of
/// `fields` in order. Missing/short entries are padded with the zero word
/// (valid bit clear).
pub fn transpose_selectors(
    fields: &[SelField],
    n_instances: usize,
    sel: impl Fn(usize, usize) -> u64,
) -> Vec<u64> {
    let pc = padded_count(n_instances);
    let total_bits: usize = fields.iter().map(|f| f.width).sum();
    let mut out = vec![0u64; total_bits * pc];
    let mut field_base = 0usize;
    let mut flat_bit = 0usize;
    for f in fields {
        for b in 0..f.width {
            let row = field_base + b * pc;
            for inst in 0..n_instances {
                out[row + inst] = sel(inst, flat_bit + b);
            }
            // instances n_instances..pc stay zero (padding lanes).
        }
        field_base += f.width * pc;
        flat_bit += f.width;
    }
    out
}

/// Address (in `u64` words, relative to the section base) of one selector.
pub fn selector_word_index(
    fields: &[SelField],
    n_instances: usize,
    field_idx: usize,
    bit_in_field: usize,
    instance: usize,
) -> usize {
    let pc = padded_count(n_instances);
    let field_base: usize = fields[..field_idx].iter().map(|f| f.width * pc).sum();
    field_base + bit_in_field * pc + instance
}

// ===========================================================================
// Phase 2 -- versioned, sectioned ABI
// ===========================================================================

pub const V2_MAGIC: u64 = 0x32_56_5F_4D_45_47_5F_47; // "G_GEM_V2" little-endian
pub const V2_VERSION: u32 = 2;
/// Fixed header size in `u64` words (512 bytes, 16-byte aligned start for any
/// following 128-bit-load section). Version 2 enlarged the inline section
/// table so the unified AIG/macro program never silently drops section
/// descriptors after the seventh entry.
pub const V2_HEADER_WORDS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SectionKind {
    /// CSR: `blocks_start`-equivalent (`u32` pairs packed 2-per-`u64`).
    BlocksStart,
    /// per-(partition, wave) descriptors.
    WaveDescriptors,
    /// type-homogeneous queue membership (node ids).
    Queues,
    /// DSP48E2 source selectors, field/bit-major.
    DspSourceSel,
    /// DSP48E2 destination selectors.
    DspDestSel,
    /// DSP48E2 per-instance compile-time controls.
    DspControls,
    Carry4SourceSel,
    Carry4DestSel,
    Srlc32eSourceSel,
    Srlc32eDestSel,
    /// generated coalesced global-load rounds (Phase 4 output; empty until built).
    GatherPlan,
    /// Four u64 words per AIG operation: depth, source A, source B, destination.
    AigOperations,
    /// Four words per state/output commit: kind, data source, enable, destination.
    EndpointOperations,
    /// 124 words per synchronous SRAM: storage base, 91 source selectors,
    /// then 32 registered read-data destinations.
    SramOperations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    /// `u64`-word offset from program start.
    pub start: u64,
    /// exclusive `u64`-word end.
    pub end: u64,
    /// required byte alignment of `start` (8, or 16 for 128-bit-load sections).
    pub align_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptV2Header {
    pub magic: u64,
    pub version: u32,
    pub total_words: u64,
    pub num_major_stages: u32,
    pub num_partitions: u32,
    pub num_waves: u32,
    pub shared_words_per_block: u32,
    pub feature_flags: u64,
    /// deterministic diagnostic hash of every section's bytes.
    pub content_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramLayoutV2 {
    pub header: ScriptV2Header,
    pub sections: Vec<Section>,
}

impl ProgramLayoutV2 {
    pub fn section(&self, kind: SectionKind) -> Option<&Section> {
        self.sections.iter().find(|s| s.kind == kind)
    }

    /// The 32 header words, ready to prepend to the program stream.
    pub fn header_words(&self) -> [u64; V2_HEADER_WORDS] {
        let h = &self.header;
        let mut w = [0u64; V2_HEADER_WORDS];
        w[0] = h.magic;
        w[1] = u64::from(h.version);
        w[2] = h.total_words;
        w[3] = u64::from(h.num_major_stages);
        w[4] = u64::from(h.num_partitions);
        w[5] = u64::from(h.num_waves);
        w[6] = u64::from(h.shared_words_per_block);
        w[7] = h.feature_flags;
        w[8] = h.content_hash;
        w[9] = self.sections.len() as u64;
        // section table: (kind<<48 | align<<32) , start , end  -- 3 words each,
        assert!(
            self.sections.len() <= (V2_HEADER_WORDS - 10) / 3,
            "V2 inline section table overflow"
        );
        for (i, s) in self.sections.iter().enumerate() {
            let base = 10 + i * 3;
            w[base] = ((s.kind as u64) << 48) | (s.align_bytes << 32);
            w[base + 1] = s.start;
            w[base + 2] = s.end;
        }
        w
    }
}

/// Phase 2 validator. `program_words` is the true length of the assembled
/// `u64` stream (header included).
pub fn validate(layout: &ProgramLayoutV2, program_words: u64) -> Result<(), FormatError> {
    let h = &layout.header;
    if h.magic != V2_MAGIC {
        return Err(FormatError::BadMagic { got: h.magic });
    }
    if h.version != V2_VERSION {
        return Err(FormatError::UnsupportedVersion { got: h.version });
    }
    if h.total_words != program_words {
        return Err(FormatError::ArithmeticOverflow {
            what: "header.total_words != program length",
        });
    }
    let mut prev_end: u64 = V2_HEADER_WORDS as u64;
    for (i, s) in layout.sections.iter().enumerate() {
        if s.align_bytes != 8 && s.align_bytes != 16 {
            return Err(FormatError::SectionMisaligned {
                section: i,
                start_byte: s.start * 8,
                align: s.align_bytes,
            });
        }
        let start_byte = s
            .start
            .checked_mul(8)
            .ok_or(FormatError::ArithmeticOverflow {
                what: "section start*8",
            })?;
        if start_byte % s.align_bytes != 0 {
            return Err(FormatError::SectionMisaligned {
                section: i,
                start_byte,
                align: s.align_bytes,
            });
        }
        if s.start < prev_end {
            return Err(FormatError::SectionNotMonotonic {
                section: i,
                start: s.start,
                prev_end,
            });
        }
        if s.end < s.start {
            return Err(FormatError::SectionNotMonotonic {
                section: i,
                start: s.end,
                prev_end: s.start,
            });
        }
        if s.end > program_words {
            return Err(FormatError::SectionOutOfBounds {
                section: i,
                end: s.end,
                total: program_words,
            });
        }
        prev_end = s.end;
    }
    Ok(())
}

/// FNV-1a over the section byte ranges, in section order -- a deterministic
/// diagnostic hash, not a security primitive.
pub fn content_hash(program: &[u64], sections: &[Section]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for s in sections {
        for &word in &program[s.start as usize..s.end as usize] {
            for b in word.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

// ===========================================================================
// Assembler -- compose a validated program from transposed selector sections
// ===========================================================================

/// One macro type's selector field tables and instance count.
pub struct MacroSelSpec<'a> {
    pub source_kind: SectionKind,
    pub dest_kind: SectionKind,
    pub source_fields: &'a [SelField],
    pub dest_fields: &'a [SelField],
    pub n_instances: usize,
}

/// Assemble the full `u64` program: header words, then, for every macro spec,
/// a 16-byte-aligned transposed source-selector section followed by an
/// 8-byte-aligned transposed destination-selector section. `src`/`dst` are
/// `(spec_index, instance, flat_bit) -> encoded u64` (return `0` for a padding
/// / unconnected lane).
///
/// Returns the flat program and its validated [`ProgramLayoutV2`].
pub fn assemble_macro_program(
    specs: &[MacroSelSpec<'_>],
    num_major_stages: u32,
    num_partitions: u32,
    num_waves: u32,
    shared_words_per_block: u32,
    src: impl Fn(usize, usize, usize) -> u64,
    dst: impl Fn(usize, usize, usize) -> u64,
) -> Result<(Vec<u64>, ProgramLayoutV2), FormatError> {
    let mut program = vec![0u64; V2_HEADER_WORDS];
    let mut sections: Vec<Section> = Vec::new();

    for (si, spec) in specs.iter().enumerate() {
        if spec.n_instances == 0 {
            continue;
        }
        // source section, 16-byte aligned (candidate for 128-bit vector loads).
        if program.len() % 2 != 0 {
            program.push(0);
        }
        let s_start = program.len() as u64;
        let s_body = transpose_selectors(spec.source_fields, spec.n_instances, |inst, fb| {
            src(si, inst, fb)
        });
        program.extend_from_slice(&s_body);
        sections.push(Section {
            kind: spec.source_kind,
            start: s_start,
            end: program.len() as u64,
            align_bytes: 16,
        });

        // destination section, 8-byte aligned.
        let d_start = program.len() as u64;
        let d_body = transpose_selectors(spec.dest_fields, spec.n_instances, |inst, fb| {
            dst(si, inst, fb)
        });
        program.extend_from_slice(&d_body);
        sections.push(Section {
            kind: spec.dest_kind,
            start: d_start,
            end: program.len() as u64,
            align_bytes: 8,
        });
    }

    let hash = content_hash(&program, &sections);
    let mut layout = ProgramLayoutV2 {
        header: ScriptV2Header {
            magic: V2_MAGIC,
            version: V2_VERSION,
            total_words: program.len() as u64,
            num_major_stages,
            num_partitions,
            num_waves,
            shared_words_per_block,
            feature_flags: 0,
            content_hash: hash,
        },
        sections,
    };
    // stamp the header words into the reserved prefix.
    let hw = layout.header_words();
    program[..V2_HEADER_WORDS].copy_from_slice(&hw);
    // recompute hash now that the header is populated? no -- hash covers only
    // section bytes, which the header stamp does not touch. keep as-is.
    layout.header.total_words = program.len() as u64;

    validate(&layout, program.len() as u64)?;
    Ok((program, layout))
}

// ===========================================================================
// Phase 6 (host half) -- independent decoder
// ===========================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedSelectorSection {
    pub kind: SectionKind,
    /// `records[flat_bit][instance]` = `Some(sel)` or `None` (padding lane).
    pub sources: Vec<Vec<Option<SourceSel>>>,
    pub dests: Vec<Vec<Option<DestinationSel>>>,
}

/// Re-read a transposed selector section straight from the bytes. Deliberately
/// does NOT use [`selector_word_index`] or any encoder helper -- an encoder bug
/// and a decoder bug must not cancel.
pub fn decode_selector_section(
    program: &[u64],
    section: &Section,
    fields: &[SelField],
    n_instances: usize,
    is_source: bool,
) -> Result<DecodedSelectorSection, FormatError> {
    let pc = padded_count(n_instances);
    let total_bits: usize = fields.iter().map(|f| f.width).sum();
    let body = &program[section.start as usize..section.end as usize];
    if pc != 0 && body.len() != total_bits * pc {
        return Err(FormatError::ArithmeticOverflow {
            what: "selector section length",
        });
    }
    let mut sources = Vec::new();
    let mut dests = Vec::new();
    for flat_bit in 0..total_bits {
        let row = &body[flat_bit * pc..flat_bit * pc + pc];
        // padding lanes must be exactly zero.
        for (i, &w) in row.iter().enumerate().skip(n_instances) {
            if w != 0 {
                return Err(FormatError::PaddingNotZero {
                    word_index: section.start as usize + flat_bit * pc + i,
                    word: w,
                });
            }
        }
        if is_source {
            let mut r = Vec::with_capacity(n_instances);
            for &w in &row[..n_instances] {
                r.push(decode_source(w)?);
            }
            sources.push(r);
        } else {
            let mut r = Vec::with_capacity(n_instances);
            for &w in &row[..n_instances] {
                r.push(decode_destination(w)?);
            }
            dests.push(r);
        }
    }
    Ok(DecodedSelectorSection {
        kind: section.kind,
        sources,
        dests,
    })
}

/// Phase 6 gate: every scheduled node id appears exactly once across the queue
/// sections, and every serialized same-cycle edge has producer wave < consumer
/// wave.
pub fn check_schedule_serialization(
    node_ids_per_queue: &[Vec<usize>],
    total_nodes: usize,
    same_cycle_edges_by_wave: &[(u32, u32)],
) -> Result<(), FormatError> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    for q in node_ids_per_queue {
        for &n in q {
            if !seen.insert(n) {
                return Err(FormatError::NodeScheduledTwice { node: n });
            }
        }
    }
    if seen.len() != total_nodes {
        return Err(FormatError::ArithmeticOverflow {
            what: "not every node serialized exactly once",
        });
    }
    for &(pw, cw) in same_cycle_edges_by_wave {
        if pw >= cw {
            return Err(FormatError::SameCycleEdgeMisordered {
                producer_wave: pw,
                consumer_wave: cw,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `csrc/format_v2_abi.h` hard-codes the same numbers; this catches drift on
    /// the Rust side. The C++ side has its own `static_assert`s.
    #[test]
    fn abi_constants_match_the_shared_header() {
        assert_eq!(SEL_INDEX_MASK, 0x0000_0000_FFFF_FFFF);
        assert_eq!(SEL_BIT_SHIFT, 32);
        assert_eq!(SEL_BIT_MASK, 0x0000_003F_0000_0000);
        assert_eq!(SEL_SPACE_SHIFT, 38);
        assert_eq!(SEL_SPACE_MASK, 0x0000_00C0_0000_0000);
        assert_eq!(SEL_INVERT_BIT, 1 << 40);
        assert_eq!(SEL_VALID_BIT, 1 << 41);
        assert_eq!(SEL_RESERVED_MASK, 0xFFFF_FC00_0000_0000);
        assert_eq!(V2_MAGIC, 0x3256_5F4D_4547_5F47);
        assert_eq!(V2_VERSION, 2);
        assert_eq!(V2_HEADER_WORDS, 64);
        assert_eq!(V2_HEADER_WORDS % 2, 0, "header ends 16-byte aligned");
        // the selector bit-fields tile the whole u64, no gaps, no overlap.
        assert_eq!(
            SEL_INDEX_MASK
                | SEL_BIT_MASK
                | SEL_SPACE_MASK
                | SEL_INVERT_BIT
                | SEL_VALID_BIT
                | SEL_RESERVED_MASK,
            u64::MAX
        );
        assert_eq!(SEL_INDEX_MASK & SEL_BIT_MASK, 0);
        assert_eq!(SEL_BIT_MASK & SEL_SPACE_MASK, 0);
    }

    #[test]
    fn source_selector_round_trips_over_every_field() {
        for &space in &[
            SourceSpace::Constant,
            SourceSpace::PreviousState,
            SourceSpace::CurrentStage,
            SourceSpace::LocalShared,
        ] {
            for &bit in &[0u8, 1, 5, 31, 32, 47, 63] {
                for &invert in &[false, true] {
                    for &index in &[0u32, 1, 255, 65_535, u32::MAX] {
                        let sel = SourceSel {
                            space,
                            index,
                            bit,
                            invert,
                        };
                        let w = encode_source(&sel).unwrap();
                        assert_eq!(w & SEL_RESERVED_MASK, 0, "reserved bits must stay clear");
                        assert_eq!(decode_source(w).unwrap(), Some(sel));
                    }
                }
            }
        }
    }

    #[test]
    fn destination_selector_round_trips() {
        for &space in &[
            DestinationSpace::LocalShared,
            DestinationSpace::CurrentStage,
            DestinationSpace::NextState,
        ] {
            for &bit in &[0u8, 31, 63] {
                for &index in &[0u32, 42, u32::MAX] {
                    let sel = DestinationSel { space, index, bit };
                    let w = encode_destination(&sel).unwrap();
                    assert_eq!(decode_destination(w).unwrap(), Some(sel));
                }
            }
        }
    }

    #[test]
    fn zero_word_is_a_padding_lane_not_an_error() {
        assert_eq!(decode_source(0).unwrap(), None);
        assert_eq!(decode_destination(0).unwrap(), None);
    }

    #[test]
    fn reserved_bits_and_bad_space_are_rejected() {
        let mut w = encode_source(&SourceSel::CONST1).unwrap();
        w |= 1 << 42; // a reserved bit
        assert_eq!(
            decode_source(w),
            Err(FormatError::ReservedBitsSet { word: w })
        );

        // valid + destination space 3 (reserved).
        let bad = SEL_VALID_BIT | (3u64 << SEL_SPACE_SHIFT);
        assert_eq!(
            decode_destination(bad),
            Err(FormatError::InvalidDestinationSpace { raw: 3 })
        );

        assert_eq!(
            encode_source(&SourceSel {
                space: SourceSpace::Constant,
                index: 0,
                bit: 64,
                invert: false
            }),
            Err(FormatError::BitIndexOutOfRange { bit: 64 })
        );
    }

    #[test]
    fn transpose_puts_adjacent_lanes_one_word_apart() {
        let fields = [
            SelField {
                name: "A",
                width: 3,
            },
            SelField {
                name: "B",
                width: 2,
            },
        ];
        for &n in &[0usize, 1, 31, 32, 33, 63, 64, 255, 256] {
            let pc = padded_count(n);
            assert_eq!(pc % 32, 0);
            assert!(pc >= n);
            // encode a recognizable selector per (instance, flat_bit).
            let sel = |inst: usize, fb: usize| {
                encode_source(&SourceSel {
                    space: SourceSpace::LocalShared,
                    index: (inst as u32) << 8 | fb as u32,
                    bit: (fb % 64) as u8,
                    invert: false,
                })
                .unwrap()
            };
            let stream = transpose_selectors(&fields, n, sel);
            assert_eq!(stream.len(), 5 * pc);

            if n >= 2 {
                // field B (index 1), bit 1 -> flat_bit 4
                let w0 = selector_word_index(&fields, n, 1, 1, 0);
                let w1 = selector_word_index(&fields, n, 1, 1, 1);
                assert_eq!(
                    w1 - w0,
                    1,
                    "adjacent active lanes are consecutive u64 words"
                );
                assert_eq!(decode_source(stream[w0]).unwrap().unwrap().index, 4);
                assert_eq!(
                    decode_source(stream[w1]).unwrap().unwrap().index,
                    (1 << 8) | 4
                );
            }
            // padding lanes are exactly zero.
            for fb in 0..5 {
                for inst in n..pc {
                    assert_eq!(stream[fb * pc + inst], 0, "padding lane must be zero");
                }
            }
            // fields do not overlap: A occupies [0, 3*pc), B occupies [3*pc, 5*pc).
            let a_end = selector_word_index(&fields, n, 0, 2, 0) + if n > 0 { pc } else { 0 };
            let b_start = if n > 0 {
                selector_word_index(&fields, n, 1, 0, 0)
            } else {
                3 * pc
            };
            assert!(a_end <= b_start);
        }
    }

    #[test]
    fn decoder_reconstructs_without_encoder_helpers() {
        let fields = [
            SelField {
                name: "S",
                width: 4,
            },
            SelField {
                name: "DI",
                width: 4,
            },
        ];
        let n = 33;
        let sel = |inst: usize, fb: usize| {
            encode_source(&SourceSel {
                space: SourceSpace::PreviousState,
                index: inst as u32,
                bit: fb as u8,
                invert: (inst + fb) % 2 == 0,
            })
            .unwrap()
        };
        let body = transpose_selectors(&fields, n, sel);
        // wrap in a program with a header-sized prefix.
        let mut program = vec![0u64; V2_HEADER_WORDS];
        let start = program.len() as u64;
        program.extend_from_slice(&body);
        let section = Section {
            kind: SectionKind::Carry4SourceSel,
            start,
            end: start + body.len() as u64,
            align_bytes: 8,
        };
        let decoded = decode_selector_section(&program, &section, &fields, n, true).unwrap();
        assert_eq!(decoded.sources.len(), 8); // 4 + 4 flat bits
        for fb in 0..8 {
            for inst in 0..n {
                let got = decoded.sources[fb][inst].unwrap();
                assert_eq!(got.index, inst as u32);
                assert_eq!(got.bit, fb as u8);
                assert_eq!(got.invert, (inst + fb) % 2 == 0);
            }
        }
    }

    #[test]
    fn decoder_flags_nonzero_padding() {
        let fields = [SelField {
            name: "X",
            width: 1,
        }];
        let n = 1;
        let pc = padded_count(n); // 32
        let mut program = vec![0u64; V2_HEADER_WORDS];
        let start = program.len() as u64;
        program.extend(std::iter::repeat(0u64).take(pc));
        program[start as usize] = encode_source(&SourceSel::CONST1).unwrap();
        program[start as usize + 5] = 0xdead_beef; // a garbage padding lane
        let section = Section {
            kind: SectionKind::DspSourceSel,
            start,
            end: start + pc as u64,
            align_bytes: 8,
        };
        assert!(matches!(
            decode_selector_section(&program, &section, &fields, n, true),
            Err(FormatError::PaddingNotZero { .. })
        ));
    }

    #[test]
    fn validate_catches_overlap_misalignment_and_bounds() {
        let mk = |sections: Vec<Section>, total: u64| ProgramLayoutV2 {
            header: ScriptV2Header {
                magic: V2_MAGIC,
                version: V2_VERSION,
                total_words: total,
                num_major_stages: 1,
                num_partitions: 1,
                num_waves: 1,
                shared_words_per_block: 0,
                feature_flags: 0,
                content_hash: 0,
            },
            sections,
        };
        let h = V2_HEADER_WORDS as u64;
        // good
        let ok = mk(
            vec![
                Section {
                    kind: SectionKind::DspSourceSel,
                    start: h,
                    end: h + 64,
                    align_bytes: 16,
                },
                Section {
                    kind: SectionKind::DspDestSel,
                    start: h + 64,
                    end: h + 96,
                    align_bytes: 8,
                },
            ],
            h + 96,
        );
        assert_eq!(validate(&ok, h + 96), Ok(()));
        // overlap
        let overlap = mk(
            vec![
                Section {
                    kind: SectionKind::DspSourceSel,
                    start: h,
                    end: h + 64,
                    align_bytes: 8,
                },
                Section {
                    kind: SectionKind::DspDestSel,
                    start: h + 32,
                    end: h + 96,
                    align_bytes: 8,
                },
            ],
            h + 96,
        );
        assert!(matches!(
            validate(&overlap, h + 96),
            Err(FormatError::SectionNotMonotonic { .. })
        ));
        // 16-byte misalignment (odd word start)
        let mis = mk(
            vec![Section {
                kind: SectionKind::DspSourceSel,
                start: h + 1,
                end: h + 9,
                align_bytes: 16,
            }],
            h + 9,
        );
        assert!(matches!(
            validate(&mis, h + 9),
            Err(FormatError::SectionMisaligned { .. })
        ));
        // out of bounds
        let oob = mk(
            vec![Section {
                kind: SectionKind::DspSourceSel,
                start: h,
                end: h + 200,
                align_bytes: 8,
            }],
            h + 100,
        );
        assert!(matches!(
            validate(&oob, h + 100),
            Err(FormatError::SectionOutOfBounds { .. })
        ));
        // bad magic
        let mut bad = ok.clone();
        bad.header.magic = 0;
        assert!(matches!(
            validate(&bad, h + 96),
            Err(FormatError::BadMagic { .. })
        ));
    }

    #[test]
    fn header_words_are_stable_and_carry_the_section_table() {
        let layout = ProgramLayoutV2 {
            header: ScriptV2Header {
                magic: V2_MAGIC,
                version: V2_VERSION,
                total_words: 200,
                num_major_stages: 2,
                num_partitions: 3,
                num_waves: 4,
                shared_words_per_block: 96,
                feature_flags: 0b101,
                content_hash: 0x1234_5678_9abc_def0,
            },
            sections: vec![
                Section {
                    kind: SectionKind::BlocksStart,
                    start: 32,
                    end: 40,
                    align_bytes: 8,
                },
                Section {
                    kind: SectionKind::DspSourceSel,
                    start: 40,
                    end: 168,
                    align_bytes: 16,
                },
            ],
        };
        let w = layout.header_words();
        assert_eq!(w[0], V2_MAGIC);
        assert_eq!(w[1], u64::from(V2_VERSION));
        assert_eq!(w[2], 200);
        assert_eq!(w[9], 2); // section count
        assert_eq!(w[10] & 0xffff, 0); // BlocksStart kind == 0 discriminant slot
        assert_eq!(w[11], 32);
        assert_eq!(w[12], 40);
        assert_eq!(w[14], 40); // second section start
        assert_eq!(w[15], 168);
        // determinism
        assert_eq!(w, layout.header_words());
    }

    #[test]
    fn schedule_serialization_gate() {
        assert_eq!(
            check_schedule_serialization(&[vec![0, 1], vec![2]], 3, &[(0, 1), (1, 2)]),
            Ok(())
        );
        assert_eq!(
            check_schedule_serialization(&[vec![0, 1], vec![1]], 2, &[]),
            Err(FormatError::NodeScheduledTwice { node: 1 })
        );
        assert_eq!(
            check_schedule_serialization(&[vec![0, 1]], 2, &[(2, 1)]),
            Err(FormatError::SameCycleEdgeMisordered {
                producer_wave: 2,
                consumer_wave: 1
            })
        );
    }

    #[test]
    fn assemble_macro_program_round_trips_through_the_independent_decoder() {
        // one CARRY4 queue with 5 instances, S/DI/CIN/CYINIT sources, O/CO dests.
        let src_fields = [
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
        let dst_fields = [
            SelField {
                name: "O",
                width: 4,
            },
            SelField {
                name: "CO",
                width: 4,
            },
        ];
        let n = 5;
        let specs = [MacroSelSpec {
            source_kind: SectionKind::Carry4SourceSel,
            dest_kind: SectionKind::Carry4DestSel,
            source_fields: &src_fields,
            dest_fields: &dst_fields,
            n_instances: n,
        }];
        let src = |_si: usize, inst: usize, fb: usize| {
            encode_source(&SourceSel {
                space: SourceSpace::LocalShared,
                index: 100 + inst as u32,
                bit: fb as u8,
                invert: false,
            })
            .unwrap()
        };
        let dst = |_si: usize, inst: usize, fb: usize| {
            encode_destination(&DestinationSel {
                space: DestinationSpace::CurrentStage,
                index: 200 + inst as u32,
                bit: fb as u8,
            })
            .unwrap()
        };
        let (program, layout) = assemble_macro_program(&specs, 1, 1, 2, 64, src, dst).unwrap();

        assert_eq!(validate(&layout, program.len() as u64), Ok(()));
        assert_eq!(program[0], V2_MAGIC);
        // source section is 16-byte aligned.
        let ssec = layout.section(SectionKind::Carry4SourceSel).unwrap();
        assert_eq!((ssec.start * 8) % 16, 0);

        let decoded_src = decode_selector_section(&program, ssec, &src_fields, n, true).unwrap();
        assert_eq!(decoded_src.sources.len(), 10); // 4+4+1+1
        assert_eq!(decoded_src.sources[0][3].unwrap().index, 103);

        let dsec = layout.section(SectionKind::Carry4DestSel).unwrap();
        let decoded_dst = decode_selector_section(&program, dsec, &dst_fields, n, false).unwrap();
        assert_eq!(decoded_dst.dests[7][4].unwrap().index, 204);
        assert_eq!(
            decoded_dst.dests[7][4].unwrap().space,
            DestinationSpace::CurrentStage
        );

        // content hash is reproducible.
        assert_eq!(
            layout.header.content_hash,
            content_hash(&program, &layout.sections)
        );
    }

    #[test]
    fn content_hash_is_order_sensitive_and_deterministic() {
        let program: Vec<u64> = (0..64).collect();
        let a = Section {
            kind: SectionKind::DspSourceSel,
            start: 0,
            end: 32,
            align_bytes: 8,
        };
        let b = Section {
            kind: SectionKind::DspDestSel,
            start: 32,
            end: 64,
            align_bytes: 8,
        };
        let h1 = content_hash(&program, &[a, b]);
        let h2 = content_hash(&program, &[b, a]);
        assert_ne!(h1, h2);
        assert_eq!(h1, content_hash(&program, &[a, b]));
    }
}
