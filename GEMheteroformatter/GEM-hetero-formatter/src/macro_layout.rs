// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Host-side memory formatter for preserved macros (Zenith PS, Part A / bullet 2).
//!
//! > "Extend GEM's host-side memory formatter to map these intercepted
//! > topological nodes into flattened, 64-bit aligned CUDA memory buffers
//! > optimized for coalesced global memory bandwidth."
//!
//! GEM's V1 script (`FlattenedScriptV1::blocks_data`) is a `u32` stream walked
//! by a shared `script_pi` cursor; splicing wide arithmetic operands into it
//! would break the 256-lane `VectorRead4` cadence and every downstream offset.
//! Instead this module emits a **separate, structure-of-arrays device buffer**:
//!
//! ```text
//!  MacroDeviceLayout
//!   ├─ header            : [u64; HEADER_U64]        (magic, version, counts, section offsets)
//!   ├─ dsp   fields      : F_dsp   arrays, each [u64; n_dsp]     8-byte aligned
//!   ├─ carry4 fields     : F_carry arrays, each [u64; n_carry]   8-byte aligned
//!   └─ srlc32e fields    : F_srl   arrays, each [u64; n_srl]     8-byte aligned
//! ```
//!
//! Every field of every macro type is its own contiguous array. Lane `i` of a
//! warp evaluating macro-instance `i` touches element `i` of each array, so a
//! whole warp's read of one field is a single naturally-aligned
//! `128 B` (u32) / `256 B` (u64) transaction — fully coalesced. Instance-AoS
//! (`struct Dsp { a, b, c, ... }[]`) would strand adjacent lanes across a
//! cache line per field; SoA is what "optimized for coalesced bandwidth"
//! means here.
//!
//! The buffer is uploaded exactly like `blocks_start` / `blocks_data`: wrap it
//! in `ulib::UVec<u64>` and pass `&buf` to the ucc binding; `UVec::as_uptr`
//! does the H2D copy. Nothing here allocates device memory directly.

use crate::aig::AIG;
use crate::schedule::{HeteroSchedule, MacroKind};

/// `u64` words of the fixed header. Kept a multiple of 8 for 64-byte
/// alignment of the first field array.
pub const HEADER_U64: usize = 8;

/// `'G' 'E' 'M' 'H' 'M' 'A' 'C' '2'` little-endian.
pub const MACRO_LAYOUT_MAGIC: u64 = 0x32_43_41_4D_48_4D_45_47;
pub const MACRO_LAYOUT_VERSION: u64 = 1;

/// Per-macro-type field tables. Each entry is one SoA array in the final
/// buffer, in declaration order. Widths are the *logical* bit width; storage
/// is always one `u64` per instance so the arrays stay 8-byte aligned and the
/// wide DSP operands (`C`, `P`, 48-bit) need no split.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DspField {
    /// 27-bit A operand (already sign-meaningful, right-aligned).
    A,
    /// 27-bit D operand.
    D,
    /// 18-bit B operand.
    B,
    /// 48-bit C operand.
    C,
    /// 48-bit `P` value clocked on the previous edge (old state, read for MAC).
    PrevP,
    /// packed controls: bit0..1 = 2-bit op state (0=C,1=M,2=P+M),
    /// bit2 = pre-adder enable, bit3 = CEP, bit4 = RSTP.
    Ctl,
    /// 48-bit `P_next` result written back by the kernel.
    NextP,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Carry4Field {
    /// bit0..3 = S[3:0], bit4..7 = DI[3:0], bit8 = CIN, bit9 = CYINIT.
    In,
    /// bit0..3 = O[3:0], bit4..7 = CO[3:0].
    Out,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Srlc32eField {
    /// bit0 = D, bit1 = CE, bit2 = global rising edge, bit3..7 = A[4:0].
    In,
    /// bit0 = Q, bit1 = Q31.
    Out,
    /// 32-bit shift-register storage (old state in, next state out).
    Storage,
}

pub const DSP_FIELDS: [DspField; 7] = [
    DspField::A,
    DspField::D,
    DspField::B,
    DspField::C,
    DspField::PrevP,
    DspField::Ctl,
    DspField::NextP,
];
pub const CARRY4_FIELDS: [Carry4Field; 2] = [Carry4Field::In, Carry4Field::Out];
pub const SRLC32E_FIELDS: [Srlc32eField; 3] =
    [Srlc32eField::In, Srlc32eField::Out, Srlc32eField::Storage];

/// One SoA array placed in the buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSection {
    pub kind: MacroKind,
    /// index into `DSP_FIELDS` / `CARRY4_FIELDS` / `SRLC32E_FIELDS`.
    pub field_index: usize,
    /// `u64` offset from the start of the buffer. Always a multiple of `1`
    /// `u64` (8 bytes); the first section is `HEADER_U64`.
    pub offset_u64: usize,
    /// element count == number of instances of `kind`.
    pub len: usize,
}

/// One macro instance's operand provenance: which AIG net drives each logical
/// input bit. The V2 kernel's gather step (or a host pre-pass) uses this to
/// fill the `In` / `A` / `B` / ... arrays from the current-cycle value image.
#[derive(Clone, Debug)]
pub struct MacroOperandMap {
    pub kind: MacroKind,
    /// `aig.carry4s` / `aig.dsps` / `aig.srlc32es` cell id.
    pub cell_id: usize,
    /// dense instance index (== lane the kernel evaluates it on).
    pub instance: usize,
    /// `(logical_bit, pin_with_invert)` for every driven operand bit.
    /// `pin_with_invert == 0` means "tied to constant 0". `logical_bit` is the
    /// position inside the concatenated operand encoding for `kind`
    /// (matches the `*Field::In` / DSP field bit assignments above).
    pub operand_bits: Vec<(u16, usize)>,
    /// same shape, for the macro's output bits (so the writeback / scatter
    /// step knows which net each result bit updates).
    pub result_bits: Vec<(u16, usize)>,
}

/// The finished, flat description. `total_u64` `u64` words; upload as
/// `UVec<u64>` of exactly this length.
#[derive(Clone, Debug)]
pub struct MacroDeviceLayout {
    pub n_dsp: usize,
    pub n_carry4: usize,
    pub n_srlc32e: usize,
    pub sections: Vec<FieldSection>,
    pub total_u64: usize,
    /// instance order per kind == the order used for `instance` in
    /// [`MacroOperandMap`] and for the SoA array index.
    pub dsp_cells: Vec<usize>,
    pub carry4_cells: Vec<usize>,
    pub srlc32e_cells: Vec<usize>,
    pub operand_maps: Vec<MacroOperandMap>,
}

impl MacroDeviceLayout {
    /// `u64` offset of one field's array, or `None` if that kind has zero
    /// instances (the section is omitted).
    pub fn section_of(&self, kind: MacroKind, field_index: usize) -> Option<&FieldSection> {
        self.sections
            .iter()
            .find(|s| s.kind == kind && s.field_index == field_index)
    }

    /// Encode the fixed header words. The kernel/validator checks these before
    /// touching any field array.
    pub fn header_words(&self) -> [u64; HEADER_U64] {
        let off = |k, f| self.section_of(k, f).map_or(0, |s| s.offset_u64 as u64);
        [
            MACRO_LAYOUT_MAGIC,
            MACRO_LAYOUT_VERSION,
            (self.n_dsp as u64) | ((self.n_carry4 as u64) << 21) | ((self.n_srlc32e as u64) << 42),
            self.total_u64 as u64,
            off(MacroKind::Dsp48e2, 0),
            off(MacroKind::Carry4, 0),
            off(MacroKind::Srlc32e, 0),
            // number of distinct field arrays actually emitted.
            self.sections.len() as u64,
        ]
    }

    /// Host-side structural validator (PS Part A: "optimized for coalesced
    /// bandwidth" ⇒ every field array 8-byte aligned, monotonic, in bounds).
    pub fn validate(&self) -> Result<(), String> {
        if self.sections.is_empty() {
            return if self.total_u64 == HEADER_U64 {
                Ok(())
            } else {
                Err(format!("no sections but total_u64 = {}", self.total_u64))
            };
        }
        let mut cursor = HEADER_U64;
        for (i, s) in self.sections.iter().enumerate() {
            if s.offset_u64 != cursor {
                return Err(format!(
                    "section {i} ({:?}/{}) at {} u64, expected {}",
                    s.kind, s.field_index, s.offset_u64, cursor
                ));
            }
            // one u64 per instance ⇒ inherently 8-byte aligned and coalesced.
            let want_len = match s.kind {
                MacroKind::Dsp48e2 => self.n_dsp,
                MacroKind::Carry4 => self.n_carry4,
                MacroKind::Srlc32e => self.n_srlc32e,
            };
            if s.len != want_len {
                return Err(format!("section {i} len {} != {want_len}", s.len));
            }
            cursor += s.len;
        }
        if cursor != self.total_u64 {
            return Err(format!("sections end at {cursor}, total_u64 = {}", self.total_u64));
        }
        for m in &self.operand_maps {
            let bound = match m.kind {
                MacroKind::Dsp48e2 => self.n_dsp,
                MacroKind::Carry4 => self.n_carry4,
                MacroKind::Srlc32e => self.n_srlc32e,
            };
            if m.instance >= bound {
                return Err(format!(
                    "operand map for cell {} has instance {} >= {bound}",
                    m.cell_id, m.instance
                ));
            }
        }
        Ok(())
    }
}

/// Build the layout from the resolved AIG and the heterogeneous schedule.
///
/// `schedule` fixes the instance order (its wave order is stable), which keeps
/// same-wave macros contiguous in the SoA arrays — the kernel evaluates a wave
/// as one warp-stride loop, so contiguous instances = contiguous lanes.
pub fn build_macro_layout(aig: &AIG, schedule: &HeteroSchedule) -> MacroDeviceLayout {
    // instance order: schedule wave order, then cell id, per kind.
    let mut dsp_cells = Vec::new();
    let mut carry4_cells = Vec::new();
    let mut srlc32e_cells = Vec::new();
    for wave in &schedule.waves {
        for &ni in &wave.dsp48e2 {
            dsp_cells.push(schedule.nodes[ni].cell_id);
        }
        for &ni in &wave.carry4 {
            carry4_cells.push(schedule.nodes[ni].cell_id);
        }
        for &ni in &wave.srlc32e {
            srlc32e_cells.push(schedule.nodes[ni].cell_id);
        }
    }

    let n_dsp = dsp_cells.len();
    let n_carry4 = carry4_cells.len();
    let n_srlc32e = srlc32e_cells.len();

    // section table.
    let mut sections = Vec::new();
    let mut cursor = HEADER_U64;
    let push_sections = |sections: &mut Vec<FieldSection>,
                         cursor: &mut usize,
                         kind: MacroKind,
                         n_fields: usize,
                         n_inst: usize| {
        if n_inst == 0 {
            return;
        }
        for f in 0..n_fields {
            sections.push(FieldSection {
                kind,
                field_index: f,
                offset_u64: *cursor,
                len: n_inst,
            });
            *cursor += n_inst;
        }
    };
    push_sections(&mut sections, &mut cursor, MacroKind::Dsp48e2, DSP_FIELDS.len(), n_dsp);
    push_sections(&mut sections, &mut cursor, MacroKind::Carry4, CARRY4_FIELDS.len(), n_carry4);
    push_sections(
        &mut sections,
        &mut cursor,
        MacroKind::Srlc32e,
        SRLC32E_FIELDS.len(),
        n_srlc32e,
    );
    let total_u64 = cursor;

    // operand / result provenance.
    let mut operand_maps = Vec::new();

    for (instance, &cell) in dsp_cells.iter().enumerate() {
        let blk = &aig.dsps[&cell];
        let mut operand_bits = Vec::new();
        // A[0..27] -> logical bits 0..27 of the A field.
        for (k, &p) in blk.a_iv.iter().enumerate() {
            operand_bits.push((k as u16, p));
        }
        // D, B, C, controls are separate *fields*; encode logical bit as
        // field_index * 64 + bit so the gather step can demux.
        for (k, &p) in blk.d_iv.iter().enumerate() {
            operand_bits.push((64 + k as u16, p));
        }
        for (k, &p) in blk.b_iv.iter().enumerate() {
            operand_bits.push((128 + k as u16, p));
        }
        for (k, &p) in blk.c_iv.iter().enumerate() {
            operand_bits.push((192 + k as u16, p));
        }
        for (k, &p) in blk.opmode_iv.iter().enumerate() {
            operand_bits.push((320 + k as u16, p)); // Ctl field raw OPMODE bits; host decodes to 2-bit state
        }
        for (k, &p) in blk.alumode_iv.iter().enumerate() {
            operand_bits.push((330 + k as u16, p));
        }
        for (k, &p) in blk.inmode_iv.iter().enumerate() {
            operand_bits.push((334 + k as u16, p));
        }
        operand_bits.push((340, blk.cep_iv));
        operand_bits.push((341, blk.rstp_iv));

        let mut result_bits = Vec::new();
        for (k, &p) in blk.p_out.iter().enumerate() {
            if p != 0 {
                result_bits.push((384 + k as u16, p << 1)); // NextP field
            }
        }
        operand_maps.push(MacroOperandMap {
            kind: MacroKind::Dsp48e2,
            cell_id: cell,
            instance,
            operand_bits,
            result_bits,
        });
    }

    for (instance, &cell) in carry4_cells.iter().enumerate() {
        let blk = &aig.carry4s[&cell];
        let mut operand_bits = Vec::new();
        for (k, &p) in blk.s_iv.iter().enumerate() {
            operand_bits.push((k as u16, p));
        }
        for (k, &p) in blk.di_iv.iter().enumerate() {
            operand_bits.push((4 + k as u16, p));
        }
        operand_bits.push((8, blk.cin_iv));
        operand_bits.push((9, blk.cyinit_iv));
        let mut result_bits = Vec::new();
        for (k, &p) in blk.o_out.iter().enumerate() {
            if p != 0 {
                result_bits.push((64 + k as u16, p << 1));
            }
        }
        for (k, &p) in blk.co_out.iter().enumerate() {
            if p != 0 {
                result_bits.push((64 + 4 + k as u16, p << 1));
            }
        }
        operand_maps.push(MacroOperandMap {
            kind: MacroKind::Carry4,
            cell_id: cell,
            instance,
            operand_bits,
            result_bits,
        });
    }

    for (instance, &cell) in srlc32e_cells.iter().enumerate() {
        let blk = &aig.srlc32es[&cell];
        let mut operand_bits = vec![(0u16, blk.d_iv), (1u16, blk.ce_iv)];
        // logical bit 2 = global rising edge, filled by the kernel from the
        // clock flag, not an operand pin.
        for (k, &p) in blk.a_iv.iter().enumerate() {
            operand_bits.push((3 + k as u16, p));
        }
        let mut result_bits = Vec::new();
        if blk.q_out != 0 {
            result_bits.push((64, blk.q_out << 1));
        }
        if blk.q31_out != 0 {
            result_bits.push((65, blk.q31_out << 1));
        }
        operand_maps.push(MacroOperandMap {
            kind: MacroKind::Srlc32e,
            cell_id: cell,
            instance,
            operand_bits,
            result_bits,
        });
    }

    MacroDeviceLayout {
        n_dsp,
        n_carry4,
        n_srlc32e,
        sections,
        total_u64,
        dsp_cells,
        carry4_cells,
        srlc32e_cells,
        operand_maps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{Carry4Block, DriverType, DSPBlock, Srlc32eBlock, AIG};
    use crate::schedule::build_schedule;

    /// Fresh AIG whose pin 1 is a primary input (a genuine value source that
    /// macro operand pins can safely reference via `iv(1, 0) == 2`).
    fn empty_aig() -> AIG {
        let mut aig = AIG::default();
        aig.num_aigpins = 1;
        aig.drivers = vec![DriverType::Tie0, DriverType::InputPort(0)];
        aig
    }

    fn add_carry4(aig: &mut AIG, cell: usize, cin_iv: usize) {
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
                di_iv: [2, 2, 2, 2],
                s_iv: [2, 2, 2, 2],
                cin_iv,
                cyinit_iv: 0,
                o_out,
                co_out,
            },
        );
    }

    fn add_dsp(aig: &mut AIG, cell: usize) {
        let mut p_out = [0usize; 48];
        for k in 0..48 {
            aig.num_aigpins += 1;
            aig.drivers.push(DriverType::DSP(cell, k));
            p_out[k] = aig.num_aigpins;
        }
        let mut blk = DSPBlock::default();
        blk.a_iv = [2; 27];
        blk.d_iv = [2; 27];
        blk.b_iv = [2; 18];
        blk.c_iv = [2; 48];
        blk.p_out = p_out;
        aig.dsps.insert(cell, blk);
    }

    fn add_srlc(aig: &mut AIG, cell: usize) {
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(cell, 0));
        let q = aig.num_aigpins;
        aig.num_aigpins += 1;
        aig.drivers.push(DriverType::SRLC32E(cell, 1));
        let q31 = aig.num_aigpins;
        aig.srlc32es.insert(
            cell,
            Srlc32eBlock {
                d_iv: 2,
                ce_iv: 2,
                a_iv: [0; 5],
                clk_iv: 0,
                q_out: q,
                q31_out: q31,
                init: 0,
            },
        );
    }

    fn finish(mut aig: AIG) -> AIG {
        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        aig
    }

    #[test]
    fn every_field_array_is_8byte_aligned_and_monotonic() {
        let mut aig = empty_aig();
        add_dsp(&mut aig, 1);
        add_dsp(&mut aig, 2);
        add_carry4(&mut aig, 3, 0);
        add_srlc(&mut aig, 4);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let layout = build_macro_layout(&aig, &sched);

        layout.validate().unwrap();
        assert_eq!(layout.n_dsp, 2);
        assert_eq!(layout.n_carry4, 1);
        assert_eq!(layout.n_srlc32e, 1);

        // header keeps the first field array at a 64-byte boundary.
        assert_eq!(HEADER_U64 % 8, 0);
        let mut prev_end = HEADER_U64;
        for s in &layout.sections {
            assert_eq!(s.offset_u64, prev_end, "sections are contiguous & monotonic");
            // one u64 per instance ⇒ every array base is 8-byte aligned.
            assert_eq!((s.offset_u64 * 8) % 8, 0);
            prev_end += s.len;
        }
        assert_eq!(prev_end, layout.total_u64);

        // SoA: DSP field count * instances, etc.
        let dsp_words = DSP_FIELDS.len() * 2;
        let carry_words = CARRY4_FIELDS.len() * 1;
        let srl_words = SRLC32E_FIELDS.len() * 1;
        assert_eq!(layout.total_u64, HEADER_U64 + dsp_words + carry_words + srl_words);
    }

    #[test]
    fn zero_macros_is_just_the_header() {
        let aig = finish(empty_aig());
        let sched = build_schedule(&aig).unwrap();
        let layout = build_macro_layout(&aig, &sched);
        assert_eq!(layout.total_u64, HEADER_U64);
        assert!(layout.sections.is_empty());
        layout.validate().unwrap();
        assert_eq!(layout.header_words()[0], MACRO_LAYOUT_MAGIC);
    }

    #[test]
    fn instance_order_follows_schedule_waves() {
        // carry 1 feeds carry 2: wave 0 then wave 1, so instance 0 then 1.
        let mut aig = empty_aig();
        add_carry4(&mut aig, 1, 0);
        // wire carry 2's CIN to carry 1's CO[3].
        let co3 = aig.carry4s[&1].co_out[3];
        add_carry4(&mut aig, 2, co3 << 1);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let layout = build_macro_layout(&aig, &sched);
        assert_eq!(layout.carry4_cells, vec![1, 2]);
        assert_eq!(layout.operand_maps[0].cell_id, 1);
        assert_eq!(layout.operand_maps[0].instance, 0);
        assert_eq!(layout.operand_maps[1].cell_id, 2);
        assert_eq!(layout.operand_maps[1].instance, 1);
    }

    #[test]
    fn dsp_operand_map_covers_all_operand_bits() {
        let mut aig = empty_aig();
        add_dsp(&mut aig, 7);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let layout = build_macro_layout(&aig, &sched);
        let m = &layout.operand_maps[0];
        // 27 A + 27 D + 18 B + 48 C + 9 OPMODE + 4 ALUMODE + 5 INMODE + CEP + RSTP
        assert_eq!(m.operand_bits.len(), 27 + 27 + 18 + 48 + 9 + 4 + 5 + 2);
        // 48 result bits (all connected in the fixture).
        assert_eq!(m.result_bits.len(), 48);
    }

    #[test]
    fn header_offsets_point_at_the_right_sections() {
        let mut aig = empty_aig();
        add_carry4(&mut aig, 1, 0);
        add_srlc(&mut aig, 2);
        let aig = finish(aig);
        let sched = build_schedule(&aig).unwrap();
        let layout = build_macro_layout(&aig, &sched);
        let h = layout.header_words();
        assert_eq!(h[4], 0, "no DSP section");
        assert_eq!(h[5] as usize, HEADER_U64, "CARRY4 section right after header");
        assert_eq!(h[6] as usize, HEADER_U64 + CARRY4_FIELDS.len());
    }
}
