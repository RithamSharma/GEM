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
    content_hash, decode_selector_section, DestinationSel, DestinationSpace, SectionKind, SourceSel,
    SourceSpace,
};
use crate::format_v2_build::{dst_fields, src_fields, ResolvedProgram};
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
}

fn read_src(sel: &Option<SourceSel>, shared: &[u64]) -> bool {
    match sel {
        None => false,
        Some(s) => {
            // post-gather every operand is Constant or LocalShared.
            let raw = match s.space {
                SourceSpace::Constant => false,
                SourceSpace::LocalShared
                | SourceSpace::PreviousState
                | SourceSpace::CurrentStage => {
                    (shared[s.index as usize] >> s.bit) & 1 == 1
                }
            };
            raw ^ s.invert
        }
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
    next_state: &mut [u32],
) {
    for &(flat_bit, val) in bits {
        let Some(sel) = dst.get(flat_bit).and_then(|row| row.get(inst)).and_then(|o| *o) else {
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
                let w = &mut next_state[sel.index as usize];
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
    if prev.len() != rp.state.persistent_words as usize {
        return Err(format!(
            "prev image is {} words, expected {}",
            prev.len(),
            rp.state.persistent_words
        ));
    }
    // ABI self-check: the program the interpreter is about to read must pass the
    // same validator the GPU loader runs.
    crate::format_v2::validate(&rp.layout, rp.program.len() as u64).map_err(|e| format!("{e:?}"))?;
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
    let mut carry4_by_inst: Vec<(u8, u8)> = Vec::new();
    let mut dsp_by_inst: Vec<u64> = Vec::new();
    let mut srlc_by_inst: Vec<(bool, bool, u32)> = Vec::new();

    let num_waves = rp
        .queues
        .iter()
        .flat_map(|q| q.waves.iter())
        .copied()
        .max()
        .map_or(0, |m| m + 1);

    for wave in 0..num_waves {
        for q in &qd {
            for inst in 0..q.cells.len() {
                if q.waves[inst] != wave {
                    continue;
                }
                // read all operand bits, releasing the &shared borrow.
                let get: Vec<bool> = (0..q.src.len())
                    .map(|fb| read_src(&q.src[fb][inst], &shared))
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
                        write_result(&q.dst, inst, &bits, &mut shared, &mut next_state);
                        ensure_len(&mut carry4_by_inst, inst, (0, 0));
                        carry4_by_inst[inst] = (r.o, r.co);
                    }
                    MacroKind::Srlc32e => {
                        let d = get[0];
                        let ce = get[1];
                        let a = pack(&get, 2, 7) as u8;
                        let storage_word = rp.state.srlc_storage_word[inst] as usize;
                        let cur = prev[storage_word];
                        let (outs, nxt) = srlc32e_step(cur, d, ce, rising_edge, a);
                        write_result(
                            &q.dst,
                            inst,
                            &[(0, outs.q), (1, outs.q31)],
                            &mut shared,
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
                        let pw = rp.state.dsp_p_word[inst] as usize;
                        let p_cur = u64::from(prev[pw]) | (u64::from(prev[pw + 1]) << 32);
                        let (mode, preadd) = decode_dsp_controls(opmode, alumode, inmode)
                            .map_err(|e| format!("DSP {inst} unsupported controls: {e:?}"))?;
                        let next_p = if rstp {
                            0
                        } else if !cep {
                            p_cur & ((1u64 << 48) - 1)
                        } else {
                            dsp48e2_next(a, b, c, d, p_cur, mode, preadd)
                        };
                        let bits: Vec<(usize, bool)> =
                            (0..48).map(|k| (k, (next_p >> k) & 1 != 0)).collect();
                        write_result(&q.dst, inst, &bits, &mut shared, &mut next_state);
                        next_state[pw] = (next_p & 0xFFFF_FFFF) as u32;
                        next_state[pw + 1] = (next_p >> 32) as u32;
                        ensure_len(&mut dsp_by_inst, inst, 0u64);
                        dsp_by_inst[inst] = next_p;
                    }
                }
            }
        }
    }

    Ok(CycleResult {
        next_state,
        carry4_out: carry4_by_inst,
        dsp_p: dsp_by_inst,
        srlc_out: srlc_by_inst,
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
    use crate::aig::{Carry4Block, DriverType, Srlc32eBlock, AIG};
    use crate::format_v2_build::build_resolved_program;
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
            Carry4Block { di_iv: di, s_iv: s, cin_iv, cyinit_iv, o_out, co_out },
        );
        (o_out, co_out)
    }
    fn finish(mut aig: AIG) -> AIG {
        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        aig
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
                clk_iv: 0,
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
