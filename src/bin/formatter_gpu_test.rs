// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Host-Side Macro Memory Formatter plan, Phase 5 verification harness.
//!
//! Builds a small heterogeneous design (a same-cycle CARRY4 chain + one SRLC32E
//! + one DSP48E2), formats it through the production V2 path, uploads the single
//! immutable program with **one** `UVec` H2D transfer, and asks the GPU to:
//!
//!   * check magic / version / `total_words`,
//!   * check every section is 8- or 16-byte aligned on the device pointer,
//!     monotonic and in bounds,
//!   * recompute the FNV-1a section-fold and compare it to the host
//!     `content_hash`,
//!
//! then runs a coalesced-read probe over the CARRY4 source-selector section so
//! Nsight Compute (Phase 7) can report sectors / request.
//!
//! Build + run (needs an NVIDIA GPU + toolkit):
//!
//! ```bash
//! cargo run --release --features v2 --bin formatter_gpu_test
//! compute-sanitizer --tool memcheck  target/release/formatter_gpu_test
//! compute-sanitizer --tool racecheck target/release/formatter_gpu_test
//! ncu --set full -k "regex:formatter_" -o formatter_v2 target/release/formatter_gpu_test
//! ```
//!
//! Exit code 0 iff every self-check flag is set.

use gem::aig::{Carry4Block, DSPBlock, DriverType, Srlc32eBlock, AIG};
use gem::aigpdk::AIGPDKLeafPins;
use gem::format_v2::SectionKind;
use gem::format_v2_build::build_resolved_program;
use gem::format_v2_cpu::interpret_cycle;
use gem::format_v2_gpu::FlattenedScriptV2;
use gem::schedule::build_schedule;
use gem::schedule::MacroKind;
use netlistdb::NetlistDB;
use ulib::{Device, UVec};

mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v2.rs"));
}

fn iv(pin: usize, inv: usize) -> usize {
    pin << 1 | inv
}

/// x, y primary inputs (pins 1, 2); c1.CO[3] -> c2.CIN (same-cycle chain);
/// one SRLC32E and one DSP48E2 fed from x/y.
fn demo_aig() -> AIG {
    let mut aig = AIG::default();
    aig.num_aigpins = 2;
    aig.drivers = vec![
        DriverType::Tie0,
        DriverType::InputPort(0),
        DriverType::InputPort(1),
    ];

    fn alloc_outs(aig: &mut AIG, driver: impl Fn(usize) -> DriverType, n: usize) -> Vec<usize> {
        (0..n)
            .map(|k| {
                aig.num_aigpins += 1;
                aig.drivers.push(driver(k));
                aig.num_aigpins
            })
            .collect()
    }

    // AIG g1 = x & !y, feeding CARRY4 #1.
    aig.num_aigpins += 1;
    aig.drivers.push(DriverType::AndGate(iv(1, 0), iv(2, 1)));
    let g1 = aig.num_aigpins;

    // CARRY4 #1
    let c1: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::CARRY4(1, k), 8);
    aig.carry4s.insert(
        1,
        Carry4Block {
            s_iv: [iv(g1, 0); 4],
            di_iv: [iv(2, 0); 4],
            cin_iv: 0,
            cyinit_iv: 0,
            o_out: [c1[0], c1[1], c1[2], c1[3]],
            co_out: [c1[4], c1[5], c1[6], c1[7]],
        },
    );
    // CARRY4 #2, CIN <- c1.CO[3] (= c1[7])
    let c2: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::CARRY4(2, k), 8);
    aig.carry4s.insert(
        2,
        Carry4Block {
            s_iv: [iv(1, 0); 4],
            di_iv: [iv(2, 0); 4],
            cin_iv: iv(c1[7], 0),
            cyinit_iv: 0,
            o_out: [c2[0], c2[1], c2[2], c2[3]],
            co_out: [c2[4], c2[5], c2[6], c2[7]],
        },
    );
    // AIG g2 consumes the end of the direct CARRY chain.
    aig.num_aigpins += 1;
    aig.drivers
        .push(DriverType::AndGate(iv(c2[0], 0), iv(1, 0)));
    let g2 = aig.num_aigpins;
    aig.primary_outputs.insert(iv(g2, 0));
    // SRLC32E #3
    let s3: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::SRLC32E(3, k), 2);
    aig.srlc32es.insert(
        3,
        Srlc32eBlock {
            d_iv: iv(1, 0),
            ce_iv: iv(2, 0),
            a_iv: [0; 5],
            clk_iv: 1,
            q_out: s3[0],
            q31_out: s3[1],
            init: 0,
        },
    );
    // DSP48E2 #4
    let p4: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::DSP(4, k), 48);
    let mut dsp = DSPBlock::default();
    dsp.clk_iv = 1;
    dsp.a_iv = [iv(1, 0); 27];
    dsp.d_iv = [iv(2, 0); 27];
    dsp.b_iv = [iv(1, 0); 18];
    dsp.c_iv = [0; 48];
    // OPMODE = 9'h005 (multiply-only): bit 0 and bit 2 set.
    dsp.opmode_iv = [1, 0, 1, 0, 0, 0, 0, 0, 0];
    for (k, o) in p4.iter().enumerate() {
        dsp.p_out[k] = *o;
    }
    aig.dsps.insert(4, dsp);

    aig.fanouts_start = vec![0; aig.num_aigpins + 2];
    aig
}

fn selected_aig() -> AIG {
    let Ok(path) = std::env::var("GEM_V2_GATELEVEL") else {
        return demo_aig();
    };
    let netlist = NetlistDB::from_sverilog_file(path, None, &AIGPDKLeafPins())
        .expect("cannot parse GEM_V2_GATELEVEL");
    AIG::from_netlistdb(&netlist)
}

fn main() {
    let aig = selected_aig();
    let sched = build_schedule(&aig).expect("heterogeneous schedule");
    let rp = build_resolved_program(&aig, &sched).expect("v2 formatter");

    let mut prev = rp.initial_state.clone();
    for (pin, driver) in aig.drivers.iter().enumerate() {
        if matches!(
            driver,
            DriverType::InputPort(_) | DriverType::InputClockFlag(..)
        ) {
            if let Some(&(word, bit)) = rp.state.prev.get(&pin) {
                prev[word as usize] |= 1u32 << bit;
            }
        }
    }
    let cpu = interpret_cycle(&rp, &prev, true).expect("CPU V2 reference");
    let dsp_state_words = rp.state.dsp_p_word.clone();
    let srl_state_words = rp.state.srlc_storage_word.clone();
    let queue_len = |kind| {
        rp.queues
            .iter()
            .find(|q| q.kind == kind)
            .map_or(0, |q| q.cells.len())
    };
    let n_carry = queue_len(MacroKind::Carry4);
    let n_dsp = queue_len(MacroKind::Dsp48e2);
    let n_srl = queue_len(MacroKind::Srlc32e);
    let n_aig = rp.aig_operations.len();
    let mut input_masks_host = vec![0u32; rp.state.persistent_words as usize];
    for (pin, driver) in aig.drivers.iter().enumerate() {
        if matches!(
            driver,
            DriverType::InputPort(_) | DriverType::InputClockFlag(..)
        ) {
            if let Some(&(word, bit)) = rp.state.prev.get(&pin) {
                input_masks_host[word as usize] |= 1u32 << bit;
            }
        }
    }

    let host_hash = rp.layout.header.content_hash;
    let n_sections = rp.layout.sections.len();
    let carry_src = rp
        .layout
        .section(SectionKind::Carry4SourceSel)
        .map(|s| s.start)
        .unwrap_or(0);
    println!(
        "waves={}  sections={}  program {} u64 words ({} bytes)",
        sched.waves.len(),
        n_sections,
        rp.program.len(),
        rp.program.len() * 8
    );

    let fs = FlattenedScriptV2::from_resolved(rp);
    assert!(
        fs.validates(),
        "host validator rejected the assembled program"
    );

    let device = Device::CUDA(0);
    let mut out_hash: UVec<u64> = vec![0u64; n_sections.max(1)].into();
    let mut out_flags: UVec<u32> = vec![0u32; 1].into();

    // one immutable H2D upload happens here (AsUPtr on &fs.program).
    ucci::formatter_gpu_selfcheck(
        &fs.program,
        fs.program.len(),
        host_hash,
        &mut out_hash,
        &mut out_flags,
        device,
    );
    device.synchronize();

    // coalesced-read probe over the CARRY4 source-selector section.
    let lanes = 32usize;
    let mut sink: UVec<u64> = vec![0u64; lanes].into();
    ucci::formatter_coalesced_probe(&fs.program, carry_src as usize, lanes, &mut sink, device);
    device.synchronize();

    let gather_pair_count = fs.staged_word_slots.len() / 2;
    let gather_pairs: UVec<u32> = fs.staged_word_slots.clone().into();
    let dsp_words: UVec<u32> = dsp_state_words.into();
    let srl_words: UVec<u32> = srl_state_words.into();
    let prev_uvec: UVec<u32> = prev.clone().into();
    let input_masks: UVec<u32> = input_masks_host.into();
    let mut next_uvec: UVec<u32> = prev.clone().into();
    let mut current_stage: UVec<u32> = vec![0; prev.len()].into();
    let mut aig_gpu: UVec<u32> = vec![0; n_aig].into();
    let mut carry_gpu: UVec<u64> = vec![0; n_carry].into();
    let mut dsp_gpu: UVec<u64> = vec![0; n_dsp].into();
    let mut srl_gpu: UVec<u64> = vec![0; n_srl].into();
    ucci::evaluate_v2_macro_waves(
        &fs.program,
        fs.program.len(),
        &gather_pairs,
        gather_pair_count,
        &dsp_words,
        n_dsp,
        &srl_words,
        n_srl,
        &prev_uvec,
        prev.len(),
        &input_masks,
        &mut next_uvec,
        &mut current_stage,
        &mut aig_gpu,
        n_aig,
        &mut carry_gpu,
        n_carry,
        &mut dsp_gpu,
        &mut srl_gpu,
        1,
        fs.shared_words_per_block.max(1) as usize,
        device,
    );
    device.synchronize();

    let carry_cpu: Vec<u64> = cpu
        .carry4_out
        .iter()
        .map(|&(o, co)| u64::from(o) | (u64::from(co) << 4))
        .collect();
    let srl_cpu: Vec<u64> = cpu
        .srlc_out
        .iter()
        .map(|&(q, q31, state)| u64::from(q) | (u64::from(q31) << 1) | (u64::from(state) << 32))
        .collect();
    assert_eq!(&carry_gpu[..], &carry_cpu, "CUDA/CPU CARRY4 mismatch");
    assert_eq!(&dsp_gpu[..], &cpu.dsp_p, "CUDA/CPU DSP48E2 mismatch");
    assert_eq!(&srl_gpu[..], &srl_cpu, "CUDA/CPU SRLC32E mismatch");
    assert_eq!(
        &next_uvec[..],
        &cpu.next_state,
        "CUDA/CPU persistent-state mismatch"
    );
    let aig_cpu: Vec<u32> = cpu
        .aig_values
        .iter()
        .map(|(_, value)| u32::from(*value))
        .collect();
    assert_eq!(&aig_gpu[..], &aig_cpu, "CUDA/CPU AIG-operation mismatch");

    let repetitions = std::env::var("GEM_V2_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000usize)
        .max(1);
    let mut elapsed_ns: UVec<u64> = vec![0; 1].into();
    ucci::benchmark_v2_macro_waves(
        &fs.program,
        fs.program.len(),
        &gather_pairs,
        gather_pair_count,
        &dsp_words,
        n_dsp,
        &srl_words,
        n_srl,
        &prev_uvec,
        prev.len(),
        &input_masks,
        &mut next_uvec,
        &mut current_stage,
        &mut aig_gpu,
        n_aig,
        &mut carry_gpu,
        n_carry,
        &mut dsp_gpu,
        &mut srl_gpu,
        1,
        fs.shared_words_per_block.max(1) as usize,
        repetitions,
        &mut elapsed_ns,
        device,
    );
    device.synchronize();
    let seconds = elapsed_ns[0] as f64 / 1e9;
    let executions_per_second = repetitions as f64 / seconds;
    let operations_per_execution = (n_aig + n_carry + n_dsp + n_srl) as f64;
    println!(
        "benchmark: repetitions={} elapsed_ms={:.3} kernel_executions_per_s={:.3} operation_evaluations_per_s={:.3}",
        repetitions,
        seconds * 1e3,
        executions_per_second,
        executions_per_second * operations_per_execution,
    );

    let flags = out_flags[0];
    const WANT: u32 = 0b1_1111;
    let names = [
        "magic",
        "version",
        "total_words",
        "section layout",
        "content hash",
    ];
    for (i, n) in names.iter().enumerate() {
        println!(
            "  [{}] {:<16} {}",
            i,
            n,
            if flags >> i & 1 == 1 { "ok" } else { "FAIL" }
        );
    }
    for (i, h) in out_hash.iter().enumerate() {
        println!("  section {i} device checksum {:#018x}", h);
    }
    println!("program uploaded once: {} bytes", fs.program_bytes());

    if flags == WANT {
        println!("PASS: formatter and dependency-wave CUDA execution match CPU V2");
        std::process::exit(0);
    } else {
        eprintln!("FAIL: flags = {:#07b}, want {:#07b}", flags, WANT);
        std::process::exit(1);
    }
}
