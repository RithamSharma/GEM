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

use gem::aig::{Carry4Block, DriverType, DSPBlock, Srlc32eBlock, AIG};
use gem::format_v2::SectionKind;
use gem::format_v2_build::build_resolved_program;
use gem::format_v2_gpu::FlattenedScriptV2;
use gem::schedule::build_schedule;
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

    // CARRY4 #1
    let c1: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::CARRY4(1, k), 8);
    aig.carry4s.insert(
        1,
        Carry4Block {
            s_iv: [iv(1, 0); 4],
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
    // SRLC32E #3
    let s3: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::SRLC32E(3, k), 2);
    aig.srlc32es.insert(
        3,
        Srlc32eBlock {
            d_iv: iv(1, 0),
            ce_iv: iv(2, 0),
            a_iv: [0; 5],
            clk_iv: 0,
            q_out: s3[0],
            q31_out: s3[1],
            init: 0,
        },
    );
    // DSP48E2 #4
    let p4: Vec<usize> = alloc_outs(&mut aig, |k| DriverType::DSP(4, k), 48);
    let mut dsp = DSPBlock::default();
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

fn main() {
    let aig = demo_aig();
    let sched = build_schedule(&aig).expect("heterogeneous schedule");
    let rp = build_resolved_program(&aig, &sched).expect("v2 formatter");

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
    assert!(fs.validates(), "host validator rejected the assembled program");

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

    let flags = out_flags[0];
    const WANT: u32 = 0b1_1111;
    let names = ["magic", "version", "total_words", "section layout", "content hash"];
    for (i, n) in names.iter().enumerate() {
        println!("  [{}] {:<16} {}", i, n, if flags >> i & 1 == 1 { "ok" } else { "FAIL" });
    }
    for (i, h) in out_hash.iter().enumerate() {
        println!("  section {i} device checksum {:#018x}", h);
    }
    println!("program uploaded once: {} bytes", fs.program_bytes());

    if flags == WANT {
        println!("PASS");
        std::process::exit(0);
    } else {
        eprintln!("FAIL: flags = {:#07b}, want {:#07b}", flags, WANT);
        std::process::exit(1);
    }
}
