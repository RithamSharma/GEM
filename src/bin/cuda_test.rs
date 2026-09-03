// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use compact_str::CompactString;
use gem::aig::{DriverType, AIG};
use gem::aigpdk::{AIGPDKLeafPins, AIGPDK_SRAM_SIZE};
use gem::flatten::FlattenedScriptV1;
use gem::pe::Partition;
use gem::staging::build_staged_aigs;
#[cfg(feature = "v2")]
use gem::{
    format_v2_build::{build_partitioned_program, ResolvedProgram},
    format_v2_gpu::FlattenedScriptV2,
    hetero_parts::{GemPartsV2, HeteroPlacementV2},
    schedule::{build_schedule, MacroKind},
};
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::path::PathBuf;
use std::rc::Rc;
use sverilogparse::SVerilogRange;
use ulib::{AsUPtrMut, Device, UVec};
use vcd_ng::{
    FFValueChange, FastFlow, FastFlowToken, Parser, Scope, ScopeItem, SimulationCommand, Var,
    Writer,
};

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,
    /// Input path for the serialized partitions.
    gemparts: PathBuf,
    /// VCD input signal path
    input_vcd: String,
    /// The scope path of top module in the input VCD.
    ///
    /// If not specified, we will use a flat view.
    /// (this view is often incorrect..)
    #[clap(long)]
    input_vcd_scope: Option<String>,
    /// Output VCD path (must be writable)
    output_vcd: String,
    /// The scope path of top module in the output VCD.
    ///
    /// If not specified, we will use `gem_top_module`.
    #[clap(long)]
    output_vcd_scope: Option<String>,
    /// the number of CUDA blocks to map and execute with.
    ///
    /// should not exceed GPU maximum simutaneous occupancy.
    num_blocks: usize,
    /// Whether to run a sanity check against CPU baseline on finish.
    #[clap(long)]
    check_with_cpu: bool,
    /// Limit the number of simulated cycles to no more than this.
    #[clap(long)]
    max_cycles: Option<usize>,
    /// Use the unified dependency-wave V2 engine for AIG, DSP48E2, CARRY4 and
    /// SRLC32E, DFF and synchronous SRAM endpoints. Equivalent to `--engine v2`.
    #[clap(long)]
    v2: bool,
    /// Which simulation engine to use:
    ///   `auto` (default) - pick V1 or V2 per design: V2 is forced whenever a
    ///                      macro output feeds a same-cycle consumer (V1 would
    ///                      read stale state there); otherwise the faster of the
    ///                      two is chosen from a cost estimate.
    ///   `v1`             - the classic bit-parallel Boomerang engine.
    ///   `v2`             - the heterogeneous wave engine.
    #[clap(long, value_enum, default_value = "auto")]
    engine: EngineChoice,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum EngineChoice {
    Auto,
    V1,
    V2,
}

/// Hierarchical name representation in VCD.
#[derive(PartialEq, Eq, Clone, Debug)]
struct VCDHier {
    cur: CompactString,
    prev: Option<Rc<VCDHier>>,
}

/// Reverse iterator of a [`VCDHier`], yielding cell names
/// from the bottom to the top module.
struct VCDHierRevIter<'i>(Option<&'i VCDHier>);

impl<'i> Iterator for VCDHierRevIter<'i> {
    type Item = &'i CompactString;

    #[inline]
    fn next(&mut self) -> Option<&'i CompactString> {
        let name = self.0?;
        if name.cur.is_empty() {
            return None;
        }
        let ret = &name.cur;
        self.0 = name.prev.as_ref().map(|a| a.as_ref());
        Some(ret)
    }
}

impl<'i> IntoIterator for &'i VCDHier {
    type Item = &'i CompactString;
    type IntoIter = VCDHierRevIter<'i>;

    #[inline]
    fn into_iter(self) -> VCDHierRevIter<'i> {
        VCDHierRevIter(Some(self))
    }
}

impl Hash for VCDHier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for s in self.iter() {
            s.hash(state);
        }
    }
}

#[allow(dead_code)]
impl VCDHier {
    #[inline]
    fn single(cur: CompactString) -> Self {
        VCDHier { cur, prev: None }
    }

    #[inline]
    fn empty() -> Self {
        VCDHier {
            cur: "".into(),
            prev: None,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.cur.as_str() == "" && self.prev.is_none()
    }

    #[inline]
    fn iter(&self) -> VCDHierRevIter {
        (&self).into_iter()
    }
}

/// Try to match one component in a scope.
/// If succeed, returns the remaining scope (can be None itself indicating
/// all paths matched).
/// If fails, return None.
fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
    if scope.len() == 0 {
        return Some("");
    }
    if scope.starts_with('/') {
        scope = &scope[1..];
    }
    if scope.len() == 0 {
        Some("")
    } else if scope.starts_with(cur) {
        if scope.len() == cur.len() {
            Some("")
        } else if scope.as_bytes()[cur.len()] == b'/' {
            Some(&scope[cur.len() + 1..])
        } else {
            None
        }
    } else {
        None
    }
}

fn find_top_scope<'i>(items: &'i [ScopeItem], top_scope: &'_ str) -> Option<&'i Scope> {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            if let Some(s1) = match_scope_path(top_scope, scope.identifier.as_str()) {
                return match s1 {
                    "" => Some(scope),
                    _ => find_top_scope(&scope.children[..], s1),
                };
            }
        }
    }
    None
}

/// CPU prototype partition executor for script version 1.
fn simulate_block_v1(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
    debug_verbose: bool,
) {
    let mut script_pi = 0;
    loop {
        let num_stages = script[script_pi];
        let is_last_part = script[script_pi + 1];
        let num_ios = script[script_pi + 2];
        let io_offset = script[script_pi + 3];
        let num_srams = script[script_pi + 4];
        let sram_offset = script[script_pi + 5];
        let num_global_read_rounds = script[script_pi + 6];
        let num_output_duplicates = script[script_pi + 7];
        let num_dsps = script[script_pi + 8];
        let num_carry4s = script[script_pi + 9];
        let num_srlc32es = script[script_pi + 10];
        let macro_output_words = num_dsps * 2 + num_carry4s + num_srlc32es * 2;
        let macro_input_words = num_dsps * 5 + num_carry4s + num_srlc32es;
        let duplicate_start = num_ios - num_srams - num_output_duplicates;
        let macro_input_start = duplicate_start - macro_input_words;
        let normal_writeouts = macro_input_start - macro_output_words;
        let mut writeout_hooks = vec![0; 256];
        for i in 0..128 {
            let t = script[script_pi + 128 + i];
            writeout_hooks[i * 2] = (t & ((1 << 16) - 1)) as u16;
            writeout_hooks[i * 2 + 1] = (t >> 16) as u16;
        }
        if num_stages == 0 {
            script_pi += 256;
            break;
        }
        // assert_eq!(part.stages.len(), num_stages as usize);
        // assert_eq!(part.stages.iter().map(|s| s.write_outs.len()).sum::<usize>(), (num_ios - num_srams - num_output_duplicates) as usize);
        script_pi += 256;
        let mut writeouts = vec![0u32; num_ios as usize];

        let mut state = vec![0u32; 256];
        for _gr_i in 0..num_global_read_rounds {
            for i in 0..256 {
                let mut cur_state = state[i];
                let idx = script[script_pi + (i * 2)];
                let mut mask = script[script_pi + (i * 2 + 1)];
                if mask == 0 {
                    continue;
                }
                let value = match (idx >> 31) != 0 {
                    false => input_state[idx as usize],
                    true => output_state[(idx ^ (1 << 31)) as usize],
                };
                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 {
                        cur_state |= 1;
                    }
                    mask ^= lowbit;
                }
                state[i] = cur_state;
            }
            script_pi += 256 * 2;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 0");
            println!("global read states:");
            for i in 0..256 {
                println!(" [{}] = {}", i, state[i]);
            }
        }

        for bs_i in 0..num_stages {
            let mut hier_inputs = vec![0; 256];
            let mut hier_flag_xora = vec![0; 256];
            let mut hier_flag_xorb = vec![0; 256];
            let mut hier_flag_orb = vec![0; 256];
            for k_outer in 0..4 {
                for i in 0..256 {
                    for k_inner in 0..4 {
                        let k = k_outer * 4 + k_inner;
                        let t_shuffle = script[script_pi + i * 4 + k_inner];
                        let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u16;
                        let t_shuffle_2_idx = (t_shuffle >> 16) as u16;
                        hier_inputs[i] |=
                            (state[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1)
                                << (k * 2);
                        hier_inputs[i] |=
                            (state[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1)
                                << (k * 2 + 1);
                    }
                }
                script_pi += 256 * 4;
            }
            for i in 0..256 {
                hier_flag_xora[i] = script[script_pi + i * 4];
                hier_flag_xorb[i] = script[script_pi + i * 4 + 1];
                hier_flag_orb[i] = script[script_pi + i * 4 + 2];
            }
            script_pi += 256 * 4;

            if debug_verbose {
                println!("debug_verbose STAGE 1.1 bs_i {bs_i}");
                println!("after local shuffle:");
                for i in 0..256 {
                    println!(" [{}] = {}", i, hier_inputs[i]);
                }
            }

            // hier[0]
            for i in 0..128 {
                let a = hier_inputs[i];
                let b = hier_inputs[128 + i];
                let xora = hier_flag_xora[128 + i];
                let xorb = hier_flag_xorb[128 + i];
                let orb = hier_flag_orb[128 + i];
                let ret = (a ^ xora) & ((b ^ xorb) | orb);
                hier_inputs[128 + i] = ret;
            }
            // hier 1 to 7
            for hi in 1..=7 {
                let hier_width = 1 << (7 - hi);
                for i in 0..hier_width {
                    let a = hier_inputs[hier_width * 2 + i];
                    let b = hier_inputs[hier_width * 3 + i];
                    let xora = hier_flag_xora[hier_width + i];
                    let xorb = hier_flag_xorb[hier_width + i];
                    let orb = hier_flag_orb[hier_width + i];
                    let ret = (a ^ xora) & ((b ^ xorb) | orb);
                    // for k in 0..32 {
                    //     let apin = part.stages[bs_i as usize].hier[hi][i * 32 + k];
                    //     let bpin = part.stages[bs_i as usize].hier[hi][part.stages[bs_i as usize].hier[hi + 1].len() + i * 32 + k];
                    //     let opin = part.stages[bs_i as usize].hier[hi + 1][i * 32 + k];
                    //     if [21876 / 2].contains(&opin) {
                    //         println!("Got ai gate at part {} bs_i {bs_i} hi {hi} i {i} k {k} (pos {} put {}): {opin}={} <- f[{apin}={} ^{}, {bpin}={} ^{}|{}]", parts_indices[part_i_dbg - 1], i * 32 + k, hier_width * 32 + i * 32 + k, ret >> k & 1, a >> k & 1, xora >> k & 1, b >> k & 1, xorb >> k & 1, orb >> k & 1);
                    //     }
                    // }
                    hier_inputs[hier_width + i] = ret;
                }
            }
            // hier 8,9,10,11,12
            let v1 = hier_inputs[1];
            let xora = hier_flag_xora[0];
            let xorb = hier_flag_xorb[0];
            let orb = hier_flag_orb[0];
            let r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;
            let r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;
            let r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;
            let r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;
            let r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;
            hier_inputs[0] = r8 | r9 | r10 | r11 | r12;

            state = hier_inputs;

            if debug_verbose {
                println!("debug_verbose STAGE 1.2 bs_i {bs_i}");
                println!("after and-invert:");
                for i in 0..256 {
                    println!(" [{}] = {}", i, state[i]);
                }
            }

            for i in 0..256 {
                let hooki = writeout_hooks[i];
                if (hooki >> 8) as u32 == bs_i {
                    writeouts[i] = state[(hooki & 255) as usize];
                }
            }
        }

        let gather_words = num_srams * 4 + num_output_duplicates + macro_input_words;
        let mut sram_duplicate_perm = vec![0u32; gather_words as usize];
        for k_outer in 0..4 {
            for i in 0..gather_words {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1)
                            << (k * 2);
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1)
                            << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..gather_words as usize {
            sram_duplicate_perm[i] &= !script[script_pi + i * 4 + 1];
            sram_duplicate_perm[i] ^= script[script_pi + i * 4];
        }
        script_pi += 256 * 4;

        for sram_i_u32 in 0..num_srams {
            let sram_i = sram_i_u32 as usize;
            let addrs = sram_duplicate_perm[sram_i * 4];
            let port_r_addr_iv = addrs & 0xffff;
            let port_w_addr_iv = (addrs & 0xffff0000) >> 16;
            let port_w_wr_en = sram_duplicate_perm[sram_i * 4 + 1];
            let port_w_wr_data_iv = sram_duplicate_perm[sram_i * 4 + 2];

            let sram_st = sram_offset as usize + sram_i * AIGPDK_SRAM_SIZE;
            let sram_ed = sram_st + AIGPDK_SRAM_SIZE;
            let ram = &mut sram_data[sram_st..sram_ed];
            let r = ram[port_r_addr_iv as usize];
            let w0 = ram[port_w_addr_iv as usize];
            writeouts[(num_ios - num_srams + sram_i_u32) as usize] = r;
            ram[port_w_addr_iv as usize] =
                (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
            // println!("sram for part id {} index {sram_i_u32}: port_r_addr_iv {port_r_addr_iv} port_w_addr_iv {port_w_addr_iv} port_w_wr_en {port_w_wr_en} port_w_wr_data_iv {port_w_wr_data_iv}", parts_indices[part_i_dbg - 1]);
        }

        for i in 0..num_output_duplicates {
            writeouts[(duplicate_start + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + i) as usize];
        }

        for i in 0..macro_input_words {
            writeouts[(macro_input_start + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + num_output_duplicates + i) as usize];
        }

        use gem::primitive_models::{carry4, decode_dsp_controls, dsp48e2_next, srlc32e_step};
        for i in 0..num_dsps {
            let start = (macro_input_start + i * 5) as usize;
            let d0 = writeouts[start];
            let d1 = writeouts[start + 1];
            let d2 = writeouts[start + 2];
            let d3 = writeouts[start + 3];
            let d4 = writeouts[start + 4];
            let a = d0 & 0x07ff_ffff;
            let b = (d0 >> 30) | ((d1 & 0xffff) << 2);
            let c = u64::from(d1 >> 16) | (u64::from(d2) << 16);
            let d = d3 & 0x07ff_ffff;
            let opmode = ((d3 >> 27) | ((d4 & 0xf) << 5)) as u16;
            let alumode = ((d4 >> 4) & 0xf) as u8;
            let inmode = ((d4 >> 8) & 0x1f) as u8;
            let cep = d4 >> 13 & 1 != 0;
            let rstp = d4 >> 14 & 1 != 0;
            let out = (normal_writeouts + i * 2) as usize;
            let previous = u64::from(input_state[io_offset as usize + out])
                | (u64::from(input_state[io_offset as usize + out + 1]) << 32);
            let next = if rstp {
                0
            } else if !cep {
                previous
            } else {
                let (mode, preadd) = decode_dsp_controls(opmode, alumode, inmode)
                    .unwrap_or_else(|error| panic!("unsupported DSP controls: {error:?}"));
                dsp48e2_next(a, b, c, d, previous, mode, preadd)
            };
            writeouts[out] = next as u32;
            writeouts[out + 1] = (next >> 32) as u32;
        }
        for i in 0..num_carry4s {
            let packed = writeouts[(macro_input_start + num_dsps * 5 + i) as usize];
            let mut di = 0_u8;
            let mut s = 0_u8;
            for bit in 0..4 {
                di |= (((packed >> (bit * 2)) & 1) as u8) << bit;
                s |= (((packed >> (bit * 2 + 1)) & 1) as u8) << bit;
            }
            let result = carry4(s, di, packed >> 8 & 1 != 0, packed >> 9 & 1 != 0);
            writeouts[(normal_writeouts + num_dsps * 2 + i) as usize] =
                u32::from(result.o) | (u32::from(result.co) << 4);
        }
        for i in 0..num_srlc32es {
            let packed = writeouts[(macro_input_start + num_dsps * 5 + num_carry4s + i) as usize];
            let out = (normal_writeouts + num_dsps * 2 + num_carry4s + i * 2) as usize;
            let current = input_state[io_offset as usize + out + 1];
            let (outputs, next) = srlc32e_step(
                current,
                packed & 1 != 0,
                packed >> 1 & 1 != 0,
                packed >> 2 & 1 != 0,
                ((packed >> 3) & 0x1f) as u8,
            );
            writeouts[out] = u32::from(outputs.q) | (u32::from(outputs.q31) << 1);
            writeouts[out + 1] = next;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 2");
            println!("before writeout_inv:");
            for i in 0..256 {
                println!(
                    " [{}] = {}",
                    i,
                    if i < num_ios as usize {
                        writeouts[i]
                    } else {
                        0
                    }
                );
            }
        }

        let mut clken_perm = vec![0u32; num_ios as usize];
        let writeouts_for_clken = writeouts.clone();
        for k_outer in 0..4 {
            for i in 0..num_ios {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_1_idx >> 5) as usize]
                        >> (t_shuffle_1_idx & 31)
                        & 1)
                        << (k * 2);
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_2_idx >> 5) as usize]
                        >> (t_shuffle_2_idx & 31)
                        & 1)
                        << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }
        script_pi += 256 * 4;
        // println!("test: clken_perm {:?}", clken_perm);

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 3");
            println!("final writeout:");
            for i in 0..num_ios {
                println!(
                    " [{}] [global {}] = {}",
                    i,
                    io_offset + i,
                    output_state[(io_offset + i) as usize]
                );
            }
        }

        if is_last_part != 0 {
            break;
        }
    }
    assert_eq!(script_pi, script.len());
}

mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}

#[cfg(feature = "v2")]
mod uccv2 {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v2.rs"));
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("cuda_test");
    clilog::enable_timer("gem");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    )
    .expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);
    let stageds = build_staged_aigs(&aig, &args.level_split);

    let wants_v2_explicit = args.v2 || args.engine == EngineChoice::V2;

    // ---- engine dispatch (V1 vs V2) --------------------------------------
    // `auto` is correctness-driven: V2 is *required* whenever a macro output is
    // consumed in the same cycle (by another macro or by an AIG region), because
    // the classic batched V1 path evaluates all AIG then all macros and would
    // read the previous cycle's value on that edge. Everywhere else V1 (which on
    // a *macro-preserved* netlist evaluates the macros natively but batched) is
    // correct, and on these small preserved graphs it avoids V2's per-cycle
    // wave barriers and bit-serial operand assembly. `--engine v1|v2` / `--v2`
    // override; `--engine v1` is refused when it would be incorrect.
    #[cfg(feature = "v2")]
    let (use_v2, v2_schedule) = {
        let wants_v1_explicit = args.engine == EngineChoice::V1 && !args.v2;
        let probe = build_schedule(&aig).ok();
        let v1_unsafe = probe.as_ref().is_some_and(|s| !s.v1_batched_is_safe());
        let n_m2c = probe.as_ref().map_or(0, |s| {
            s.edges_same
                .iter()
                .filter(|&&(u, _)| s.nodes[u].macro_kind().is_some())
                .count()
        });
        let n_macros = aig.dsps.len() + aig.carry4s.len() + aig.srlc32es.len();
        let chosen = if wants_v2_explicit {
            true
        } else if wants_v1_explicit {
            if v1_unsafe {
                panic!(
                    "--engine v1: this design has {n_m2c} same-cycle macro->consumer edge(s); \
                     the batched V1 path would read stale state. Use --engine v2 or auto."
                );
            }
            false
        } else if v1_unsafe {
            clilog::info!(
                "engine=auto: {n_m2c} same-cycle macro->consumer edge(s) -> V2 (required for correctness)"
            );
            true
        } else {
            clilog::info!(
                "engine=auto: {n_macros} macro(s), none feeding a same-cycle consumer -> V1 \
                 (batched, correct; force --engine v2 to compare)"
            );
            false
        };
        (chosen, if chosen { probe } else { None })
    };
    #[cfg(not(feature = "v2"))]
    let use_v2 = {
        if wants_v2_explicit {
            panic!("--engine v2 / --v2 requires building cuda_test with --features v2");
        }
        false
    };
    clilog::info!("selected simulation engine: {}", if use_v2 { "V2 (heterogeneous)" } else { "V1 (Boomerang)" });
    let parts_bytes = std::fs::read(&args.gemparts).expect("cannot read .gemparts");
    #[cfg(feature = "v2")]
    let (parts_in_stages, v2_placement): (Vec<Vec<Partition>>, Option<HeteroPlacementV2>) = if use_v2
    {
        match serde_bare::from_slice::<GemPartsV2>(&parts_bytes) {
            Ok(file) => {
                file.validate(v2_schedule.as_ref().unwrap())
                    .expect("invalid/stale heterogeneous .gemparts V2 payload");
                (file.legacy, Some(file.hetero))
            }
            Err(_) => {
                let legacy: Vec<Vec<Partition>> = serde_bare::from_slice(&parts_bytes)
                    .expect("invalid legacy or V2 .gemparts file");
                let placement =
                    HeteroPlacementV2::build(v2_schedule.as_ref().unwrap(), args.num_blocks.max(1))
                        .expect("cannot derive V2 placement from legacy parts");
                clilog::warn!("legacy .gemparts detected; derived V2 placement in memory");
                (legacy, Some(placement))
            }
        }
    } else {
        // A V2 artifact deliberately embeds the legacy Boomerang partitions.
        // Let the existing V1 simulator consume that embedded payload too, so
        // users do not need two subtly different `.gemparts` files.
        let legacy = match serde_bare::from_slice::<GemPartsV2>(&parts_bytes) {
            Ok(file)
                if file.magic == gem::hetero_parts::GEM_PARTS_V2_MAGIC
                    && file.version == gem::hetero_parts::GEM_PARTS_V2_VERSION =>
            {
                file.legacy
            }
            _ => serde_bare::from_slice(&parts_bytes)
                .expect("invalid legacy or V2 .gemparts file"),
        };
        (legacy, None)
    };
    #[cfg(not(feature = "v2"))]
    let parts_in_stages: Vec<Vec<Partition>> =
        serde_bare::from_slice(&parts_bytes).expect("invalid legacy .gemparts file");
    clilog::info!(
        "# of effective partitions in each stage: {:?}",
        parts_in_stages
            .iter()
            .map(|ps| ps.len())
            .collect::<Vec<_>>()
    );

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig,
        &stageds
            .iter()
            .map(|(_, _, staged)| staged)
            .collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        args.num_blocks,
        input_layout,
    );

    #[cfg(feature = "v2")]
    let v2_context: Option<(ResolvedProgram, FlattenedScriptV2)> = if use_v2 {
        let schedule = v2_schedule.as_ref().unwrap();
        v2_placement
            .as_ref()
            .unwrap()
            .validate(schedule)
            .expect("heterogeneous placement does not match the netlist schedule");
        let resolved = build_partitioned_program(&aig, schedule, v2_placement.as_ref().unwrap())
            .expect("cannot format partitioned V2 program");
        let flattened = FlattenedScriptV2::from_resolved(resolved.clone());
        Some((resolved, flattened))
    } else {
        None
    };

    #[cfg(feature = "v2")]
    let (input_map, output_map, initial_state, state_size) =
        if let Some((resolved, _)) = &v2_context {
            let inputs = resolved
                .state
                .prev
                .iter()
                .map(|(&pin, &(word, bit))| (pin, word * 32 + bit))
                .collect();
            let outputs = resolved
                .state
                .outputs
                .iter()
                .map(|(&pin, &(word, bit))| (pin, word * 32 + bit))
                .collect();
            (
                inputs,
                outputs,
                resolved.initial_state.clone(),
                resolved.state.persistent_words as usize,
            )
        } else {
            (
                script.input_map.clone(),
                script.output_map.clone(),
                script.initial_reg_io_state.clone(),
                script.reg_io_state_size as usize,
            )
        };
    #[cfg(not(feature = "v2"))]
    let (input_map, output_map, initial_state, state_size) = (
        script.input_map.clone(),
        script.output_map.clone(),
        script.initial_reg_io_state.clone(),
        script.reg_io_state_size as usize,
    );

    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut s = DefaultHasher::new();
    script.blocks_data.hash(&mut s);
    println!("Script hash: {}", s.finish());

    // simulate with the script.
    let input_vcd = File::open(&args.input_vcd).unwrap();
    let mut bufrd = BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().unwrap();
    drop(vcd_parser);
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = FastFlow::new(vcd_file, 65536);

    let top_scope = find_top_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref().unwrap_or(""),
    )
    .expect("Specified top scope not found in VCD.");

    let mut vcd2inp = HashMap::new();
    let mut vcd_widths = HashMap::new();
    let mut inp_port_given = HashSet::new();

    let mut match_one_input = |var: &Var, i: Option<isize>, vcd_pos: usize| {
        let key = (VCDHier::empty(), var.reference.as_str(), i);
        if let Some(&id) = netlistdb.pinname2id.get(&key as &dyn GeneralPinName) {
            if netlistdb.pindirect[id] != Direction::O {
                return;
            }
            vcd2inp.insert((var.code.0, vcd_pos), id);
            inp_port_given.insert(id);
        }
    };
    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            vcd_widths.insert(var.code.0, var.size as usize);
            use vcd_ng::ReferenceIndex::*;
            match var.index {
                None => match var.size {
                    1 => match_one_input(var, None, 0),
                    w @ _ => {
                        for (pos, i) in (0..w).rev().enumerate() {
                            match_one_input(var, Some(i as isize), pos)
                        }
                    }
                },
                Some(BitSelect(i)) => match_one_input(var, Some(i as isize), 0),
                Some(Range(a, b)) => {
                    for (pos, i) in SVerilogRange(a as isize, b as isize).enumerate() {
                        match_one_input(var, Some(i), pos);
                    }
                }
            }
        }
    }
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I && !inp_port_given.contains(&i) {
            clilog::warn!(
                GATESIM_VCDI_MISSING_PI,
                "Primary input port {:?} not present in \
                 the VCD input",
                netlistdb.pinnames[i]
            );
        }
    }

    // open out
    let write_buf = File::create(&args.output_vcd).unwrap();
    let write_buf = BufWriter::new(write_buf);
    let mut writer = Writer::new(write_buf);
    if let Some((ratio, unit)) = header.timescale {
        writer.timescale(ratio, unit).unwrap();
    }
    let output_vcd_scope = args.output_vcd_scope.as_deref().unwrap_or("gem_top_module");
    let output_vcd_scope = output_vcd_scope.split('/').collect::<Vec<_>>();
    for &scope in &output_vcd_scope {
        writer.add_module(scope).unwrap();
    }
    let out2vcd = netlistdb
        .cell2pin
        .iter_set(0)
        .filter_map(|i| {
            if netlistdb.pindirect[i] == Direction::I {
                let aigpin = aig.pin2aigpin_iv[i];
                if matches!(aig.drivers[aigpin >> 1], DriverType::InputPort(_)) {
                    clilog::info!(
                        "skipped output for port {} as it is a pass-through of input port.",
                        netlistdb.pinnames[i].dbg_fmt_pin()
                    );
                    return None;
                }
                if aigpin <= 1 {
                    return Some((
                        aigpin,
                        u32::MAX,
                        writer
                            .add_wire(1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin()))
                            .unwrap(),
                    ));
                }
                Some((
                    aigpin,
                    *output_map.get(&aigpin).unwrap(),
                    writer
                        .add_wire(1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin()))
                        .unwrap(),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for _ in 0..output_vcd_scope.len() {
        writer.upscope().unwrap();
    }
    writer.enddefinitions().unwrap();
    writer.begin(SimulationCommand::Dumpvars).unwrap();

    // do simulation
    let mut state = initial_state;

    // the simulator keeps 2 previous timestamps.
    // vcd_time: the last seen timestamp.
    // vcd_time_last_active: the last timestamp strictly before vcd_time that has
    // active events (e.g., watched clock posedge).
    //
    // when a new timestamp arrives and vcd_time has active events, we simulate
    // the circuit with {actived edge flags from vcd_time}, but do NOT include the
    // input port value changes. then, we associate the result output port values to
    // vcd_time_last_active.
    //
    // the above complexity rises from the necessity to emulate {update, then propagate}
    // behavior from our actual {propagate, then update} implementation.
    let mut vcd_time_last_active = u64::MAX;
    let mut vcd_time = 0;
    let mut last_vcd_time_active = true;
    let mut delayed_bit_changes = HashSet::new();

    let mut input_states = Vec::new();
    let mut offsets_timestamps = Vec::new();

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time {
                    continue;
                }
                if last_vcd_time_active {
                    // clilog::debug!("simulating t={}", vcd_time);
                    input_states.extend(state.iter().copied());
                    offsets_timestamps.push((input_states.len(), vcd_time_last_active));
                    // reset for next timestamp
                    for (_, &(pe, ne)) in &aig.clock_pin2aigpins {
                        if pe != usize::MAX {
                            let p = *input_map.get(&pe).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                        if ne != usize::MAX {
                            let p = *input_map.get(&ne).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                    }
                    if let Some(max_cycles) = args.max_cycles {
                        if offsets_timestamps.len() >= max_cycles {
                            clilog::info!("reached maximum cycles, stop reading input vcd");
                            break;
                        }
                    }
                }
                if last_vcd_time_active {
                    vcd_time_last_active = vcd_time;
                }
                vcd_time = t;
                last_vcd_time_active = false;

                for pos in std::mem::take(&mut delayed_bit_changes) {
                    state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
                }
            }
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                // Binary VCD vector values may omit leading zeroes. Restore
                // the declared width before mapping positions to bus pins.
                let width = vcd_widths.get(&id.0).copied().unwrap_or(bits.len());
                let padding = width.saturating_sub(bits.len());
                for pos in 0..width {
                    let b = if pos < padding {
                        b'0'
                    } else {
                        bits[pos - padding]
                    };
                    if let Some(&pin) = vcd2inp.get(&(id.0, pos)) {
                        let aigpin = aig.pin2aigpin_iv[pin];
                        assert_eq!(aigpin & 1, 0);
                        let aigpin = aigpin >> 1;
                        let pos = match input_map.get(&aigpin).copied() {
                            Some(pos) => pos,
                            None => {
                                panic!("input pin {:?} (netlist id {}, aigpin {}) not found in output map.", netlistdb.pinnames[pin].dbg_fmt_pin(), pin, aigpin);
                            }
                        };
                        let old_value = state[(pos >> 5) as usize] >> (pos & 31) & 1;
                        if old_value
                            == match b {
                                b'1' => 1,
                                _ => 0,
                            }
                        {
                            continue;
                        }
                        if let Some((pe, ne)) = aig.clock_pin2aigpins.get(&pin).copied() {
                            if pe != usize::MAX && old_value == 0 {
                                last_vcd_time_active = true;
                                let p = *input_map.get(&pe).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                            if ne != usize::MAX && old_value == 1 {
                                last_vcd_time_active = true;
                                let p = *input_map.get(&ne).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                        }
                        delayed_bit_changes.insert(pos);
                    }
                }
            }
        }
    }
    input_states.extend(state.iter().copied());
    clilog::info!("total number of cycles: {}", offsets_timestamps.len());
    let mut input_states_uvec: UVec<_> = input_states.clone().into();
    let device = Device::CUDA(0);
    input_states_uvec.as_mut_uptr(device);
    device.synchronize();
    let timer_sim = clilog::stimer!("simulation");
    #[cfg(feature = "v2")]
    if let Some((resolved, flattened)) = &v2_context {
        let gather_pairs: UVec<u32> = flattened.staged_word_slots.clone().into();
        let dsp_words: UVec<u32> = resolved.state.dsp_p_word.clone().into();
        let srl_words: UVec<u32> = resolved.state.srlc_storage_word.clone().into();
        let mut input_masks_host = vec![0u32; state_size];
        for (pin, driver) in aig.drivers.iter().enumerate() {
            if matches!(
                driver,
                DriverType::InputPort(_) | DriverType::InputClockFlag(..)
            ) {
                if let Some(&(word, bit)) = resolved.state.prev.get(&pin) {
                    input_masks_host[word as usize] |= 1u32 << bit;
                }
            }
        }
        let input_masks: UVec<u32> = input_masks_host.into();
        let queue_len = |kind| {
            resolved
                .queues
                .iter()
                .find(|q| q.kind == kind)
                .map_or(0, |q| q.cells.len())
        };
        uccv2::simulate_v2_cycles(
            &flattened.program,
            flattened.program.len(),
            &gather_pairs,
            flattened.staged_word_slots.len() / 2,
            &dsp_words,
            queue_len(MacroKind::Dsp48e2),
            &srl_words,
            queue_len(MacroKind::Srlc32e),
            &input_masks,
            &mut input_states_uvec,
            state_size,
            offsets_timestamps.len(),
            resolved.aig_operations.len(),
            queue_len(MacroKind::Carry4),
            flattened.shared_words_per_block.max(1) as usize,
            flattened.current_stage_words.max(1) as usize,
            flattened.num_partitions.max(1) as usize,
            aig.srams.len(),
            flattened.sram_storage_words as usize,
            device,
        );
    } else {
        let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
        ucci::simulate_v1_noninteractive_simple_scan(
            args.num_blocks,
            script.num_major_stages,
            &script.blocks_start,
            &script.blocks_data,
            &mut sram_storage,
            offsets_timestamps.len(),
            script.reg_io_state_size as usize,
            &mut input_states_uvec,
            device,
        );
    }
    #[cfg(not(feature = "v2"))]
    {
        let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
        ucci::simulate_v1_noninteractive_simple_scan(
            args.num_blocks,
            script.num_major_stages,
            &script.blocks_start,
            &script.blocks_data,
            &mut sram_storage,
            offsets_timestamps.len(),
            script.reg_io_state_size as usize,
            &mut input_states_uvec,
            device,
        );
    }
    device.synchronize();
    clilog::finish!(timer_sim);

    // sanity check.
    #[cfg(feature = "v2")]
    if args.check_with_cpu && use_v2 {
        let (resolved, _) = v2_context.as_ref().expect("V2 context");
        let mut input_masks = vec![0u32; state_size];
        for (pin, driver) in aig.drivers.iter().enumerate() {
            if matches!(
                driver,
                DriverType::InputPort(_) | DriverType::InputClockFlag(..)
            ) {
                if let Some(&(word, bit)) = resolved.state.prev.get(&pin) {
                    input_masks[word as usize] |= 1u32 << bit;
                }
            }
        }
        let mut previous = input_states[..state_size].to_vec();
        let mut sram_storage = vec![0u32; resolved.sram_storage_words as usize];
        for cycle in 0..offsets_timestamps.len() {
            let mut expected = gem::format_v2_cpu::interpret_cycle_with_sram(
                resolved,
                &previous,
                false,
                &mut sram_storage,
            )
            .unwrap_or_else(|e| panic!("V2 CPU oracle failed at cycle {cycle}: {e}"))
            .next_state;
            let external_next = &input_states[(cycle + 1) * state_size..(cycle + 2) * state_size];
            for word in 0..state_size {
                expected[word] = (expected[word] & !input_masks[word])
                    | (external_next[word] & input_masks[word]);
            }
            let gpu = &input_states_uvec[(cycle + 1) * state_size..(cycle + 2) * state_size];
            assert_eq!(&expected, gpu, "V2 CPU/CUDA mismatch at cycle {cycle}");
            previous = expected;
        }
        clilog::info!("V2 CPU sanity test passed!");
    }
    if args.check_with_cpu && !use_v2 {
        let mut sram_storage_sanity = vec![0; script.sram_storage_size as usize * AIGPDK_SRAM_SIZE];
        let mut input_states_sanity = input_states.clone();
        clilog::info!("running sanity test");
        for i in 0..offsets_timestamps.len() {
            let mut output_state = vec![0; script.reg_io_state_size as usize];
            output_state.copy_from_slice(
                &input_states_sanity[((i + 1) * script.reg_io_state_size as usize)
                    ..((i + 2) * script.reg_io_state_size as usize)],
            );
            for stage_i in 0..script.num_major_stages {
                for blk_i in 0..script.num_blocks {
                    simulate_block_v1(
                        &script.blocks_data[script.blocks_start[stage_i * script.num_blocks + blk_i]
                            ..script.blocks_start[stage_i * script.num_blocks + blk_i + 1]],
                        &input_states_sanity[(i * script.reg_io_state_size as usize)
                            ..((i + 1) * script.reg_io_state_size as usize)],
                        &mut output_state,
                        &mut sram_storage_sanity,
                        false,
                    );
                }
            }
            input_states_sanity[((i + 1) * script.reg_io_state_size as usize)
                ..((i + 2) * script.reg_io_state_size as usize)]
                .copy_from_slice(&output_state);
            if output_state
                != input_states_uvec[((i + 1) * script.reg_io_state_size as usize)
                    ..((i + 2) * script.reg_io_state_size as usize)]
            {
                println!(
                    "sanity check fail at cycle {i}.\ncpu good: {:?}\ngpu bad: {:?}",
                    output_state,
                    &input_states_uvec[((i + 1) * script.reg_io_state_size as usize)
                        ..((i + 2) * script.reg_io_state_size as usize)]
                );
                panic!()
            }
        }
        clilog::info!("sanity test passed!");
    }

    // output...
    clilog::info!("write out vcd");
    let mut last_val = vec![2; out2vcd.len()];
    for &(offset, timestamp) in &offsets_timestamps {
        if timestamp == u64::MAX {
            continue;
        }
        writer.timestamp(timestamp).unwrap();
        for (i, &(output_aigpin, output_pos, vid)) in out2vcd.iter().enumerate() {
            use vcd_ng::Value;
            let value_new = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    output_aigpin as u32
                }
                output_pos @ _ => {
                    let value_new_output = input_states_uvec[offset + (output_pos >> 5) as usize]
                        >> (output_pos & 31)
                        & 1;
                    value_new_output
                }
            };
            if value_new == last_val[i] {
                continue;
            }
            last_val[i] = value_new;
            writer
                .change_scalar(
                    vid,
                    match value_new {
                        1 => Value::V1,
                        _ => Value::V0,
                    },
                )
                .unwrap();
        }
    }
}
