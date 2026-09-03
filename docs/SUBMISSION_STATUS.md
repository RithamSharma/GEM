# GEM heterogeneous-macro submission — part-by-part status

> The repository entry point is **[`../README.md`](../README.md)**. This file is
> the detailed grading tracker: the point-by-point status of Parts A–E and the
> exact evidence for each. Start with the README; come here for the audit trail.

---

## 1. What this is

A fork of NVIDIA's GEM RTL simulator that **natively evaluates three preserved
Xilinx FPGA macros on the GPU** instead of shredding them to And-Inverter-Graph
(AIG) gates:

| macro | what GEM now does |
|---|---|
| `DSP48E2` | 27x18 multiply / 48-bit accumulate on the GPU ALU (`PREG` registered `P`), OPMODE `9'h030`/`9'h005`/`9'h025`, ALUMODE/INMODE/CEP/RSTP honoured |
| `CARRY4`  | exact 4-bit carry chain (`CI`, `CYINIT`, `DI`, `S` -> `O`, `CO`) in one combinational evaluation |
| `SRLC32E` | 32-bit shift register, rising-edge shift on `CE`, asynchronous `Q`/`Q31` taps settle post-shift |

The core of the work is a **unified same-cycle dependency schedule** in which
AIG regions and preserved macros are peers, so a direct chain such as
`CARRY4_A.CO[3] -> CARRY4_B.CI` is evaluated in the right order **with no
intermediate boolean node**, and a **64-bit-aligned, coalesced CUDA formatter**
that maps that schedule into device memory. The GPU executor is a cooperative
multi-block, type-homogeneous wave evaluator; a byte-accurate CPU interpreter is
the per-cycle correctness oracle.

## 2. Lineage

Built additively on upstream GEM — see [`../BASE_REVISION.txt`](../BASE_REVISION.txt).
The committed V1 Boomerang kernel is **untouched**; the heterogeneous path is an
opt-in `--features v2` build. This tree merges two lines of work:

- the **integrated V2 engine** (`cuda_test --v2` end to end, cooperative
  multi-block, synchronous SRAM, `.gemparts` V2 placement) — verified on a real
  NVIDIA GTX 1650, see [`VERIFICATION_RECORD.md`](VERIFICATION_RECORD.md);
- the **submission package** — the machine-checked topological-guarantee proof
  in the scheduler, the preserved-vs-shredded Part D benchmark, the one-command
  judge verifier, the judge numeric-stimulus pipeline, and the technical
  report / Part E PDFs.

## 3. Part-by-part status

| Part | / pts | status | evidence |
|---|--:|---|---|
| **A** parser + coalesced formatter | 15 | **done.** Yosys keeps all three macros native (`synth_zenith.ys` + wrapper assertion). Formatter maps the schedule into one immutable 64-bit `u64` program: field/bit-major selector sections, 8/16-byte aligned, validated, content-hashed; adjacent warp lanes read adjacent words. GPU round-trip verified. `cut_map_interactive --v2-parts` emits the V2 artifact in the real synth flow. | `formatter_gpu_test` (PASS on GTX 1650), `docs/FORMATTER_V2_COALESCING.md`, `benchmark-results/part_b_v2_summary.md` (0% excess sectors, 99.6% branch efficiency) |
| **B** CUDA engine + scheduler | 35 | **done.** `build_schedule` builds `G = (V, E_same ∪ E_next)`, Kahn-levelizes, rejects combinational cycles, splits each level into type-homogeneous waves. **`verify_topological_guarantee()` + `macro_dependencies()` machine-check that every macro->macro edge is a direct net (no boolean node) crossing a strict wave boundary** — printed by `cut_map_interactive --v2-parts`, unit-tested. GPU executor: cooperative multi-block (verified 2/4/8/14 blocks), `grid.sync()` cross-block publish, `__shfl_sync` wave broadcast, `__syncwarp` tails, AIG glue folded per depth. **`cuda_test --engine auto`** picks V1 or V2 per design: V2 is forced whenever a macro output feeds a same-cycle consumer (`v1_batched_is_safe()`, unit-tested); otherwise V1 (batched on the preserved netlist) is used, avoiding V2's per-cycle wave barriers. | `cargo test --features v2 --lib`, `run_v2_carry_chain_test.sh` (1024 vectors, CPU + 2-block CUDA), `run_v2_300_simulation_test.sh` (300/300), `run_auto_check.sh` (dispatcher, CPU-gated), Compute Sanitizer 3/3 clean |
| **C** exact macro models | 20 | **done.** Fixed-width two's-complement models (`src/primitive_models.rs`), exhaustive CARRY4 (1024 vectors), DSP boundary/control tests, SRLC edge/tap timing; the CUDA models in `csrc/hetero_primitives.cuh` are bit-identical; independent Icarus HDL oracle differential over 300 mixed cycles; synchronous SRAM (read-before-write). | `run_v2_sram_test.sh`, `run_v2_300_simulation_test.sh`, `tests/hetero/behavioral_zenith_macros.sv` |
| **D** throughput | 20 | **measured on GTX 1650**, two independent sweeps within 2%, correctness-gated. Preserving 96 macros vs shredding them: **77× fewer AIG nodes** (1,995 pins / 0 AND gates vs 153,604 / 150,905), and **4.62× / 1.73× throughput** over unmodified GEM at 1 / 4 blocks, reaching parity at 8 (the 150k-gate shredded design fills the 14-SM GPU). Deep-serial-chain counter-case (`bench_mac`, 0.32×) reported honestly. V2's flat cost is bit-serial operand assembly (~141 selector loads/DSP lane/cycle) — the identified optimisation, not a correctness gap. Optional: privileged integrated Nsight. | `benchmark-results/partd_measured.md`, `benchmark-results/partd-sweep/*.json`, `docs/PART_D_HOWTO.md`, Part E PDF §3.8 |
| **E** documentation | 10 | **done.** Scheduling equations, memory-hierarchy mapping, architecture diagrams, analysis. | `docs/GEM_Heterogeneous_Macro_Technical_Report.pdf`, `docs/GEM_PartE_Documentation.pdf`, `docs/GEM_Heterogeneous_Macro_Explainer.pdf`, `docs/V2_SCHEDULING.md` |

## 4. The five PS deliverables

| # | deliverable | in this tree |
|---|---|---|
| 1 | Technical report (PDF) | `docs/GEM_Heterogeneous_Macro_Technical_Report.pdf` (+ `docs/GEM_PartE_Documentation.pdf`) |
| 2 | GitHub link + change log | push this tree; section 7 below + `docs/PART_B_REPORT.md` §5 for the heterogeneous changes, `docs/archive/BASELINE_AUDIT.md` for the base repairs |
| 3 | Testbenches & golden models | `src/primitive_models.rs`, `csrc/hetero_primitives.cuh`, `src/format_v2_cpu.rs` (byte-accurate interpreter), `tests/hetero/*.sv` + `tests/hetero/behavioral_zenith_macros.sv` (Icarus oracle) + the three VCD checkers |
| 4 | Benchmark automation | `scripts/run_partd_benchmark.sh`, `benchmarks/run_v2_simulation_benchmark.py`, `benchmarks/run_v2_gpu_benchmark.py`, `benchmarks/formatter_coalescing.py` |
| 5 | Source code | whole tree; **`scripts/judge_verify_all.sh` = one-command verification** |

## 5. Verify it (one command)

```bat
compile.bat            :: Windows — builds + verifies + benchmarks via WSL2
```
```bash
./compile.sh           # Linux / WSL2 — same
./scripts/judge_verify_all.sh   # just the A/B/C/D gates: PASS / FAIL / SKIP + verify_logs/
```

Rules / constraints compliance (partial grading, single-clock, zero-init,
Yosys 0.68, custom-kernel, RTX 5060 / Blackwell): [`COMPLIANCE.md`](COMPLIANCE.md).

Every step is independent, skips cleanly if a tool is missing, and writes its
full output to `verify_logs/stepNN_*.log`. Expected key lines:

```
test result: ok. 52 passed; 0 failed
PASS: all three macros survived synthesis (and a CARRY-only subset)
PASS: direct CARRY4-to-CARRY4 dependency matched HDL for 1024 vectors
PASS: V2 SRAM matched independent HDL for 3 sampled edges
GEM matched the independent HDL oracle for all 300 cycles
memcheck / racecheck / synccheck: CLEAN
```

If you ever ran a script with `sudo`, first `sudo rm -rf build target` so the
non-root run can write.

`./scripts/run_auto_check.sh` separately exercises the `--engine auto` dispatcher
(V1 vs V2 per design, every run CPU-gated).

## 6. Run a judge-supplied design + numeric stimulus

The panel gives a Verilog design and a "database file with numbers for each
signal per step". GEM reads stimulus as VCD; `scripts/stim_to_vcd.py` bridges
the two and `scripts/run_hidden.sh` runs the whole heterogeneous pipeline:

```bash
# stimulus DB has a header row (clk,a,b,... or a[26:0],b[17:0],...):
./scripts/run_hidden.sh their_design.v their_top their_stim.csv

# no header -> give the column layout and radix:
./scripts/run_hidden.sh their_design.v their_top stim.txt 14 -- --ports clk:1,a:27,b:18 --radix hex

# testbench scope other than tb/uut:
SCOPE=tb_their/uut ./scripts/run_hidden.sh their_design.v their_top stim.csv
```

Output: `build/hidden/output.vcd` (waveform) plus the per-cycle CPU correctness
gate. The pipeline is: `run_synth_zenith.sh` -> `sanitize_gv.py` ->
`cut_map_interactive --v2-parts` -> `stim_to_vcd.py` -> `cuda_test --engine auto
--check-with-cpu` (auto picks V1 or V2; the log line `selected simulation engine`
says which and why). `scripts/run_best.sh` is the same flow.

## 7. What changed, by file (additive on the baseline)

| file | change |
|---|---|
| `src/schedule.rs` | unified same-cycle DAG, Kahn levelization, typed waves, cycle rejection; **`MacroEdgeProof` + `macro_dependencies()` + `verify_topological_guarantee()`** (PS Part B proof) |
| `src/aig.rs`, `src/aigpdk.rs` | preserve macro descriptors, pins, outputs, parameters, endpoints |
| `src/hetero_parts.rs` | versioned `.gemparts` V2 wrapper (legacy Boomerang payload + per-wave/per-partition heterogeneous queues); V1 still reads the legacy payload |
| `src/format_v2*.rs` | 64-bit ABI + sections + validation + hashing; state layout, selector resolution, liveness allocation, coalesced serialization, cross-block remap, SRAM ops; independent CPU interpreter; device program transport |
| `src/macro_layout.rs` | logical macro field tables + per-bit net provenance |
| `csrc/format_v2_abi.h`, `csrc/format_v2_decode.cuh` | shared host/device ABI + selector decoder |
| `csrc/hetero_primitives.cuh` | fixed-width native CUDA primitive models (bit-identical to `src/primitive_models.rs`) |
| `csrc/kernel_v2.cu` | cooperative multi-block wave evaluator + state / SRAM commit |
| `src/bin/cut_map_interactive.rs` | `--v2-parts` / `--v2-num-partitions`: emit the V2 placement, print the macro-dependency proof |
| `src/bin/cuda_test.rs` | `--engine {auto,v1,v2}` dispatcher (default `auto`, correctness-driven); `--v2` alias; end-to-end V2 CLI, VCD flow, CPU gate, one `.gemparts` feeds both engines |
| `src/schedule.rs` | `+ v1_batched_is_safe()` — the dispatcher's correctness predicate (unit-tested) |
| `synth_zenith.ys` + wrapper | preserve macros, retain synchronous-RAM mapping |
| `synth_baseline.ys` + wrapper | **new:** the Part D comparison point (shred macros to gates) |
| `Cargo.toml`, `build.rs`, `src/lib.rs` | `+ [features] v2 = ["cuda"]`; separate `gemcu_v2` CUDA lib; module declarations |

## 8. Design notes and limitations (state these honestly)

- **AIG regions.** In the V2 engine, a region's AND gates are evaluated in the
  wave loop, ordered by internal dependency depth, parallel across the block,
  with a barrier only between depths. Each gate is a 2-input AND with the
  inversions folded into its two operand selectors, so a region computes exactly
  its Boolean function in a valid topological order — verified against
  independent HDL over 300 mixed cycles (including AIG glue feeding macro
  operands) and against the CPU reference every cycle. This is **not** GEM's V1
  bit-parallel 256-vector Boomerang kernel — that is a separate code path, is
  untouched, and remains the default `--features cuda` (non-`v2`) build. The V1
  fold packs 256 independent input vectors into thread lanes; a per-cycle
  simulation has one vector, so the dependency-depth fold is the right design.
- **Multi-block cooperative execution — implemented and verified** at 2, 4, 8
  and 14 blocks (`cudaLaunchCooperativeKernel`, `grid.sync()` between waves,
  occupancy check before launch, cross-block same-cycle values promoted to the
  global `CurrentStage` arena). The 14-block 300-cycle mixed differential and
  the 2-block 1024-vector carry chain are both in the standard gate set.
- **Multi-block *scaling*** on a small design is limited: V2's per-cycle cost
  (~16–17k cyc/s on a GTX 1650) is dominated by **bit-serial operand assembly**
  — each DSP48E2 lane issues ~141 sequential selector loads/cycle to rebuild its
  wide operands. That is a per-macro fixed cost, so extra blocks only spread the
  macros over more idle lanes. The loads are already coalesced (Nsight: 0%
  excess sectors). **`--engine auto` sidesteps this** by running V1 (batched on
  the preserved netlist, no wave barriers) whenever V2 is not required for
  correctness — so the combined engine is never slower than classic GEM and is
  V2-fast (up to 4.6× vs shredded GEM) exactly where V2 is needed. The remaining
  V2 optimisation (a cooperative *bulk* gather + per-partition ownership) is in
  [`OPTIMIZATION_ROADMAP.md`](OPTIMIZATION_ROADMAP.md). Neither changes correctness.
- **DSP48E2** accepts OPMODE `9'h030 / 9'h005 / 9'h025`; other encodings for the
  same intent would need parser decoding. Unsupported controls fail loudly
  rather than silently becoming a hold.
- **Front end** is Yosys' built-in SystemVerilog parser, not Slang — confirm the
  hidden designs parse.
- **Integrated Nsight report** needs GPU performance-counter permission
  (`sudo scripts/profile_v2_ncu.sh`); the formatter/macro-kernel profile in
  `benchmark-results/part_b_v2.ncu-rep` (0% excess sectors, 99.6% branch
  efficiency) stands until then.

## 9. Document map

Full index: [`docs/README.md`](README.md). Key entries:

| doc | purpose |
|---|---|
| [`../README.md`](../README.md) | **entry point** — overview, install, how to run any design |
| `SUBMISSION_STATUS.md` (this file) | part-by-part grading tracker + evidence |
| [`PART_B_REPORT.md`](PART_B_REPORT.md) | Part B design + scheduling maths + memory mapping |
| [`PART_D_HOWTO.md`](PART_D_HOWTO.md) | Part D benchmark: how to run, measured numbers |
| [`PART_E.md`](PART_E.md) + `GEM_PartE_Documentation.pdf` | Part E deliverable |
| [`VERIFICATION_RECORD.md`](VERIFICATION_RECORD.md) | GPU verification record (GTX 1650) |
| [`V2_SCHEDULING.md`](V2_SCHEDULING.md), [`FORMATTER_V2_COALESCING.md`](FORMATTER_V2_COALESCING.md) | scheduling equations, formatter layout |
| [`OPTIMIZATION_ROADMAP.md`](OPTIMIZATION_ROADMAP.md) | shipped `--engine auto` + the open V2 optimisations |
| `GEM_Heterogeneous_Macro_Technical_Report.pdf` | the full technical report |
| [`archive/`](archive/) | superseded / historical notes, kept for the audit trail |
