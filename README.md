# GEM — heterogeneous FPGA-macro simulation on the GPU

*Takneek PS Zenith — "The Big-GEM Theory"*

GEM is NVIDIA Research's GPU-accelerated RTL logic simulator. **This fork teaches
GEM to simulate three Xilinx FPGA macros natively on the GPU instead of shredding
them into thousands of Boolean gates**, and adds the scheduler, memory formatter,
CUDA engine, and verification needed to do it correctly and measurably faster.

| macro | what GEM now evaluates directly on the GPU ALU |
|---|---|
| **`DSP48E2`** | 27×18 multiply / 48-bit accumulate, registered `P` (`PREG`), OPMODE `9'h005` / `9'h025` / `9'h030`, ALUMODE / INMODE / CEP / RSTP honoured |
| **`CARRY4`** | the exact 4-bit carry chain — `CI`, `CYINIT`, `DI`, `S` → `O`, `CO` — in one combinational step |
| **`SRLC32E`** | 32-bit shift register, rising-edge shift on `CE`, asynchronous `Q` / `Q31` taps settle after the shift |

---

## Contents

1. [The 60-second version](#the-60-second-version)
2. [Why this matters](#why-this-matters)
3. [How it works](#how-it-works)
4. [Repository layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Build](#build)
7. [How to use it](#how-to-use-it)
8. [Parts A–E and where each is proven](#parts-ae-and-where-each-is-proven)
9. [Tests and benchmarks](#tests-and-benchmarks)
10. [Documentation map](#documentation-map)
11. [Troubleshooting](#troubleshooting)
12. [Lineage, attribution, license](#lineage-attribution-license)

---

## The 60-second version

**Fresh machine?** Install every dependency first:

```bat
setup.bat                  :: Windows — installs WSL2 + Ubuntu + the toolchain
```
```bash
bash scripts/install_deps.sh   # Linux / WSL2 — Rust, Yosys 0.68, Icarus, ... (guides CUDA)
```

**Judges — one command does everything** (build + unit tests + Part A/B/C/D
correctness gates + throughput benchmark + Nsight Compute profile):

```bat
compile.bat                :: Windows  (runs the Linux toolchain in WSL2 — see below)
```
```bash
./compile.sh               # Linux / WSL2
./compile.sh --quick       # skip the long throughput sweep
```

Everything lands in **`submission-results/`** — `compile.log`, `SUMMARY.txt`,
`verify_logs/`, `partd_*.json`, `*.ncu-rep`.

Benchmark one netlist on its own:

```bat
benchmark.bat  path\to\design.sv  top_module  [cycles]
```

Run a design + numeric stimulus and get a waveform:

```bash
./scripts/run_hidden.sh  examples/dsp_datapath.v  judge_dsp_datapath  examples/dsp_datapath_stim.csv
#  -> build/hidden/output.vcd  + a per-cycle "V2 CPU sanity test passed!" gate
```

Manual build, if you prefer:

```bash
git submodule update --init --recursive
cargo build --release --features v2
./scripts/judge_verify_all.sh
```

> **Rules compliance** (partial-grading, clock/init constraints, Yosys 0.68,
> custom-kernel requirement, Blackwell / RTX 5060): see
> [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).
> **Publishing the GitHub URL:** [`PUBLISHING.md`](PUBLISHING.md).

---

## Why this matters

An FPGA design is full of **hard macros** — a `DSP48E2` is a whole
multiply-accumulate unit, a `CARRY4` is a hand-tuned adder slice. A normal
gate-level simulator (GEM included) *shreds* each macro into an And-Inverter
Graph (AIG): a single `DSP48E2` becomes ~1,500–4,700 AND gates. That:

* **inflates the graph** — one of our test designs goes from **1,995 nodes to
  153,604** (77× bigger),
* **inflates logic depth**, so more sequential GPU work per clock cycle,
* and, in the classic "evaluate all AIG, then all macros" batched path,
  **produces wrong answers** whenever one macro feeds another *in the same
  cycle* — e.g. `CARRY4_A.CO[3] → CARRY4_B.CI`. The second macro reads last
  cycle's carry.

This fork keeps the macros whole and evaluates them in true dependency order.
The measured payoff (NVIDIA GTX 1650): **77× fewer graph nodes** and **up to
4.6× more simulated cycles/second** versus stock GEM on the same RTL — see
[`docs/PART_D_HOWTO.md`](docs/PART_D_HOWTO.md).

---

## How it works

The flow is the same "compile once, simulate many times" model as stock GEM.

```
  your RTL (.v/.sv)
        │
        ▼
  ┌─────────────────┐   Yosys, macros kept as black boxes
  │  synthesis      │   scripts/run_synth_zenith.sh  +  synth_zenith.ys
  └─────────────────┘   -> gate-level netlist that still contains DSP48E2/CARRY4/SRLC32E
        │
        ▼
  ┌─────────────────┐   src/schedule.rs
  │  schedule       │   build the unified same-cycle graph  G = (V, E_same ∪ E_next):
  │                 │   AIG regions and macros are peers; Kahn-levelize; reject
  │                 │   combinational loops; split each level into type-homogeneous
  │                 │   "waves" (a warp never mixes primitive types)
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐   src/format_v2*.rs   ("Part A" — the host-side formatter)
  │  format         │   resolve every macro operand bit to an explicit source
  │                 │   (constant / previous-state / this-cycle / cross-block),
  │                 │   transpose the tables so warp lane i reads word i, align to
  │                 │   64 bits, validate, hash, pack into ONE buffer, upload once
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐   csrc/kernel_v2.cu   ("Part B" — the CUDA engine)
  │  simulate       │   cooperative multi-block; per wave: fold the AIG glue by
  │  (GPU)          │   dependency depth, then drain the CARRY4 / SRLC32E / DSP48E2
  │                 │   queues on the ALU; grid.sync() only at real dependency
  │                 │   boundaries; commit registered state after the cycle settles
  └─────────────────┘
        │
        ▼
  output waveform (.vcd)   — gated every cycle against src/format_v2_cpu.rs,
                             a byte-accurate CPU re-implementation of the kernel
```

### Three engines, one artifact

`cuda_test` can run either engine, and picks for you by default:

| `--engine` | behaviour |
|---|---|
| `auto` *(default)* | **V2** whenever a macro output feeds a same-cycle consumer (the batched V1 path would be wrong there); **V1** otherwise. The choice and its reason are printed. |
| `v1` | the classic bit-parallel Boomerang engine (unmodified GEM). Refused if the design needs V2 for correctness. |
| `v2` | the heterogeneous wave engine. `--v2` is an alias. |

One synthesis and one `cut_map_interactive --v2-parts` produce a single
`.gemparts` file that carries **both** the V2 placement and the classic V1
partitions, so the dispatcher never needs a second synthesis. Predicate:
`HeteroSchedule::v1_batched_is_safe()` in `src/schedule.rs` (unit-tested).
Details: [`docs/OPTIMIZATION_ROADMAP.md`](docs/OPTIMIZATION_ROADMAP.md).

---

## Repository layout

```
GEM-heterogeneous-macro/
├── README.md                  ← you are here
├── setup.bat                  ← fresh Windows machine: WSL2 + Ubuntu + toolchain
├── compile.bat / compile.sh   ← ONE command: build + verify + benchmark
├── benchmark.bat              ← benchmark a single netlist (throughput + Nsight)
├── PUBLISHING.md              ← how to push the GitHub-URL deliverable
├── Cargo.toml  build.rs       ← Rust + CUDA build definition
│
├── src/                       ← host code (Rust)
│   ├── schedule.rs              unified same-cycle DAG, levelization, wave split,
│   │                            v1_batched_is_safe(), the Part B topological proof
│   ├── aig.rs  aigpdk.rs        netlist → AIG, macro descriptors preserved
│   ├── format_v2.rs             64-bit device ABI: sections, alignment, validation, hash
│   ├── format_v2_build.rs       operand→source resolution, coalesced serialization, SRAM
│   ├── format_v2_cpu.rs         byte-accurate CPU interpreter (the correctness oracle)
│   ├── format_v2_gpu.rs         device program transport
│   ├── hetero_parts.rs          the versioned .gemparts wrapper (V2 + embedded V1)
│   ├── macro_layout.rs          per-macro field tables + per-bit net provenance
│   ├── primitive_models.rs      exact fixed-width macro models ("Part C")
│   └── bin/
│       ├── cut_map_interactive.rs   "compile": synth netlist → .gemparts
│       └── cuda_test.rs             "simulate": .gemparts + stimulus → output VCD
│
├── csrc/                      ← device code (CUDA)
│   ├── kernel_v2.cu             cooperative multi-block wave evaluator ("Part B")
│   ├── hetero_primitives.cuh    macro models, bit-identical to primitive_models.rs
│   ├── format_v2_abi.h          shared host/device struct layout
│   ├── format_v2_decode.cuh     the 64-bit selector decoder
│   └── kernel_v1*.cu/.cuh       the untouched classic Boomerang kernel
│
├── aigpdk/                    ← cell libraries + the macro black-box stubs
│   └── zenith_macros.v         DSP48E2 / CARRY4 / SRLC32E port + parameter definitions
│
├── synth_zenith.ys            ← Yosys script: synthesize, keep macros native
├── synth_baseline.ys          ← Yosys script: synthesize, shred macros (Part D baseline)
│
├── scripts/                   ← all automation — see scripts/README.md
│   ├── judge_verify_all.sh       one-command Parts A/B/C/D verification
│   ├── run_hidden.sh             run a judge design + numeric stimulus (main entry)
│   ├── run_best.sh               same, via --engine auto
│   ├── run_synth_zenith.sh       synthesis wrapper (preserve)
│   ├── run_synth_baseline.sh     synthesis wrapper (shred)
│   ├── sanitize_gv.py            extract cell params → params.json sidecar
│   ├── stim_to_vcd.py            numeric stimulus table → VCD
│   ├── install_deps.sh          install Rust / Yosys 0.68 / Icarus / … on a fresh box
│   ├── run_v2_*.sh               the Part B/C correctness gates
│   └── run_partd_benchmark.sh, partd_sweep.sh, partd_evidence.sh   Part D
│
├── tests/hetero/              ← SystemVerilog fixtures + independent HDL oracles
│   ├── behavioral_zenith_macros.sv   behavioural macro models for the Icarus oracle
│   ├── carry_chain8.sv, hetero_farm.sv, bench_mac.sv, sram_top.sv, …
│   └── check_*.py                    VCD differential checkers
│
├── examples/                  ← worked design + stimulus (start here to try it)
│   ├── dsp_datapath.v
│   └── dsp_datapath_stim.csv
│
├── benchmarks/                ← Part D / coalescing measurement drivers (Python)
├── benchmark-results/         ← recorded measurements (GTX 1650) + Nsight report
├── test_circuit/              ← tiny upstream-GEM smoke-test circuit
├── eda-infra-rs/              ← vendored Rust deps (netlistdb, sverilogparse, ulib, …)
│
└── docs/                      ← everything else — see docs/README.md
    ├── COMPLIANCE.md                point-by-point rules & constraints response
    ├── SUBMISSION_STATUS.md          part-by-part grading tracker + evidence
    ├── PART_B_REPORT.md              engine + scheduling maths write-up
    ├── PART_D_HOWTO.md               benchmark: how to run + measured numbers
    ├── PART_E.md                     documentation deliverable index
    ├── V2_SCHEDULING.md              the scheduling equations
    ├── FORMATTER_V2_COALESCING.md    the 64-bit memory layout
    ├── OPTIMIZATION_ROADMAP.md       --engine auto + open kernel optimisations
    ├── VERIFICATION_RECORD.md        what was run on a real GPU
    ├── *.pdf                         technical report, Part E, explainer
    └── archive/                      superseded / historical notes
```

Load-bearing paths (`src/`, `csrc/`, `aigpdk/`, `eda-infra-rs/`, the two
`synth_*.ys`, `tests/hetero/`, `scripts/`) are referenced by the build and by
each other — keep them where they are.

---

## Prerequisites

| need | why | notes |
|---|---|---|
| **Linux userland** | the whole toolchain is Linux-native | on Windows: **WSL2** (`wsl --install -d Ubuntu`); `compile.bat` drives it for you |
| **NVIDIA GPU + CUDA toolkit** (`nvcc`) | the V2 engine | compute capability ≥ 7.0. Verified on a GTX 1650 (CC 7.5); the target machine's **RTX 5060 is Blackwell (CC 12.0)** — `compile.sh` handles the arch automatically (native SASS if CUDA ≥ 12.8, otherwise forward-compatible PTX). CUDA on WSL2 is fully supported. |
| **Rust** (stable, via [rustup](https://rustup.rs)) | host code + build | edition 2021 |
| **Yosys 0.68** | synthesis (keeps/shreds the macros) | `read_verilog -sv` (IEEE 1800-2012); swap in `read_slang` with `GEM_SYNTH_READ` — see [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) |
| **Icarus Verilog** (`iverilog`, `vvp`) | the *independent* HDL oracles | only for the verification gates, not to run a design |
| **Python 3** | `sanitize_gv.py`, `stim_to_vcd.py`, checkers, benchmark drivers | no non-stdlib deps for the core flow; `reportlab` only to regenerate the PDFs |

The hypergraph partitioner (`mt-kahypar-sc`) is compiled and linked
**automatically** — no external binary. (It builds under GCC/Clang, not MSVC —
another reason the Windows path goes through WSL2.)

Every script degrades gracefully: `judge_verify_all.sh` marks a step `SKIP`
(not `FAIL`) when its tools are absent.

### Windows (the target machine) — one-time WSL2 setup

```powershell
wsl --install -d Ubuntu          # then reboot
```
```bash
# inside the Ubuntu shell:
sudo apt update && sudo apt install -y build-essential yosys iverilog python3 git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#  + install the CUDA toolkit "WSL-Ubuntu" package from developer.nvidia.com/cuda-downloads
```

Then `compile.bat` works from the Windows side. The full setup is also in the
header of `compile.bat`.

---

## Build

`compile.bat` / `compile.sh` does all of this and the verification too. The
manual equivalent:

```bash
git submodule update --init --recursive     # vendored deps in eda-infra-rs/

cargo build --release --features v2          # the heterogeneous engine + tools
cargo test  --features v2 --lib              # ~52 host unit tests (scheduler, formatter, models)
```

**CUDA architecture.** `build.rs` defaults to `compute_75` PTX, which the driver
JIT-compiles forward onto any newer GPU (Turing → Blackwell / RTX 50-series).
For native SASS on a specific GPU set `UCC_CUDA_GENCODE` / `UCC_CUDA_PTX`
(e.g. `UCC_CUDA_GENCODE=120 UCC_CUDA_PTX=120` for an RTX 5060, needs CUDA ≥ 12.8) —
`compile.sh` does this detection automatically.

Feature flags: `cuda` builds the classic V1 kernel; `v2` (implies `cuda`) adds
the heterogeneous engine. A plain `cargo build` (no features) builds only the
host library.

Binaries you get with `--features v2`:

| binary | role |
|---|---|
| `cut_map_interactive` | **compile** — gate-level netlist → `.gemparts` |
| `cuda_test` | **simulate** — `.gemparts` + stimulus VCD → output VCD |
| `formatter_gpu_test` | Part A self-test: formatter buffer round-trips through GPU memory |

If `nvcc` rejects your host compiler as too new, see
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

---

## How to use it

### A. Verify the whole submission — one command

```bash
./scripts/judge_verify_all.sh            # everything available
./scripts/judge_verify_all.sh --quick    # host tests + GPU functional only
```

It runs each Part A/B/C/D gate in order, prints `PASS` / `FAIL` / `SKIP`, and
writes full output to `verify_logs/stepNN_*.log`. Expected key lines:

```
test result: ok. 52 passed; 0 failed
PASS: all three macros survived synthesis (and a CARRY-only subset)
PASS: direct CARRY4-to-CARRY4 dependency matched HDL for 1024 vectors
PASS: V2 SRAM matched independent HDL for 3 sampled edges
GEM matched the independent HDL oracle for all 300 cycles
memcheck / racecheck / synccheck: CLEAN
```

> If you ever ran a script with `sudo`, first `sudo rm -rf build target` so the
> normal user can write again.

### B. Run your own design + numeric stimulus  ← the main use case

```bash
./scripts/run_hidden.sh  <design.v>  <top-module>  <stimulus>  [num_blocks]  [-- <extra stim args>]
```

The panel typically gives you a Verilog design and a table of "numbers for each
signal, per simulation step". That table is your `<stimulus>`. Examples:

```bash
# stimulus CSV has a header row naming the input ports:  clk,a,b,...
./scripts/run_hidden.sh design.v top stim.csv

# no header row -> declare the column layout and radix:
./scripts/run_hidden.sh design.v top stim.txt 14 -- --ports clk:1,a:27,b:18 --radix hex

# testbench scope other than the default tb/uut:
SCOPE=tb_top/uut ./scripts/run_hidden.sh design.v top stim.csv
```

The pipeline: `run_synth_zenith.sh` → `sanitize_gv.py` →
`cut_map_interactive --v2-parts` → `stim_to_vcd.py` →
`cuda_test --engine auto --check-with-cpu`.

**Outputs**

| path | what |
|---|---|
| `build/hidden/output.vcd` | the simulated output waveform |
| `build/hidden/gem.log` | the run log — engine choice, cycle count, correctness result |
| `build/hidden/{yosys,map}.log` | synthesis + mapping logs |

**What "correct" looks like** — the log ends with:

```
engine=auto: N same-cycle macro->consumer edge(s) -> V2 (required for correctness)
selected simulation engine: V2 (heterogeneous)
total number of cycles: <N>
V2 CPU sanity test passed!
```

`--check-with-cpu` re-runs every cycle on the byte-accurate CPU interpreter and
`assert`s the GPU matches word-for-word. `V2 CPU sanity test passed!` (or
`V1 ...`) means the run is trustworthy.

### C. Stimulus table format

`scripts/stim_to_vcd.py` accepts CSV / TSV / whitespace-separated tables, **one
row per simulated cycle**:

* **Headered** — first row names the design's input ports. Bit-slice names are
  fine: `a[26:0],b[17:0]`. Values are decimal, or hex with a `0x` prefix (or
  `--radix hex` for the whole table).
* **Headerless** — pass `--ports clk:1,a:27,b:18` (in column order) after a `--`.
* A `clk` column, if present, is used verbatim. If absent, a clock is
  synthesised (0 while inputs settle, then one rising edge per row).
* An existing `.vcd` is passed straight through.

`python3 scripts/stim_to_vcd.py --help` documents everything.

### D. Choose the engine yourself

```bash
# force the heterogeneous engine
GEM_PARAMS_FILE=build/hidden/params.json target/release/cuda_test \
    build/hidden/gatelevel.gv build/hidden/result.gemparts \
    build/hidden/stim.vcd out.vcd 8 --engine v2 --check-with-cpu \
    --input-vcd-scope tb/uut

# --engine v1  -> classic Boomerang (refused if the design has a same-cycle macro->macro edge)
# --engine auto -> the default; prints which it chose and why
```

`scripts/run_auto_check.sh` exercises all three engines across the bundled
fixtures, each CPU-gated.

### E. The classic (non-macro) GEM flow

For a design with no FPGA macros, the upstream flow still applies — see
[`docs/UPSTREAM_GEM_USAGE.md`](docs/UPSTREAM_GEM_USAGE.md) (synthesis kit,
`cut_map_interactive`, `cuda_test` without `--v2`). `scripts/upstream_smoke_map.sh`
+ `upstream_smoke_sim.sh` run the bundled `test_circuit/` end to end as a sanity
check.

---

## Parts A–E and where each is proven

Full tracker with evidence: [`docs/SUBMISSION_STATUS.md`](docs/SUBMISSION_STATUS.md).

| part | what it asks | where it lives | how it's checked |
|---|---|---|---|
| **A** — parser + coalesced formatter | keep the macros through synthesis; lay them out for the GPU | `synth_zenith.ys`, `aigpdk/zenith_macros.v`, `src/format_v2*.rs` | `formatter_gpu_test` (GPU round-trip), `docs/FORMATTER_V2_COALESCING.md`, Nsight: 0 % excess sectors |
| **B** — CUDA engine + scheduler | evaluate macros natively, in correct same-cycle dependency order, on the GPU | `src/schedule.rs`, `csrc/kernel_v2.cu` | `verify_topological_guarantee()` (machine-checked proof) + unit tests; `run_v2_carry_chain_test.sh`; 300-cycle differential; Compute Sanitizer clean; `run_auto_check.sh` |
| **C** — exact macro models | cycle-accurate DSP48E2 / CARRY4 / SRLC32E, incl. synchronous SRAM | `src/primitive_models.rs`, `csrc/hetero_primitives.cuh` (bit-identical) | exhaustive CARRY4 (1024 vectors), DSP/SRLC boundary tests, `run_v2_sram_test.sh`, independent Icarus HDL oracle |
| **D** — throughput | measure preserved vs shredded on identical RTL | `scripts/run_partd_benchmark.sh`, `partd_sweep.sh` | GTX 1650: **77× fewer nodes**, **4.62× / 1.73×** cycles/s at 1 / 4 blocks; `benchmark-results/partd_measured.md` |
| **E** — documentation | scheduling maths, memory-hierarchy diagrams, throughput analysis | `docs/*.pdf`, `docs/V2_SCHEDULING.md`, `docs/PART_E.md` | regenerate: `python docs/build_report.py`, `python docs/build_part_e.py` |

**All three macros are complete** — this is not a partial submission. Each macro
is nonetheless an independent scheduler queue type and kernel branch, so its
individual throughput contribution is measurable in isolation
(`./scripts/run_partd_benchmark.sh <one-macro-design> <top>`). The rules,
clocking / initialisation constraints, Yosys version, and custom-kernel
requirement are answered point-by-point in
[`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).

---

## Tests and benchmarks

```bash
cargo test --features v2 --lib          # host logic: scheduler, formatter, macro models (~52 tests)

./scripts/run_v2_carry_chain_test.sh    # Part B: CARRY4->CARRY4 chain vs HDL, CPU + 2-block CUDA
./scripts/run_v2_sram_test.sh           # Part C: synchronous SRAM vs HDL
./scripts/run_v2_300_simulation_test.sh # Part B/C: 300-cycle mixed DSP/CARRY4/SRLC32E/AIG differential
./scripts/run_auto_check.sh             # Part B: the --engine auto dispatcher, CPU-gated

./scripts/partd_sweep.sh                # Part D: hetero_farm @ 1/4/8 blocks + bench_mac
sudo ./scripts/profile_v2_ncu.sh        # Part D: Nsight Compute profile (needs perf-counter permission)
```

Fixtures (`tests/hetero/`): `carry_chain8.sv` (deep macro chain), `hetero_farm.sv`
(wide independent macros), `bench_mac.sv` (serial DSP+CARRY), `sram_top.sv`
(synchronous RAM), `preservation_top.sv` (one of each macro),
`behavioral_zenith_macros.sv` (the independent oracle models).

---

## Documentation map

| start with | then |
|---|---|
| **this README** | [`docs/README.md`](docs/README.md) — the full index |
| [`docs/SUBMISSION_STATUS.md`](docs/SUBMISSION_STATUS.md) — grading tracker | [`docs/PART_B_REPORT.md`](docs/PART_B_REPORT.md), [`docs/PART_D_HOWTO.md`](docs/PART_D_HOWTO.md), [`docs/PART_E.md`](docs/PART_E.md) |
| `docs/GEM_Heterogeneous_Macro_Technical_Report.pdf` | `docs/GEM_PartE_Documentation.pdf`, `docs/GEM_Heterogeneous_Macro_Explainer.pdf` |
| design internals | [`docs/V2_SCHEDULING.md`](docs/V2_SCHEDULING.md), [`docs/FORMATTER_V2_COALESCING.md`](docs/FORMATTER_V2_COALESCING.md), [`docs/OPTIMIZATION_ROADMAP.md`](docs/OPTIMIZATION_ROADMAP.md) |
| running things | [`scripts/README.md`](scripts/README.md), [`examples/README.md`](examples/README.md) |

---

## Troubleshooting

| symptom | fix |
|---|---|
| `nvcc` error: *unsupported GNU version* | your host GCC is newer than the CUDA toolkit allows — see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| `Module 'DSP48E2' … does not have a parameter named 'INMODE'` | your design uses a non-GEM macro signature — `OPMODE`/`ALUMODE`/`INMODE` are **ports**, only `PREG`/`USE_MULT`/`USE_SIMD` are parameters. See `aigpdk/zenith_macros.v` and [`examples/dsp_datapath.v`](examples/dsp_datapath.v) |
| `Module '<top>' not found` in Yosys | wrong top-module name, or a stale design file — `grep '^module' your_design.v` |
| `cannot schedule heterogeneous DAG: CombinationalCycle {...}` | a macro's data input is a combinational function of its own output (e.g. an LFSR feeding `SRLC32E.D` from `Q`). GEM can't prove such a loop is safe — register the feedback through a flip-flop. |
| cooperative launch fails / low occupancy | pass a smaller `num_blocks` (the scripts cap at 14 on a 14-SM GPU); one block always works |
| a `judge_verify_all.sh` step says `SKIP` | that step's tool (Yosys / Icarus / `ncu` / GPU) isn't installed — not a failure |
| permission errors after a `sudo` run | `sudo rm -rf build target` then re-run as your normal user |

---

## Lineage, attribution, license

* **Base:** NVIDIA Research GEM — see [`BASE_REVISION.txt`](BASE_REVISION.txt).
  The committed V1 Boomerang kernel is **untouched**; the heterogeneous path is
  the opt-in `--features v2` build. Baseline defects fixed along the way are
  recorded in [`docs/archive/BASELINE_AUDIT.md`](docs/archive/BASELINE_AUDIT.md).
* **New code** in this deliverable was written with Claude Code under human
  direction and is delivered for review.
* **License:** Apache-2.0 (`LICENSE`, `LICENSES.txt`).

```bibtex
@inproceedings{gem,
  author    = {Guo, Zizheng and Zhang, Yanqing and Wang, Runsheng and Lin, Yibo and Ren, Haoxing},
  booktitle = {Proceedings of the 62nd Annual Design Automation Conference 2025},
  title     = {{GEM}: {GPU}-Accelerated Emulator-Inspired {RTL} Simulation},
  year      = {2025}
}
```
