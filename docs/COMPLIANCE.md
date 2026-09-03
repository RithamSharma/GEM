# Rules & constraints compliance

Point-by-point response to the PS Zenith rules, and where each is handled in the code.

---

## Judging constraints

### (1) Partial submissions are graded proportionally on throughput gains

**All three macros are implemented and verified** — this is not a partial
submission. But the architecture *is* modular, so a partial grade is well-defined:

| macro | model | native CUDA | verified |
|---|---|---|---|
| `DSP48E2` | `src/primitive_models.rs::dsp48e2_next` | `csrc/hetero_primitives.cuh` | boundary + control tests, 300-cycle HDL differential |
| `CARRY4` | `src/primitive_models.rs::carry4` | `csrc/hetero_primitives.cuh` | **exhaustive** (all 1024 inputs), direct `CO[3]→CI` chain vs HDL (1024 vectors) |
| `SRLC32E` | `src/primitive_models.rs::srlc32e_step` | `csrc/hetero_primitives.cuh` | edge/tap timing tests, 300-cycle HDL differential |

Each macro is an independent queue type in the scheduler
(`src/schedule.rs`, `MacroKind`) and an independent branch in the kernel
(`csrc/kernel_v2.cu`). Throughput contribution per macro is measured directly by
`scripts/run_partd_benchmark.sh` on a design containing only that macro.

### (2) Single global clock domain, all synchronous macros on the same rising edge

Enforced by construction. `src/aig.rs` infers exactly one clock net for the
whole netlist (`inferred clock port <clk> (posedge)` in the `cut_map_interactive`
log). Every sequential endpoint — DSP48E2 `PREG`, SRLC32E shift storage, plain
DFFs, synchronous SRAM — is committed by `csrc/kernel_v2.cu` **after** the
combinational wave schedule for that cycle has fully settled
(`src/schedule.rs`: `E_next` edges do not participate in the topological order;
the commit is one barrier at end-of-cycle). There is no per-macro clock, no
multi-edge handling, and no negedge path in the heterogeneous engine.

### (3) All internal macro registers initialise to zero; INIT parsing not required

Honoured unconditionally. `src/aig.rs` forces `srlc.init = 0` and the DSP `PREG`
starts at zero (`src/format_v2_build.rs` builds `initial_state` as all-zero for
macro state words). The netlist `INIT` string is **ignored** by default so our
state matches the panel's golden model exactly. (Setting
`GEM_HONOR_SRLC_INIT=1` re-enables INIT parsing; it is used only by the
project's own standalone fixtures and is never needed for grading.)

### (4) Yosys 0.68, IEEE 1800-2012, Slang frontend permitted, JSON-netlist compatible

* **Version** — the synthesis wrappers call whatever `yosys` is first on `PATH`;
  they are written against and tested with **Yosys 0.68**. `scripts/install_deps.sh`
  installs it (via the OSS CAD Suite bundle, which also brings `yosys-slang` and
  Icarus Verilog); `compile.sh` prints `yosys --version` at the top of every run
  and `install_deps.sh --check` warns if the version is below 0.68.
* **SystemVerilog** — `synth_zenith.ys` / `synth_baseline.ys` read the design
  with Yosys 0.68's built-in `read_verilog -sv` frontend, which parses
  IEEE 1800-2012. The read command is a single templated line
  (`@@GEM_READ@@ @@GEM_DESIGN@@`); set `GEM_SYNTH_READ="read_slang"` (or any
  other reader) to switch frontends without editing files, e.g. if a hidden
  benchmark needs the yosys-slang plugin:

  ```bash
  GEM_SYNTH_READ="read_slang" ./scripts/run_hidden.sh design.sv top stim.csv
  ```
* **JSON netlist** — GEM's Rust front-end (`NetlistDB::from_sverilog_file`)
  consumes **Verilog** (`write_verilog -noattr`), which is stable across Yosys
  releases. We do not depend on the JSON schema; if a JSON netlist is supplied
  instead, convert it once with `yosys -p 'read_json in.json; write_verilog out.v'`.

### (5) Standard CUDA libraries allowed; the core loop + macro eval must be a custom kernel

Compliant. `csrc/kernel_v2.cu` is **entirely hand-written**:

* the cooperative multi-block wave scheduler loop (`grid.sync()` between
  dependency levels, `__shfl_sync` wave-descriptor broadcast, `__syncwarp`
  partial tails),
* the 64-bit selector decode (`csrc/format_v2_decode.cuh`),
* every macro evaluator (`csrc/hetero_primitives.cuh` — DSP48E2 multiply /
  accumulate, CARRY4 chain, SRLC32E shift + taps),
* the AIG-region dependency-depth fold.

No Thrust, CUB, or library kernel is used for scheduling or evaluation. The only
external runtime dependency is `cudart` (CUDA runtime — malloc / memcpy /
cooperative launch). The classic V1 kernel (`csrc/kernel_v1.cu`) is the
unmodified NVIDIA GEM Boomerang kernel and is likewise custom.

### (6) Final evaluation on hidden SystemVerilog netlists, on the panel's machine

The whole flow is netlist-agnostic:

```bash
./scripts/run_hidden.sh   <hidden_design.sv> <top> <panel_stimulus.csv|vcd>
./scripts/run_partd_benchmark.sh <hidden_design.sv> <top> <cycles>   # + GEM_PARTD_STIM for a raw table
```

`scripts/run_hidden.sh` gates every simulated cycle against the byte-accurate
CPU model, so a correctness regression on an unseen design is caught, not
hidden. `compile.bat` / `compile.sh` build, verify, and benchmark in one
command on the panel's machine.

---

## Target machine (Intel Core 7 240H · 16 GB DDR5 · RTX 5060 8 GB)

* **RTX 5060 is Blackwell (compute capability 12.0 / `sm_120`).** `compile.sh`
  detects the installed compute capability and, if this CUDA toolkit knows the
  arch (CUDA ≥ 12.8), builds native `sm_120` SASS. Otherwise it embeds
  `compute_75` **PTX**, which the driver JIT-compiles forward onto Blackwell —
  correct on any GPU from Turing onward, with no toolkit-version assumption.
  Override with `UCC_CUDA_GENCODE` / `UCC_CUDA_PTX` if needed.
* **Cooperative launch** — the kernel now *clamps* the requested block count to
  `SM_count × maxActiveBlocksPerSM` (`csrc/kernel_v2.cu`), so an over-large
  `num_blocks` degrades gracefully instead of failing the launch. Scripts
  default `num_blocks` to the GPU's SM count (~30 on RTX 5060).
* **8 GB VRAM** — the whole-trace input-state buffer scales with
  `cycles × state_words`. For a very large hidden netlist run fewer cycles per
  invocation (`--max-cycles N`, or the `cycles` argument to the benchmark).
* **Windows host** — `setup.bat` installs WSL2 + Ubuntu + the whole toolchain
  (`scripts/install_deps.sh` inside WSL); `compile.bat` then runs the build in
  WSL2 (NVIDIA's supported path for CUDA on Windows). On native Linux, just
  `bash scripts/install_deps.sh` then `./compile.sh`.

---

## Deliverables checklist

| # | asked for | in this repo |
|---|---|---|
| 1 | GitHub URL | push this tree; see [`../PUBLISHING.md`](../PUBLISHING.md) |
| 2 | Technical report (PDF) | `GEM_Heterogeneous_Macro_Technical_Report.pdf`, `GEM_PartE_Documentation.pdf` |
| 3 | Testbenches & golden models (CPU reference + verification scripts you authored) | `src/primitive_models.rs` + `src/format_v2_cpu.rs` (byte-accurate CPU oracle), `tests/hetero/behavioral_zenith_macros.sv` (independent HDL oracle), `tests/hetero/*.sv` fixtures, `tests/hetero/check_*.py` (VCD differential checkers), `scripts/run_v2_*.sh` (the gates) |
| 4 | Benchmark automation (Nsight Compute performance logs) | `scripts/profile_v2_ncu.sh` (writes `benchmark-results/part_b_v2_integrated.ncu-rep` + `.txt`), `scripts/run_partd_benchmark.sh`, `scripts/partd_sweep.sh`; `benchmarks/*.py` |
| 5 | Complete, properly structured source code | whole tree — layout in [`../README.md`](../README.md); `compile.bat` / `compile.sh` build it |
