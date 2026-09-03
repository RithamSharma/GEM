# Scripts

Every script is run **from the repository root** (`bash scripts/<name>`), is
idempotent, writes working files under `build/` or `verify_logs/`, and skips
cleanly when a tool it needs is missing. Nothing here modifies tracked files.

## Start here

| script | what it does |
|---|---|
| **`install_deps.sh`** | fresh machine — installs Rust, Yosys 0.68 + Icarus + yosys-slang (OSS CAD Suite), Python/reportlab; checks and guides the CUDA Toolkit. `--check` just reports status. |
| **`judge_verify_all.sh`** | one command — runs every Part A/B/C/D correctness gate in order, prints `PASS` / `FAIL` / `SKIP`, full logs in `verify_logs/`. `--quick` skips the Yosys flows and Part D. |
| **`run_hidden.sh`** `<design.v> <top> <stim.csv>` | the judge hand-off pipeline: synth → schedule → stimulus→VCD → GPU simulate with `--engine auto`, every cycle gated against the CPU reference. Output: `build/hidden/output.vcd`. |
| `run_best.sh` | same pipeline as `run_hidden.sh`; name emphasises that `--engine auto` picks the faster correct engine. |

## Run a design, step by step (building blocks — the two above call these)

| script | what it does |
|---|---|
| `run_synth_zenith.sh` `<design> <top> <out.gv>` | Yosys synthesis **keeping** DSP48E2 / CARRY4 / SRLC32E native; fails if any is lost. |
| `run_synth_baseline.sh` `<design> <top> <out.gv>` | Yosys synthesis **shredding** the macros to gates (unmodified-GEM behaviour) — the Part D comparison point. |
| `sanitize_gv.py` `<gatelevel.gv>` | pulls cell parameters (`PREG`, `INIT`, …) into a `params.json` sidecar next to the netlist. |
| `stim_to_vcd.py` `--stim <csv> --out <vcd> …` | converts a numeric stimulus table (CSV/TSV, headered or positional) into a VCD GEM can read. `--help` documents every option. |

## Verification gates (Parts B / C)

| script | proves |
|---|---|
| `run_v2_carry_chain_test.sh` | a direct `CARRY4.CO[3] → CARRY4.CI` chain matches independent HDL for 1024 vectors, CPU **and** 2-block cooperative CUDA. |
| `run_v2_sram_test.sh` | synthesized synchronous SRAM (write / read / read-before-write) matches independent HDL. |
| `run_v2_300_simulation_test.sh` | 300 cycles of mixed DSP48E2 + CARRY4 + SRLC32E + AIG glue: HDL == CPU V2 == CUDA V2, word for word. |
| `run_auto_check.sh` | the `--engine auto` dispatcher: forces V2 on chained macros, picks V1 otherwise, every run CPU-gated. |
| `run_300_simulation_test.sh` | the classic (pre-`--v2`) 300-cycle path — kept for regression. |
| `verify_submission.sh` | fast contract check: macros survive synthesis, parameter sidecar, ABI stability, unit tests. |
| `profile_v2_ncu.sh` | Nsight Compute profile of the integrated V2 kernel (needs `sudo` for GPU perf counters). |

## Benchmarks (Part D)

| script | what it does |
|---|---|
| `run_partd_benchmark.sh` `[design top cycles blocks]` | one design, same RTL preserved-vs-shredded, median of timed reps, speedup + AIG-node reduction → `benchmark-results/`. |
| `partd_sweep.sh` `["1 4 8 14"]` | the wide-parallel `hetero_farm` at several block counts plus the serial `bench_mac` for contrast → `benchmark-results/partd-sweep/`. |
| `partd_evidence.sh` | one command: the B/C gates + a Part D run, everything to `partd_evidence.log`. |

## Upstream GEM smoke test (not part of this submission)

| script | what it does |
|---|---|
| `upstream_smoke_map.sh` then `upstream_smoke_sim.sh` | synth + map + simulate the tiny reference circuit in `test_circuit/` with the classic V1 engine — a quick check that the base GEM toolchain works on your machine. |
