# Current `updated_gem_branch` audit and repair record

## Scope

The repository was cloned from the remote tip rather than copied from an older
ZIP. `BASE_REVISION.txt` records the exact source revision. The reconstructed
GEM PS contract used for this audit is:

- Part A: preserve `DSP48E2`, `CARRY4`, and `SRLC32E`; retain required
  controls/parameters; flatten to a safe GPU data layout.
- Part B: include AIG and macro producers/consumers in a dependency-aware
  schedule and use type-homogeneous GPU work to avoid stale reads/divergence.
- Part C: exact bounded primitive models with fixed widths, current/next state,
  and independent behavioral verification.

## Defects found at the latest remote tip

| Severity | Defect | Evidence in the remote tip | Repair in this tree |
|---|---|---|---|
| Critical | `synth_zenith.ys` used Tcl environment syntax in a plain Yosys script, so the documented command looked for a literal `$::env(GEM_DESIGN)` filename. | `synth_zenith.ys` | Added template tokens and `scripts/run_synth_zenith.sh`. |
| Critical | DSP CUDA used private OPMODE bits instead of DSP48E2 W/X/Y/Z values; ALUMODE/INMODE/CEP/RSTP were ignored. | `csrc/kernel_v1_impl.cuh`, `src/aig.rs` | Real supported encodings, fixed widths, pre-add control, enable/reset, and explicit rejection. |
| Critical | CPU `--check-with-cpu` did not understand macro layout or evaluate macros. | `src/bin/cuda_test.rs` | CPU executor now gathers/evaluates DSP, CARRY, and SRLC words using the independent Rust model. |
| High | SRLC `INIT` was parsed into a descriptor but never placed in simulator state. | `src/aig.rs`, `src/flatten.rs` | Added a flattened initial-state image and simulator initialization. |
| High | Synthesis required all three macro types even for a valid subset design. | `synth_zenith.ys` | Wrapper conditionally compares input/output counts; CARRY-only regression passes. |
| High | The sidecar was read only from `test_circuit/params.json`; malformed literals silently became zero. | `src/aig.rs`, `scripts/sanitize_gv.py` | Added `GEM_PARAMS_FILE`, binary/octal/hex/decimal parsing, and fail-fast validation. |
| High | CUDA 13 build defaults requested removed `sm_50` and C++14. | `eda-infra-rs/ucc/src/compile.rs` | C++17 and verified cc7.5 defaults; environment overrides remain available. |
| High | Benchmark script changed branches in one mutable worktree and ignored profiler/process failures. | `benchmark.sh`, `benchmarks/run_benchmarks.py` | No branch mutation; repeated statistics; CPU correctness gate; strict failures; JSON/Nsight artifacts. |
| High | Original GEM memory mapping was commented out in the Zenith flow. | `synth_zenith.ys` | Restored `memory -nomap` and `memory_libmap`. |
| Critical | Short VCD binary values were aligned at the MSB instead of zero-extended to the declaration width, corrupting changing buses. | `src/bin/flatten_test.rs`, `src/bin/cuda_test.rs` | Restore omitted leading zeroes before bus mapping; caught by the 300-cycle differential test. |
| High | SRLC taps reported pre-edge storage, disagreeing with HDL behavior after the shift. | Rust and CUDA primitive executors | Rising-edge storage transition now occurs before asynchronous `Q/Q31` taps settle. |

## What the September 1 fanout commit did not solve

The new code adds macro inputs to `fanouts_start`/`fanouts`, but production
topological traversal and Boomerang placement still branch only on
`DriverType::AndGate`. The kernel still gathers all macro inputs, executes one
DSP loop, then one CARRY loop, then one SRLC loop. A same-cycle combinational
macro consumer can therefore sample the producer's old global-state value.

`src/primitive_models.rs::build_typed_schedule` now defines and tests the
required topological/type-queue behavior, including cycle rejection, but the
remote Boomerang serializer/kernel has not yet been converted to consume this
schedule. This is the remaining PS-level architectural blocker. The ZIP must
not be represented as a proven complete Part B implementation until that
integration and a chained-macro GPU differential test pass.

## Verification evidence

Run `scripts/verify_submission.sh` to regenerate the evidence. The expected
gates are:

1. all-three and CARRY-only Yosys preservation;
2. a parameter sidecar containing DSP `PREG` and SRLC `INIT`;
3. exhaustive 1,024-vector CARRY4 test;
4. DSP control/boundary tests;
5. SRLC current/next-state test;
6. typed mixed-DAG ordering and cycle detection;
7. all-target Rust check;
8. heterogeneous parser/mapper and CPU synth-to-VCD execution;
9. 300-cycle independent Icarus-Verilog differential simulation;
10. CUDA release translation when NVCC exists.

GPU execution/performance requires a working NVIDIA driver. Compilation alone
is not evidence of runtime correctness or speedup.
