# GEM heterogeneous-primitive submission

This tree is based on `Aishwary-Kumar-25/GEM` branch
`updated_gem_branch` at `0764603`, repaired in an isolated working copy.

## Supported contract

- `CARRY4`: standard `CI`, `CYINIT`, `DI`, `S`, `O`, and `CO`; exact four-stage
  carry equation in one evaluation.
- `SRLC32E`: standard pins and 32-bit `INIT`; a CE-controlled rising edge
  updates storage and the asynchronous `Q/Q31` taps settle to that new storage
  in the same HDL timestamp.
- `DSP48E2`: `PREG=1`, `USE_MULT="MULTIPLY"`, `USE_SIMD="ONE48"`,
  `ALUMODE=0`, and real OPMODE encodings `9'h030` (C), `9'h005` (M), and
  `9'h025` (P+M). `INMODE[2] & !INMODE[3]` enables the 27-bit pre-adder.
  `CEP` and `RSTP` are honored. Unsupported controls fail instead of silently
  becoming a hold operation.

## Important repairs

1. `scripts/run_synth_zenith.sh` is now the synthesis entry point. It expands
   paths safely, accepts designs containing any subset of the three macros,
   and checks that synthesis did not reduce the number of instantiated macros.
2. Parameters survive the NetlistDB compatibility sanitizer in a JSON sidecar.
   Set `GEM_PARAMS_FILE` when invoking GEM manually; the supplied run scripts
   do this automatically.
3. The CUDA DSP model uses real supported control encodings, exact 27/18/45/48
   widths, signed extension, reset, and enable behavior.
4. SRLC initialization is included in the flattened initial-state image.
5. The CPU partition executor evaluates all three macro types and is used by
   `cuda_test --check-with-cpu` as a correctness gate.
6. `src/primitive_models.rs` provides exhaustive CARRY4 tests, DSP boundary and
   control tests, SRLC timing tests, and a cycle-detecting typed topological
   scheduler contract for mixed AIG/macro dependency levels.
7. `scripts/run_300_simulation_test.sh` drives 300 deterministic cycles through
   an independent HDL oracle and GEM's synth/map/CPU-simulation path, comparing
   every DSP, CARRY, and SRLC output bit.
8. Short VCD binary vectors are restored to their declared width before bus
   mapping; VCD producers are allowed to omit leading zeroes.
9. CUDA compilation now uses C++17 and defaults to compute capability 7.5,
   avoiding CUDA 13's removed `sm_50` target.
10. Benchmarking no longer changes branches in one mutable worktree and refuses
   to record timing unless the CPU comparison succeeds on every run.

## Verify

```bash
./scripts/verify_submission.sh
```

The script runs synthesis preservation for all three macros and a CARRY-only
subset, sanitizes parameters, runs the Rust model tests, checks every Rust
target, maps and CPU-simulates the supplied heterogeneous fixture, and compiles
the CUDA executable when NVCC is installed. It also runs the 300-cycle
independent HDL differential test. Exact observed results are recorded in
`VERIFICATION_RESULTS.md`. A GPU runtime test requires a working NVIDIA driver
and cannot be replaced by compilation.

To run only the 300-cycle test:

```bash
./scripts/run_300_simulation_test.sh
```

## Manual flow

```bash
# 1. Preserve the macros and synthesize the rest to GEM's AIG PDK.
./scripts/run_synth_zenith.sh design.sv top build/gatelevel.gv

# 2. Preserve cell parameters for the Rust parser.
python3 scripts/sanitize_gv.py build/gatelevel.gv
export GEM_PARAMS_FILE="$PWD/build/params.json"

# 3. Map the circuit.
cargo run --release --features cuda --bin cut_map_interactive -- \
  build/gatelevel.gv build/result.gemparts

# 4. Simulate. Choose NUM_BLOCKS for the installed GPU and keep the CPU gate.
cargo run --release --features cuda --bin cuda_test -- \
  build/gatelevel.gv build/result.gemparts input.vcd build/output.vcd NUM_BLOCKS \
  --check-with-cpu --input-vcd-scope tb/uut --output-vcd-scope tb/uut
```

## Benchmark

```bash
python3 benchmarks/run_benchmarks.py \
  --bin target/release/cuda_test \
  --gatelevel build/gatelevel.gv --gemparts build/result.gemparts \
  --input-vcd input.vcd --num-blocks NUM_BLOCKS \
  --input-scope tb/uut --output-scope tb/uut --ncu
```

Use separate Git worktrees for the baseline and this submission. Compare the
generated JSON medians and profiler reports only on identical hardware,
waveforms, cycles, and block counts.

## Evidence boundary

The verification script is the source of truth. CUDA translation can be
verified without a driver; kernel execution and performance cannot. Do not
claim a GPU speedup until `nvidia-smi`, the CPU comparison, repeated timing,
and Nsight Compute all succeed on the target machine.
