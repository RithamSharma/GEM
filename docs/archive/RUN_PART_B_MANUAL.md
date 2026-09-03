# Run Part B from a fresh checkout or ZIP

## Prerequisites

- Linux, NVIDIA GPU and CUDA toolkit
- Rust/Cargo
- Yosys, Icarus Verilog, Python 3
- repository submodules present (`eda-infra-rs`)

From the repository root:

```bash
git submodule update --init --recursive
cargo build --release --features v2 --bin cuda_test --bin formatter_gpu_test
```

## Fastest complete verification

```bash
scripts/run_v2_carry_chain_test.sh
scripts/run_v2_sram_test.sh
scripts/run_v2_300_simulation_test.sh
cargo test --features v2 --lib
```

Expected final lines include:

```text
PASS: direct CARRY4-to-CARRY4 dependency matched HDL for 1024 vectors
PASS: V2 SRAM matched independent HDL for 3 sampled edges
PASS: GEM matched the independent HDL oracle for all 300 cycles
test result: ok. 49 passed; 0 failed
```

## Run your own RTL

```bash
scripts/run_synth_zenith.sh design.sv top build/design/gatelevel.gv
python3 scripts/sanitize_gv.py build/design/gatelevel.gv

GEM_PARAMS_FILE=build/design/params.json \
  target/release/cut_map_interactive \
  build/design/gatelevel.gv build/design/result.gemparts \
  --v2-parts --v2-num-partitions 14

GEM_PARAMS_FILE=build/design/params.json \
  target/release/cuda_test \
  build/design/gatelevel.gv build/design/result.gemparts \
  input.vcd output.vcd 14 \
  --input-vcd-scope your_testbench/uut --check-with-cpu --v2
```

Begin with one block, then benchmark larger counts on the real design.
`--check-with-cpu` should remain enabled for every reported result.

## Sanitizers, benchmarks, and Nsight

```bash
compute-sanitizer --tool memcheck  target/release/formatter_gpu_test
compute-sanitizer --tool racecheck target/release/formatter_gpu_test
compute-sanitizer --tool synccheck  target/release/formatter_gpu_test

benchmarks/run_v2_gpu_benchmark.py \
  --bin target/release/formatter_gpu_test \
  --out benchmark-results/v2_macro_runtime.json

benchmarks/run_v2_simulation_benchmark.py --help
sudo scripts/profile_v2_ncu.sh
```

Root is only needed when Linux restricts GPU performance counters.
