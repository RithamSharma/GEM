#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
cd -- "$repo_dir"

echo "[1/9] Shell and Python syntax"
bash -n scripts/run_synth_zenith.sh benchmark.sh run_cut_map.sh run_simulate.sh
python3 -m py_compile scripts/sanitize_gv.py benchmarks/run_benchmarks.py tests/hetero/check_300_vcd.py

echo "[2/9] All-macro synthesis preservation"
scripts/run_synth_zenith.sh tests/hetero/preservation_top.sv preservation_top "$work_dir/all.gv" >/dev/null

echo "[3/9] Subset-safe synthesis preservation"
scripts/run_synth_zenith.sh tests/hetero/carry_only.sv carry_only "$work_dir/carry.gv" >/dev/null

echo "[4/9] Parameter sidecar"
python3 scripts/sanitize_gv.py "$work_dir/all.gv"
grep -q '"PREG"' "$work_dir/params.json"
grep -q '"INIT"' "$work_dir/params.json"

echo "[5/9] Rust models and all targets"
cargo test --lib primitive_models
cargo check --all-targets

echo "[6/9] Heterogeneous parser and mapper"
GEM_PARAMS_FILE="$work_dir/params.json" \
    cargo run --release --features cuda --bin cut_map_interactive -- \
    "$work_dir/all.gv" "$work_dir/all.gemparts"

echo "[7/9] CPU end-to-end heterogeneous simulation"
GEM_PARAMS_FILE="$work_dir/params.json" \
    cargo run --release --bin flatten_test -- \
    "$work_dir/all.gv" "$work_dir/all.gemparts" \
    tests/hetero/input.vcd "$work_dir/cpu-output.vcd"
test -s "$work_dir/cpu-output.vcd"

echo "[8/9] Independent 300-cycle HDL differential simulation"
command -v iverilog >/dev/null
command -v vvp >/dev/null
scripts/run_300_simulation_test.sh "$work_dir/test300"

echo "[9/9] CUDA translation (when NVCC is installed)"
if command -v nvcc >/dev/null 2>&1; then
    cargo build --release --features cuda --bin cuda_test
else
    echo "SKIP: nvcc is not installed"
fi

echo "Submission verification passed"
