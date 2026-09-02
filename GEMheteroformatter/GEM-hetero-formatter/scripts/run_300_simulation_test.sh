#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-"$repo_dir/build/test300"}
mkdir -p -- "$build_dir"
cd -- "$repo_dir"

echo "[1/5] Generate 300-cycle independent HDL oracle"
iverilog -g2012 -s tb_300 -o "$build_dir/oracle.vvp" \
    tests/hetero/behavioral_zenith_macros.sv \
    tests/hetero/preservation_top.sv \
    tests/hetero/tb_300.sv
vvp "$build_dir/oracle.vvp" \
    "+VCD=$build_dir/oracle.vcd" \
    "+EXPECTED=$build_dir/expected.csv"

echo "[2/5] Synthesize while preserving heterogeneous macros"
scripts/run_synth_zenith.sh \
    tests/hetero/preservation_top.sv preservation_top "$build_dir/gatelevel.gv" \
    >"$build_dir/yosys.log"

echo "[3/5] Extract parameters and map partitions"
python3 scripts/sanitize_gv.py "$build_dir/gatelevel.gv"
if ! GEM_PARAMS_FILE="$build_dir/params.json" \
    cargo run --quiet --release --features cuda --bin cut_map_interactive -- \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    >"$build_dir/map.log" 2>&1; then
    tail -80 "$build_dir/map.log" >&2
    exit 1
fi

echo "[4/5] Run GEM's flattened CPU simulator"
if ! GEM_PARAMS_FILE="$build_dir/params.json" \
    cargo run --quiet --release --bin flatten_test -- \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    "$build_dir/oracle.vcd" "$build_dir/gem-output.vcd" \
    --input-vcd-scope tb_300/uut \
    >"$build_dir/gem-sim.log" 2>&1; then
    tail -80 "$build_dir/gem-sim.log" >&2
    exit 1
fi

echo "[5/5] Compare all macro outputs for all 300 cycles"
python3 tests/hetero/check_300_vcd.py \
    "$build_dir/expected.csv" "$build_dir/gem-output.vcd"

echo "300-cycle heterogeneous simulation test passed"
