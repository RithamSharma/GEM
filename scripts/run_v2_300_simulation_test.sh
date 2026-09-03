#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-"$repo_dir/build/test300-v2"}
num_blocks=${GEM_V2_BLOCKS:-14}
mkdir -p -- "$build_dir"
cd -- "$repo_dir"

echo "[1/6] Generate the independent 300-cycle HDL oracle"
iverilog -g2012 -s tb_300 -o "$build_dir/oracle.vvp" \
    tests/hetero/behavioral_zenith_macros.sv \
    tests/hetero/preservation_top.sv \
    tests/hetero/tb_300.sv
vvp "$build_dir/oracle.vvp" \
    "+VCD=$build_dir/oracle.vcd" \
    "+EXPECTED=$build_dir/expected.csv"

echo "[2/6] Preserve DSP48E2, CARRY4 and SRLC32E during synthesis"
scripts/run_synth_zenith.sh \
    tests/hetero/preservation_top.sv preservation_top "$build_dir/gatelevel.gv" \
    >"$build_dir/yosys.log"
python3 scripts/sanitize_gv.py "$build_dir/gatelevel.gv"

echo "[3/6] Build the partition artifact required by the compatible CLI"
GEM_PARAMS_FILE="$build_dir/params.json" \
    cargo run --quiet --release --features cuda --bin cut_map_interactive -- \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" --v2-parts \
    --v2-num-partitions "$num_blocks" \
    >"$build_dir/map.log" 2>&1

echo "[4/6] Build the unified V2 CUDA simulator"
cargo build --quiet --release --features v2 --bin cuda_test

echo "[5/6] Run V2 on the GPU and gate every cycle against the CPU V2 oracle"
GEM_PARAMS_FILE="$build_dir/params.json" \
    target/release/cuda_test \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    "$build_dir/oracle.vcd" "$build_dir/gem-v2-output.vcd" "$num_blocks" \
    --input-vcd-scope tb_300/uut --check-with-cpu --v2 \
    >"$build_dir/gem-v2.log" 2>&1

echo "[6/6] Compare V2 outputs with the independent HDL oracle"
python3 tests/hetero/check_300_vcd.py \
    "$build_dir/expected.csv" "$build_dir/gem-v2-output.vcd"

echo "V2 CPU/CUDA/HDL 300-cycle verification passed"
