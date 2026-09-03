#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-"$repo_dir/build/carry-chain2"}
num_blocks=${GEM_V2_BLOCKS:-2}
mkdir -p -- "$build_dir"
cd -- "$repo_dir"

iverilog -g2012 -s tb_carry_chain2 -o "$build_dir/oracle.vvp" \
    tests/hetero/behavioral_zenith_macros.sv \
    tests/hetero/carry_chain2.sv tests/hetero/tb_carry_chain2.sv
vvp "$build_dir/oracle.vvp" \
    "+VCD=$build_dir/oracle.vcd" "+EXPECTED=$build_dir/expected.csv"

scripts/run_synth_zenith.sh \
    tests/hetero/carry_chain2.sv carry_chain2 "$build_dir/gatelevel.gv" \
    >"$build_dir/yosys.log"
python3 scripts/sanitize_gv.py "$build_dir/gatelevel.gv"

GEM_PARAMS_FILE="$build_dir/params.json" \
    cargo run --quiet --release --features cuda --bin cut_map_interactive -- \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    --v2-parts --v2-num-partitions "$num_blocks" >"$build_dir/map.log" 2>&1
cargo build --quiet --release --features v2 --bin cuda_test

GEM_PARAMS_FILE="$build_dir/params.json" target/release/cuda_test \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    "$build_dir/oracle.vcd" "$build_dir/gem-output.vcd" "$num_blocks" \
    --input-vcd-scope tb_carry_chain2/uut --check-with-cpu --v2 \
    >"$build_dir/gem.log" 2>&1

python3 tests/hetero/check_carry_chain_vcd.py \
    "$build_dir/expected.csv" "$build_dir/gem-output.vcd"
