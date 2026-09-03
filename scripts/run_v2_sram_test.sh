#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-"$repo_dir/build/sram"}
mkdir -p -- "$build_dir"
cd -- "$repo_dir"

# The testbench uses stable relative artifact names.
mkdir -p build/sram
iverilog -g2012 -s tb_sram -o "$build_dir/oracle.vvp" \
    tests/hetero/sram_top.sv tests/hetero/tb_sram.sv
vvp "$build_dir/oracle.vvp" \
    "+VCD=$build_dir/oracle.vcd" "+EXPECTED=$build_dir/expected.csv"

scripts/run_synth_zenith.sh tests/hetero/sram_top.sv sram_top "$build_dir/gatelevel.gv" \
    >"$build_dir/yosys.log"
python3 scripts/sanitize_gv.py "$build_dir/gatelevel.gv"

GEM_PARAMS_FILE="$build_dir/params.json" \
    cargo run --quiet --release --features cuda --bin cut_map_interactive -- \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    --v2-parts --v2-num-partitions 4 >"$build_dir/map.log" 2>&1
cargo build --quiet --release --features v2 --bin cuda_test

GEM_PARAMS_FILE="$build_dir/params.json" target/release/cuda_test \
    "$build_dir/gatelevel.gv" "$build_dir/result.gemparts" \
    "$build_dir/oracle.vcd" "$build_dir/gem-output.vcd" 4 \
    --input-vcd-scope tb_sram/uut --check-with-cpu --v2 \
    >"$build_dir/gem.log" 2>&1

python3 tests/hetero/check_sram_vcd.py "$build_dir/expected.csv" "$build_dir/gem-output.vcd"
