#!/bin/bash

# Ensure the CUDA workaround directory exists
mkdir -p ~/.cuda_workaround/lib64
ln -sf /usr/lib/x86_64-linux-gnu/* ~/.cuda_workaround/lib64/ 2>/dev/null

# Export the required variable for the Rust CUDA wrapper
export CUDA_LIBRARY_PATH=~/.cuda_workaround
export GEM_PARAMS_FILE="$PWD/test_circuit/params.json"

echo "Running cuda_test..."
cargo run -r --features cuda --bin cuda_test -- test_circuit/gatelevel.gv test_circuit/result.gemparts test_circuit/golden_output.vcd test_circuit/gem_output.vcd 108 --input-vcd-scope tb_top/uut --output-vcd-scope tb_top/uut
