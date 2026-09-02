#!/bin/bash

# Ensure the CUDA workaround directory exists
mkdir -p ~/.cuda_workaround/lib64
ln -sf /usr/lib/x86_64-linux-gnu/* ~/.cuda_workaround/lib64/ 2>/dev/null

# Export the required variable for the Rust CUDA wrapper
export CUDA_LIBRARY_PATH=~/.cuda_workaround

echo "Running Yosys Synthesis..."
yosys test_circuit/synth.ys

echo "Running cut_map_interactive..."
python3 scripts/sanitize_gv.py test_circuit/gatelevel.gv
export GEM_PARAMS_FILE="$PWD/test_circuit/params.json"
cargo run -r --features cuda --bin cut_map_interactive -- test_circuit/gatelevel.gv test_circuit/result.gemparts
