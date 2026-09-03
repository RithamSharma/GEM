#!/usr/bin/env bash
# ===========================================================================
#  Upstream-GEM smoke test (part 2 of 2): simulate the reference circuit that
#  scripts/upstream_smoke_map.sh just mapped, with the CLASSIC V1 engine, and
#  compare against test_circuit/golden_output.vcd.
#
#    bash scripts/upstream_smoke_map.sh   (first)
#    bash scripts/upstream_smoke_sim.sh   (then this, from the repo root)
# ===========================================================================
set -uo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p ~/.cuda_workaround/lib64
ln -sf /usr/lib/x86_64-linux-gnu/* ~/.cuda_workaround/lib64/ 2>/dev/null || true
export CUDA_LIBRARY_PATH=~/.cuda_workaround
export GEM_PARAMS_FILE="$PWD/test_circuit/params.json"

echo "Running cuda_test (V1)..."
cargo run -r --features cuda --bin cuda_test -- \
    test_circuit/gatelevel.gv test_circuit/result.gemparts \
    test_circuit/golden_output.vcd test_circuit/gem_output.vcd 108 \
    --input-vcd-scope tb_top/uut --output-vcd-scope tb_top/uut
