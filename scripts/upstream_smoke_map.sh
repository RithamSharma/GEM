#!/usr/bin/env bash
# ===========================================================================
#  Upstream-GEM smoke test (part 1 of 2): synthesize + map the tiny reference
#  circuit in test_circuit/ with the CLASSIC V1 flow. Not part of the
#  heterogeneous-macro submission -- use it only to confirm the base GEM
#  toolchain builds and runs on your machine.
#
#    bash scripts/upstream_smoke_map.sh      (run from the repo root)
#    bash scripts/upstream_smoke_sim.sh      (then this)
# ===========================================================================
set -uo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Some CUDA installs need the host libs visible under one prefix.
mkdir -p ~/.cuda_workaround/lib64
ln -sf /usr/lib/x86_64-linux-gnu/* ~/.cuda_workaround/lib64/ 2>/dev/null || true
export CUDA_LIBRARY_PATH=~/.cuda_workaround

echo "Running Yosys synthesis (test_circuit/synth.ys)..."
yosys test_circuit/synth.ys

echo "Running cut_map_interactive..."
python3 scripts/sanitize_gv.py test_circuit/gatelevel.gv
export GEM_PARAMS_FILE="$PWD/test_circuit/params.json"
cargo run -r --features cuda --bin cut_map_interactive -- \
    test_circuit/gatelevel.gv test_circuit/result.gemparts
