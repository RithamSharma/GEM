#!/usr/bin/env bash
# ===========================================================================
#  Run a design with the AUTO engine: GEM picks V1 (classic Boomerang) or V2
#  (heterogeneous wave engine) per design.
#
#    ./scripts/run_best.sh <design.v> <top> <stim.(csv|vcd)> [num_blocks] [-- <stim_to_vcd args>]
#
#  The pick rule (printed at run time):
#    * V2 is FORCED whenever a macro output feeds a same-cycle consumer -- the
#      batched V1 path would read stale state there (this is the PS Part B
#      correctness requirement).
#    * otherwise the faster of V1 / V2 is chosen from a cost estimate.
#
#  Synthesis and mapping happen ONCE (macros preserved); the same .gemparts
#  carries both the V2 placement and the embedded legacy V1 partitions, so the
#  dispatcher has both engines available with no second synthesis.
#
#  Requires: yosys, python3, CUDA toolkit + NVIDIA GPU. (iverilog optional.)
# ===========================================================================
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
chmod +x scripts/*.sh 2>/dev/null || true

[[ $# -ge 3 ]] || { grep -E '^#( |$)' "$0" | sed 's/^# \?//'; exit 2; }
DESIGN=$1 TOP=$2 STIM=$3; shift 3
NB=""
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then NB=$1; shift; fi
[[ "${1:-}" == "--" ]] && shift
STIM_ARGS=("$@")
if [[ -z "$NB" ]]; then
    NB=$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader -i 0 2>/dev/null | head -1 || echo 1)
fi
[[ "$NB" =~ ^[0-9]+$ ]] || NB=1

b="$repo/build/best"; rm -rf "$b"; mkdir -p "$b"
SCOPE=${SCOPE:-tb/uut}

echo "== [1/6] synthesize (macros preserved) =="
bash scripts/run_synth_zenith.sh "$DESIGN" "$TOP" "$b/gatelevel.gv" | tee "$b/yosys.log"

echo "== [2/6] parameter sidecar =="
python3 scripts/sanitize_gv.py "$b/gatelevel.gv"

echo "== [3/6] schedule + placement (V2 wrapper embeds the legacy V1 payload) =="
GEM_PARAMS_FILE="$b/params.json" \
    cargo run -q --release --features cuda --bin cut_map_interactive -- \
    "$b/gatelevel.gv" "$b/result.gemparts" --v2-parts --v2-num-partitions "$NB" 2>&1 | tee "$b/map.log"

echo "== [4/6] stimulus -> VCD =="
python3 scripts/stim_to_vcd.py --stim "$STIM" --out "$b/stim.vcd" --scope "$SCOPE" "${STIM_ARGS[@]}" \
    || cp "$STIM" "$b/stim.vcd"

echo "== [5/6] build cuda_test =="
cargo build -q --release --features v2 --bin cuda_test

echo "== [6/6] simulate with --engine auto (CPU-gated) =="
GEM_PARAMS_FILE="$b/params.json" target/release/cuda_test \
    "$b/gatelevel.gv" "$b/result.gemparts" "$b/stim.vcd" "$b/output.vcd" "$NB" \
    --engine auto --check-with-cpu \
    --input-vcd-scope "$SCOPE" --output-vcd-scope "$SCOPE" 2>&1 | tee "$b/gem.log"

echo
grep -E 'engine=auto|selected simulation engine|simulation, Elapsed=|total number of cycles|sanity test passed' "$b/gem.log" || true
echo "  output waveform: $b/output.vcd"
