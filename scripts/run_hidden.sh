#!/usr/bin/env bash
# ===========================================================================
#  Judge hand-off pipeline:  design.v  +  numeric stimulus DB  ->  output VCD
#
#  The panel supplies an RTL/gate design plus a "database file with numbers for
#  each signal per simulation step". GEM reads stimulus as VCD, so this bridges
#  the two and runs the full heterogeneous (V2) engine end to end, with the CPU
#  interpreter as a per-cycle correctness gate.
#
#    ./scripts/run_hidden.sh <design.v> <top> <stim.db> [num_blocks] [-- <stim_to_vcd args>]
#
#  Examples
#    ./scripts/run_hidden.sh hidden.v top hidden_stim.csv
#    ./scripts/run_hidden.sh hidden.v top stim.txt 14 -- --ports clk:1,a:27,b:18 --radix hex
#    SCOPE=tb_hidden/uut ./scripts/run_hidden.sh hidden.v top stim.csv
#
#  Stimulus DB formats (auto-detected; see scripts/stim_to_vcd.py --help):
#    - CSV/TSV/whitespace, one row per cycle, header row OR positional --ports
#    - values decimal by default, hex with --radix hex
#    - an existing .vcd is passed straight through
#
#  Requires: yosys, iverilog(optional), python3, CUDA toolkit + NVIDIA GPU.
#  If the design already instantiates DSP48E2/CARRY4/SRLC32E they are kept
#  native; plain RTL is synthesized against the same library.
# ===========================================================================
set -euo pipefail
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

b="$repo/build/hidden"; rm -rf "$b"; mkdir -p "$b"
SCOPE=${SCOPE:-tb/uut}

echo "== [1/6] synthesize (DSP48E2 / CARRY4 / SRLC32E preserved) =="
bash scripts/run_synth_zenith.sh "$DESIGN" "$TOP" "$b/gatelevel.gv" | tee "$b/yosys.log"

echo "== [2/6] preserve cell parameters into a JSON sidecar =="
python3 scripts/sanitize_gv.py "$b/gatelevel.gv"

echo "== [3/6] schedule + heterogeneous V2 placement ($NB partitions) =="
GEM_PARAMS_FILE="$b/params.json" \
    cargo run -q --release --features cuda --bin cut_map_interactive -- \
    "$b/gatelevel.gv" "$b/result.gemparts" \
    --v2-parts --v2-num-partitions "$NB" 2>&1 | tee "$b/map.log"

echo "== [4/6] convert numeric stimulus DB -> VCD =="
python3 scripts/stim_to_vcd.py --stim "$STIM" --out "$b/stim.vcd" \
    --scope "$SCOPE" "${STIM_ARGS[@]}"

echo "== [5/6] build the unified V2 CUDA simulator =="
cargo build -q --release --features v2 --bin cuda_test

echo "== [6/6] simulate on GPU (engine auto), gating every cycle against the CPU oracle =="
GEM_PARAMS_FILE="$b/params.json" target/release/cuda_test \
    "$b/gatelevel.gv" "$b/result.gemparts" "$b/stim.vcd" "$b/output.vcd" "$NB" \
    --engine auto --check-with-cpu \
    --input-vcd-scope "$SCOPE" --output-vcd-scope "$SCOPE" 2>&1 | tee "$b/gem.log"

echo
echo "  output waveform : $b/output.vcd"
echo "  input VCD       : $b/stim.vcd"
echo "  logs            : $b/{yosys,map,gem}.log"
grep -E 'engine=auto|selected simulation engine|simulation, Elapsed=|total number of cycles|sanity test passed' "$b/gem.log" || true
