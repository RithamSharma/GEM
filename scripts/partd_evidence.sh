#!/usr/bin/env bash
# ===========================================================================
#  Part D evidence run -- one command, everything to partd_evidence.log
#
#    ./scripts/partd_evidence.sh                 # bench_mac, 2000 cycles
#    ./scripts/partd_evidence.sh <design.sv> <top> [cycles] [num_blocks]
#
#  Does, in order and without stopping on the first failure:
#    1. build the V2 engine + tools
#    2. merge sanity: scheduler proof test, direct carry chain, 300-cycle mixed
#    3. Part D benchmark: preserved (V2) vs shredded (V1), correctness-gated
#    4. dump every result file
#
#  Send back:  partd_evidence.log   and   benchmark-results/partd_*.json
# ===========================================================================
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
chmod +x scripts/*.sh 2>/dev/null || true
sed -i 's/\r$//' scripts/*.sh scripts/*.py 2>/dev/null || true

LOG="$repo/partd_evidence.log"
exec > >(tee "$LOG") 2>&1
sep() { printf '\n\033[1;36m========================= %s =========================\033[0m\n' "$1"; }
rc_note() { local rc=$1 name=$2; if [[ $rc -eq 0 ]]; then echo "[OK]   $name"; else echo "[FAIL rc=$rc]   $name"; fi; }

sep "environment"
uname -a || true
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc: MISSING"
nvidia-smi --query-gpu=name,driver_version,multiprocessor_count --format=csv,noheader 2>/dev/null || echo "nvidia-smi: MISSING"
cargo --version || true
yosys --version 2>/dev/null | head -1 || echo "yosys: MISSING"
iverilog -V 2>/dev/null | head -1 || echo "iverilog: MISSING"
git rev-parse HEAD 2>/dev/null || echo "(not a git tree)"

sep "1. build V2 engine + tools"
cargo build --release --features v2 --bin cuda_test --bin formatter_gpu_test --bin cut_map_interactive
rc_note $? "cargo build --release --features v2"

sep "2a. merge sanity -- scheduler proof unit tests"
cargo test --features v2 --lib schedule
rc_note $? "cargo test --lib schedule"

sep "2b. merge sanity -- direct CARRY4 -> CARRY4 chain (V2, 2 blocks, 1024 vectors)"
bash scripts/run_v2_carry_chain_test.sh
rc_note $? "run_v2_carry_chain_test.sh"

sep "2c. merge sanity -- 300-cycle mixed DSP/CARRY4/SRLC32E/AIG (HDL == CPU == CUDA)"
bash scripts/run_v2_300_simulation_test.sh
rc_note $? "run_v2_300_simulation_test.sh"

sep "3. PART D -- preserved (V2) vs shredded (V1) throughput"
DESIGN=${1:-tests/hetero/bench_mac.sv}
TOP=${2:-bench_mac}
CYCLES=${3:-2000}
NB=${4:-0}
bash scripts/run_partd_benchmark.sh "$DESIGN" "$TOP" "$CYCLES" "$NB"
PARTD_RC=$?
rc_note $PARTD_RC "run_partd_benchmark.sh $DESIGN $TOP"

if [[ $PARTD_RC -ne 0 && "$TOP" == "bench_mac" ]]; then
    sep "3b. PART D fallback -- preservation_top.sv (1 DSP + 1 CARRY4 + 1 SRLC32E, fully verified)"
    bash scripts/run_partd_benchmark.sh tests/hetero/preservation_top.sv preservation_top "$CYCLES" "$NB"
    rc_note $? "run_partd_benchmark.sh preservation_top"
fi

sep "4. result files"
for f in benchmark-results/partd_summary.txt benchmark-results/partd_preserved.json \
         benchmark-results/partd_shredded.json; do
    echo "----- $f -----"
    cat "$f" 2>/dev/null || echo "(missing)"
    echo
done
echo "----- build/partd tail logs (last 25 lines each) -----"
for f in build/partd/*.log; do
    [[ -f "$f" ]] || continue
    echo "=== $f ==="; tail -25 "$f"; echo
done

sep "done -- send partd_evidence.log + benchmark-results/partd_*.json"
