#!/usr/bin/env bash
# ===========================================================================
#  GEM heterogeneous-macro simulator — ONE-COMMAND compile + verify + benchmark
#
#  Judges: run  compile.bat  (Windows) or this script directly (Linux / WSL2).
#
#    ./compile.sh                full: build + unit tests + A/B/C/D correctness
#                               gates + Part D throughput sweep + Nsight profile
#    ./compile.sh --quick       build + unit tests + functional gates only
#    ./compile.sh --build-only   just compile the simulator
#    ./compile.sh --bench <design.sv> <top>   build + benchmark ONE netlist
#
#  Everything (logs, JSON, .ncu-rep, netlists) lands in  ./submission-results/.
#
#  Needs (Linux / WSL2): rust+cargo, CUDA toolkit (nvcc) + NVIDIA driver,
#  Yosys 0.68, Icarus Verilog, Python 3.  ncu (Nsight Compute) is optional.
# ===========================================================================
set -uo pipefail
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chmod +x scripts/*.sh 2>/dev/null || true
sed -i 's/\r$//' scripts/*.sh scripts/*.py compile.sh 2>/dev/null || true

MODE=${1:-full}
OUT=submission-results
rm -rf "$OUT"; mkdir -p "$OUT"
exec > >(tee "$OUT/compile.log") 2>&1

hr() { printf '\n=================  %s  =================\n' "$1"; }

# --------------------------------------------------------------------- 0. env
hr "environment"
for t in cargo rustc nvcc yosys iverilog vvp python3 ncu nvidia-smi; do
    printf '  %-9s : ' "$t"
    if command -v "$t" >/dev/null 2>&1; then "$t" --version 2>&1 | head -1; else echo "NOT FOUND"; fi
done
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,multiprocessor_count \
    --format=csv 2>/dev/null || echo "  (no GPU visible)"

missing=""
for t in cargo nvcc yosys; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
if [[ -n "$missing" ]]; then
    echo
    echo "[FATAL] missing required tool(s): $missing"
    echo "        install everything with:   bash scripts/install_deps.sh"
    echo "        (then open a new shell and re-run ./compile.sh)"
    exit 1
fi

# ------------------------------------------------- 1. CUDA architecture flags
# Embed forward-compatible PTX (JITs onto ANY newer GPU, incl. Blackwell /
# RTX 50-series), and add native SASS when this CUDA toolkit knows the arch.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d ' .')
export UCC_CUDA_PTX=75
if [[ "$CC" =~ ^[0-9]+$ ]] && nvcc --list-gpu-arch 2>/dev/null | grep -q "compute_$CC"; then
    export UCC_CUDA_GENCODE="$CC"
    export UCC_CUDA_PTX="$CC"
    echo "  CUDA codegen : native sm_$CC + PTX compute_$CC"
else
    echo "  CUDA codegen : portable PTX compute_75 (driver JITs it forward to your GPU)"
fi

# ------------------------------------------------------ 2. submodules + build
hr "build  (cargo build --release --features v2)"
git submodule update --init --recursive 2>/dev/null || true
if ! cargo build --release --features v2; then
    echo "[FATAL] build failed — see the compiler output above."
    exit 1
fi
echo "  binaries: target/release/{cut_map_interactive,cuda_test,formatter_gpu_test}"
[[ "$MODE" == "--build-only" ]] && { echo "build-only: done."; exit 0; }

# ----------------------------------------------- optional: benchmark ONE file
if [[ "$MODE" == "--bench" ]]; then
    DES=${2:?usage: ./compile.sh --bench <design.sv> <top>}
    TOP=${3:?usage: ./compile.sh --bench <design.sv> <top>}
    hr "benchmark  $TOP  ($DES)"
    bash scripts/run_partd_benchmark.sh "$DES" "$TOP" "${4:-4000}" "${5:-0}" || true
    cp -f benchmark-results/partd_summary.txt benchmark-results/partd_*.json "$OUT/" 2>/dev/null || true
    if command -v ncu >/dev/null 2>&1 && [[ -f build/partd/preserved.gemparts ]]; then
        GEM_NCU_GV=build/partd/preserved.gv \
        GEM_NCU_PARTS=build/partd/preserved.gemparts \
        GEM_NCU_PARAMS=build/partd/preserved.params.json \
        GEM_NCU_VCD=build/partd/stim.vcd GEM_NCU_SCOPE="${TOP}/uut" \
            bash scripts/profile_v2_ncu.sh || echo "  (ncu skipped/failed — see above)"
        cp -f benchmark-results/part_b_v2_integrated.* "$OUT/" 2>/dev/null || true
    fi
    echo "  results in ./$OUT/"; exit 0
fi

# ---------------------------------------------------------- 3. host unit tests
hr "host unit tests  (cargo test --features v2 --lib)"
cargo test --release --features v2 --lib 2>&1 | tail -25

# --------------------------------------------- 4. correctness gates (A/B/C/D)
hr "correctness + throughput + Nsight  (scripts/judge_verify_all.sh)"
QV=""; [[ "$MODE" == "--quick" ]] && QV="--quick"
bash scripts/judge_verify_all.sh $QV || true
cp -r verify_logs "$OUT/verify_logs" 2>/dev/null || true
cp -f benchmark-results/part*_v2*.ncu-rep "$OUT/" 2>/dev/null || true
cp -f benchmark-results/partd_summary.txt benchmark-results/partd_*.json "$OUT/" 2>/dev/null || true

[[ "$MODE" == "--quick" ]] && { hr "quick run done — see ./$OUT/"; ls -la "$OUT"; exit 0; }

# --------------------------------- 5. multi-block throughput scaling (Part D)
hr "throughput scaling sweep  (scripts/partd_sweep.sh)"
bash scripts/partd_sweep.sh || true
cp -f partd_sweep.log "$OUT/" 2>/dev/null || true
cp -rf benchmark-results/partd-sweep "$OUT/" 2>/dev/null || true

# --------------------------------------------------------------- 6. summarise
hr "SUMMARY"
{
  echo "GEM heterogeneous-macro — compile.sh summary"
  date
  echo
  echo "--- verification gates (full detail: $OUT/compile.log, $OUT/verify_logs/) ---"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/compile.log" 2>/dev/null \
    | grep -E '(^| )(PASS|FAIL|SKIP) ' | sed 's/^ *//' | sort -u || echo "(see compile.log)"
  echo
  [[ -f "$OUT/partd_summary.txt" ]] && { echo "--- Part D (single config) ---"; cat "$OUT/partd_summary.txt"; }
  echo
  [[ -f "$OUT/partd_sweep.log" ]] && { echo "--- Part D scaling sweep tail ---"; tail -20 "$OUT/partd_sweep.log"; }
} | tee "$OUT/SUMMARY.txt"

hr "DONE — deliverable artifacts in ./$OUT/"
ls -la "$OUT"
