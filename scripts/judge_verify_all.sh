#!/usr/bin/env bash
# ===========================================================================
#  GEM heterogeneous-macro simulation  --  one-command verification for a judge.
#
#  Runs every correctness gate for Parts A / B / C / D in order, SKIPS any step
#  whose tools are missing, writes each step's full output under ./verify_logs/,
#  and prints a PASS / FAIL / SKIP summary naming the exact log for any FAIL.
#  Nothing here modifies the repository.
#
#    ./scripts/judge_verify_all.sh            # everything available
#    ./scripts/judge_verify_all.sh --quick    # host tests + GPU functional only
#                                             # (skip Yosys flows and Part D)
#
#  Tooling (each step notes what it needs):
#    Rust / Cargo          host-logic tests, all builds
#    CUDA toolkit + GPU     nvcc, the V2 GPU tests, compute-sanitizer, ncu
#    Yosys                  macro-preservation + end-to-end netlist tests
#    Icarus Verilog         the independent HDL oracles
#    python3               parameter sidecar + benchmark + VCD checkers
# ===========================================================================
set -u
cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# self-heal a Windows checkout: restore +x and strip CRLF (harmless if clean).
chmod +x scripts/*.sh 2>/dev/null || true
sed -i 's/\r$//' scripts/*.sh scripts/*.py 2>/dev/null || true

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

LOGDIR="verify_logs"; rm -rf -- "$LOGDIR"; mkdir -p -- "$LOGDIR"

have() { command -v "$1" >/dev/null 2>&1; }
HAVE_CARGO=$(have cargo && echo 1 || echo 0)
HAVE_NVCC=$(have nvcc && echo 1 || echo 0)
HAVE_GPU=$( (have nvidia-smi && nvidia-smi >/dev/null 2>&1) && echo 1 || echo 0)
HAVE_YOSYS=$(have yosys && echo 1 || echo 0)
HAVE_IVERILOG=$(have iverilog && echo 1 || echo 0)
HAVE_SANI=$(have compute-sanitizer && echo 1 || echo 0)
HAVE_NCU=$(have ncu && echo 1 || echo 0)
HAVE_PY=$(have python3 && echo 1 || echo 0)

declare -a NAMES STATES
RC=0; N=0

step() {   # step "<name>" <0|1 canrun> "<skip reason>" -- <cmd...>
  local name="$1" canrun="$2" reason="$3"; shift 3; [[ "$1" == "--" ]] && shift
  N=$((N + 1))
  local slug; slug=$(printf 'step%02d_%s' "$N" "$(printf '%s' "$name" | tr -cs 'A-Za-z0-9' '_' | cut -c1-46)")
  local log="$LOGDIR/${slug}.log"
  printf '\n\033[1;36m========== [%d] %s ==========\033[0m\n' "$N" "$name"
  if [[ "$canrun" != 1 ]]; then
    printf '\033[1;33mSKIP\033[0m  (%s)\n' "$reason"
    NAMES+=("$name"); STATES+=("SKIP  (${reason})")
    echo "SKIPPED: $reason" > "$log"; return
  fi
  "$@" > >(tee "$log") 2>&1
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    printf '\033[1;32mPASS\033[0m  %s\n' "$name"
    NAMES+=("$name"); STATES+=("PASS")
  else
    printf '\033[1;31mFAIL (exit %d)\033[0m  %s\n' "$rc" "$name"
    printf '      full output: \033[1m%s\033[0m\n' "$log"
    NAMES+=("$name"); STATES+=("FAIL (exit $rc)  ->  $log"); RC=1
  fi
}

printf 'tools: cargo=%s nvcc=%s gpu=%s yosys=%s iverilog=%s sanitizer=%s ncu=%s python3=%s\n' \
  "$HAVE_CARGO" "$HAVE_NVCC" "$HAVE_GPU" "$HAVE_YOSYS" "$HAVE_IVERILOG" "$HAVE_SANI" "$HAVE_NCU" "$HAVE_PY"
printf 'per-step logs: %s/\n' "$LOGDIR"

V2_BUILD_OK=$([[ $HAVE_CARGO -eq 1 && $HAVE_NVCC -eq 1 ]] && echo 1 || echo 0)
GPU_OK=$([[ $V2_BUILD_OK -eq 1 && $HAVE_GPU -eq 1 ]] && echo 1 || echo 0)
HDL_OK=$([[ $GPU_OK -eq 1 && $HAVE_YOSYS -eq 1 && $HAVE_IVERILOG -eq 1 && $HAVE_PY -eq 1 ]] && echo 1 || echo 0)

# 1 -- host logic: topological scheduler (incl. the Part B macro->macro proof),
#      64-bit coalesced formatter, exact macro models. Builds the CUDA lib
#      (feature v2 -> cuda) so it needs nvcc, but runs no GPU kernel.
step "Host-logic + model unit tests (cargo test --features v2 --lib)" \
  "$V2_BUILD_OK" "needs cargo + nvcc (the v2 feature pulls in the CUDA build)" \
  -- cargo test --features v2 --lib

# 2 -- compile the unified V2 CUDA engine (no --quiet: compiler errors must show)
step "Compile V2 GPU engine + tools (cuda_test, formatter_gpu_test, cut_map_interactive)" \
  "$V2_BUILD_OK" "needs cargo + nvcc" \
  -- cargo build --release --features v2 --bin cuda_test --bin formatter_gpu_test --bin cut_map_interactive

# 3 -- Part A: Yosys keeps DSP48E2/CARRY4/SRLC32E as native units
synth_preserve() {
  scripts/run_synth_zenith.sh tests/hetero/preservation_top.sv preservation_top verify_logs/_all.gv &&
  scripts/run_synth_zenith.sh tests/hetero/carry_only.sv       carry_only       verify_logs/_carry.gv &&
  grep -qE '\bDSP48E2\b'  verify_logs/_all.gv &&
  grep -qE '\bCARRY4\b'   verify_logs/_all.gv &&
  grep -qE '\bSRLC32E\b'  verify_logs/_all.gv &&
  echo "PASS: all three macros survived synthesis (and a CARRY-only subset)"
}
step "Part A: macro interception / preservation during Yosys synthesis" \
  "$([[ $HAVE_YOSYS -eq 1 && $HAVE_PY -eq 1 ]] && echo 1 || echo 0)" "needs Yosys + python3" \
  -- synth_preserve

# 4 -- Part A: the 64-bit-aligned coalesced formatter buffer, on the GPU
step "Part A: formatter GPU round-trip (alignment + device-vs-host hash + one upload)" \
  "$GPU_OK" "needs cargo + nvcc + GPU" \
  -- cargo run --release --features v2 --bin formatter_gpu_test

# 5 -- Part B: a DIRECT CARRY4.CO[3] -> CARRY4.CI chain, 1024 HDL vectors,
#      CPU V2 and two-block cooperative CUDA
step "Part B: direct CARRY4->CARRY4 dependency, 1024 vectors, CPU + 2-block CUDA vs HDL" \
  "$HDL_OK" "needs Yosys + Icarus + python3 + GPU" \
  -- bash scripts/run_v2_carry_chain_test.sh

# 6 -- Part C: synthesized synchronous SRAM (read-before-write), HDL vs CPU vs CUDA
step "Part C: synchronous SRAM (write / read / read-before-write) vs independent HDL" \
  "$HDL_OK" "needs Yosys + Icarus + python3 + GPU" \
  -- bash scripts/run_v2_sram_test.sh

# 7 -- Part B/C end-to-end: 300 cycles of mixed DSP48E2 + CARRY4 + SRLC32E + AIG
step "Part B/C: 300-cycle mixed DSP/CARRY4/SRLC32E/AIG, HDL == CPU V2 == CUDA V2" \
  "$HDL_OK" "needs Yosys + Icarus + python3 + GPU" \
  -- bash scripts/run_v2_300_simulation_test.sh

# 7b -- Part B: the --engine auto dispatcher (V1 vs V2 per design, CPU-gated)
step "Part B: --engine auto dispatcher (forces V2 on chained macros, else V1; every run CPU-gated)" \
  "$HDL_OK" "needs Yosys + Icarus + python3 + GPU" \
  -- bash scripts/run_auto_check.sh

# 8 -- Part B: memory / race / sync safety
sanitize() {
  local bin=target/release/formatter_gpu_test tool bad=0
  for tool in memcheck racecheck synccheck; do
    printf -- '\n--- compute-sanitizer --tool %s ---\n' "$tool"
    if compute-sanitizer --tool "$tool" "$bin" 2>&1 | tee "verify_logs/_sani_$tool.txt" | \
         grep -qE 'ERROR SUMMARY: 0 errors|0 hazards displayed|RACECHECK SUMMARY: 0 hazards'; then
      printf '  %s: CLEAN\n' "$tool"
    else
      printf '  %s: NOT CLEAN (see verify_logs/_sani_%s.txt)\n' "$tool" "$tool"; bad=1
    fi
  done
  return $bad
}
step "Part B: Compute Sanitizer (memcheck / racecheck / synccheck)" \
  "$([[ $GPU_OK -eq 1 && $HAVE_SANI -eq 1 ]] && echo 1 || echo 0)" "needs compute-sanitizer + GPU" \
  -- sanitize

# 9 -- Part D: cycles/sec + AIG graph size, macros preserved (V2) vs shredded (V1)
step "Part D: throughput + AIG size, preserved-vs-shredded on identical RTL" \
  "$([[ $QUICK -eq 0 && $HDL_OK -eq 1 ]] && echo 1 || echo 0)" "quick mode, or needs Yosys + Icarus + python3 + GPU" \
  -- bash scripts/run_partd_benchmark.sh

# 10 -- Part D: Nsight Compute profile of the integrated V2 kernel (optional)
profile() {
  scripts/profile_v2_ncu.sh 2>&1 | tee verify_logs/_ncu.txt
  local rc=${PIPESTATUS[0]}
  grep -q ERR_NVGPUCTRPERM verify_logs/_ncu.txt && {
    echo "  GPU perf counters need permission: rerun with sudo (scripts/profile_v2_ncu.sh)"; return 1; }
  return $rc
}
step "Part D: Nsight Compute profile of simulate_v2_cycles_kernel (optional; needs perms)" \
  "$([[ $QUICK -eq 0 && $GPU_OK -eq 1 && $HAVE_NCU -eq 1 ]] && echo 1 || echo 0)" "quick mode, or needs ncu + GPU (+ perf-counter permission)" \
  -- profile

printf '\n\033[1;36m======================== SUMMARY ========================\033[0m\n'
for i in "${!NAMES[@]}"; do
  case "${STATES[$i]}" in
    PASS*) c='\033[1;32m'; tag='PASS' ;;
    FAIL*) c='\033[1;31m'; tag='FAIL' ;;
    *)     c='\033[1;33m'; tag='SKIP' ;;
  esac
  printf "${c}%-5s\033[0m %s\n" "$tag" "${NAMES[$i]}"
  [[ "${STATES[$i]}" == FAIL* ]] && printf '       \033[1;31m%s\033[0m\n' "${STATES[$i]}"
done
printf '\nall step logs: %s/\n\n' "$LOGDIR"
if [[ $RC -eq 0 ]]; then
  printf '\033[1;32mAll runnable gates passed.\033[0m Skipped gates need the tools noted above.\n'
else
  printf '\033[1;31mOne or more gates FAILED\033[0m -- open the log printed next to each FAIL.\n'
fi
exit $RC
