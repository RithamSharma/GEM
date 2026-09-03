#!/usr/bin/env bash
# ===========================================================================
#  Verify the --engine auto dispatcher: for each fixture, synth once, then run
#  cuda_test --engine auto and report which engine it picked, the timing, and
#  that --check-with-cpu passed. Also runs the same fixture with --engine v1 and
#  --engine v2 explicitly for comparison (v1 is skipped when it would be
#  incorrect).
#
#    ./scripts/run_auto_check.sh                 # farm, bench_mac, carry_chain8
#
#  Send back: auto_check.log
# ===========================================================================
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
chmod +x scripts/*.sh 2>/dev/null || true
LOG="$repo/auto_check.log"; : > "$LOG"
exec > >(tee "$LOG") 2>&1

NB=$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader -i 0 2>/dev/null | head -1 || echo 4)
[[ "$NB" =~ ^[0-9]+$ ]] || NB=4
[[ "$NB" -gt 16 ]] && NB=16          # plenty for the dispatcher check; keep it quick
CYCLES=1000

cargo build -q --release --features v2 --bin cuda_test --bin cut_map_interactive || {
    echo "FATAL build"; exit 1; }

one() {  # one <design.sv> <top> <tbname> <scope-uut>
    local design=$1 top=$2 tb=$3
    local w="$repo/build/autochk/$top"; rm -rf "$w"; mkdir -p "$w"
    echo
    echo "############################  $top  ############################"
    iverilog -g2012 -s "$tb" -o "$w/stim.vvp" \
        tests/hetero/behavioral_zenith_macros.sv "$design" "tests/hetero/$tb.sv" 2>&1 | tail -3
    ( cd "$w" && vvp stim.vvp "+CYCLES=$CYCLES" "+VCD=stim.vcd" ) >"$w/stim.log" 2>&1
    bash scripts/run_synth_zenith.sh "$design" "$top" "$w/gv.gv" >"$w/yosys.log" 2>&1
    python3 scripts/sanitize_gv.py "$w/gv.gv" >/dev/null 2>&1 || true
    GEM_PARAMS_FILE="$w/params.json" cargo run -q --release --features cuda --bin cut_map_interactive -- \
        "$w/gv.gv" "$w/parts.gemparts" --v2-parts --v2-num-partitions "$NB" >"$w/map.log" 2>&1
    local scope="${tb}/uut"
    run_eng() {  # run_eng <engine>
        local eng=$1
        echo "----- --engine $eng -----"
        GEM_PARAMS_FILE="$w/params.json" target/release/cuda_test \
            "$w/gv.gv" "$w/parts.gemparts" "$w/stim.vcd" "$w/out_$eng.vcd" "$NB" \
            --engine "$eng" --check-with-cpu --input-vcd-scope "$scope" 2>&1 \
            | grep -E 'engine=auto|selected simulation engine|--engine v1:|simulation, Elapsed=|total number of cycles|sanity test passed|panicked|CPU/CUDA mismatch' \
            || echo "  (no matching lines - see full run)"
    }
    run_eng auto
    run_eng v2
    run_eng v1 || echo "  (v1 refused - design needs V2)"
}

one tests/hetero/hetero_farm.sv        hetero_farm        tb_hetero_farm
one tests/hetero/preservation_top.sv   preservation_top   tb_preservation_top
one tests/hetero/bench_mac.sv          bench_mac          tb_bench_mac
# carry_chain8 has no tb/plusargs stimulus wrapper; drive it via the netlist test path instead
echo
echo "############################  carry_chain8 (V2-forced check)  ############################"
w="$repo/build/autochk/carry_chain8"; rm -rf "$w"; mkdir -p "$w"
bash scripts/run_synth_zenith.sh tests/hetero/carry_chain8.sv carry_chain8 "$w/gv.gv" >"$w/yosys.log" 2>&1
python3 scripts/sanitize_gv.py "$w/gv.gv" >/dev/null 2>&1 || true
GEM_PARAMS_FILE="$w/params.json" cargo run -q --release --features cuda --bin cut_map_interactive -- \
    "$w/gv.gv" "$w/parts.gemparts" --v2-parts --v2-num-partitions "$NB" 2>&1 \
    | grep -E 'topological guarantee|macro->macro|Carry4#' | head -12

echo
echo "===================== EXPECTATION ====================="
echo "  hetero_farm      : no same-cycle macro->macro  => auto -> V1 (batched)."
echo "                     --engine v2 also runs; compare the two Elapsed lines."
echo "  preservation_top : same, tiny  => auto -> V1."
echo "  bench_mac        : has the 8-deep CO[3]->CI chain  => auto FORCES V2;"
echo "                     --engine v1 is refused (would read stale state)."
echo "  carry_chain8     : 7 macro->macro edges printed  => auto forces V2."
echo
echo "  Every run above uses --check-with-cpu: any 'sanity test passed' line means"
echo "  that engine matched the CPU reference for this fixture."
echo "done -- send auto_check.log"
