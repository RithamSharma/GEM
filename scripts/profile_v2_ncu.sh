#!/usr/bin/env bash
# ===========================================================================
#  Nsight Compute profile of the CUSTOM V2 kernel (simulate_v2_cycles_kernel).
#
#    scripts/profile_v2_ncu.sh                    # profile the 300-cycle fixture
#    GEM_NCU_GV=build/x/design.gv \
#    GEM_NCU_PARTS=build/x/design.gemparts \
#    GEM_NCU_VCD=build/x/stim.vcd \
#    GEM_NCU_PARAMS=build/x/params.json \
#    GEM_NCU_SCOPE=tb/uut  scripts/profile_v2_ncu.sh   # profile any design
#
#  Output: benchmark-results/part_b_v2_integrated.ncu-rep  (+ .txt summary)
#
#  GPU performance counters normally need elevated privileges:
#      sudo scripts/profile_v2_ncu.sh
#  Without them ncu still runs but some sections show ERR_NVGPUCTRPERM.
# ===========================================================================
set -uo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_dir"

command -v ncu >/dev/null 2>&1 || { echo "ncu (Nsight Compute) not found on PATH"; exit 3; }

GV=${GEM_NCU_GV:-}
PARTS=${GEM_NCU_PARTS:-}
VCD=${GEM_NCU_VCD:-}
PARAMS=${GEM_NCU_PARAMS:-}
SCOPE=${GEM_NCU_SCOPE:-tb_300/uut}
NB=${GEM_NCU_BLOCKS:-14}

if [[ -z "$GV" || -z "$PARTS" || -z "$VCD" ]]; then
    # default: build (if needed) the 300-cycle mixed fixture and profile that
    test_dir=${GEM_V2_NCU_TEST_DIR:-build/test300-ncu}
    if [[ ! -f "$test_dir/gatelevel.gv" || ! -f "$test_dir/result.gemparts" ]]; then
        scripts/run_v2_300_simulation_test.sh "$test_dir"
    fi
    GV="$test_dir/gatelevel.gv"
    PARTS="$test_dir/result.gemparts"
    VCD="$test_dir/oracle.vcd"
    PARAMS="${PARAMS:-$test_dir/params.json}"
    SCOPE="tb_300/uut"
fi
[[ -z "$PARAMS" ]] && PARAMS="$(dirname "$GV")/params.json"

mkdir -p benchmark-results
rep=benchmark-results/part_b_v2_integrated

echo "profiling simulate_v2_cycles_kernel  gv=$GV  blocks=$NB"
GEM_PARAMS_FILE="$PARAMS" ncu --set full \
    --kernel-name regex:simulate_v2_cycles_kernel --launch-count 1 \
    --force-overwrite --export "$rep" \
    target/release/cuda_test \
    "$GV" "$PARTS" "$VCD" "$(dirname "$GV")/ncu-output.vcd" "$NB" \
    --input-vcd-scope "$SCOPE" --engine v2 --check-with-cpu --max-cycles 10

# human-readable extract next to the .ncu-rep
ncu --import "$rep.ncu-rep" --page details 2>/dev/null | tee "$rep.txt" | tail -40 || true
echo
echo "wrote $rep.ncu-rep  and  $rep.txt"
