#!/usr/bin/env bash
# ===========================================================================
#  Part D sweep -- the wide-parallel farm at a few block counts, plus the
#  serial-heavy bench_mac for contrast. Everything to partd_sweep.log.
#
#    ./scripts/partd_sweep.sh                 # hetero_farm @ 1,4,8 blocks + bench_mac @ 1
#    ./scripts/partd_sweep.sh "1 8 14"        # custom block-count list for the farm
#
#  The release build is assumed already done (run partd_evidence.sh first, or
#  cargo build --release --features v2). Each configuration is ~3-6 min.
#  Send back: partd_sweep.log
# ===========================================================================
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
chmod +x scripts/*.sh 2>/dev/null || true

BLOCKS=${1:-"1 4 8"}
CYCLES=${2:-2000}
LOG="$repo/partd_sweep.log"
: > "$LOG"
arch="$repo/benchmark-results/partd-sweep"; mkdir -p "$arch"

run() {  # run <design.sv> <top> <nb> <tag>
    echo | tee -a "$LOG"
    echo "############### $4  (nb=$3, cycles=$CYCLES) ###############" | tee -a "$LOG"
    bash scripts/run_partd_benchmark.sh "$1" "$2" "$CYCLES" "$3" 2>&1 | tee -a "$LOG"
    # archive this configuration's JSON so later configs don't overwrite it
    for f in preserved shredded; do
        [[ -f benchmark-results/partd_$f.json ]] && \
            cp benchmark-results/partd_$f.json "$arch/${4}_nb${3}_${f}.json"
    done
    cp benchmark-results/partd_summary.txt "$arch/${4}_nb${3}_summary.txt" 2>/dev/null || true
}

for nb in $BLOCKS; do
    run tests/hetero/hetero_farm.sv hetero_farm "$nb" "farm"
done
run tests/hetero/bench_mac.sv bench_mac 1 "benchmac"

echo | tee -a "$LOG"
echo "===================== SWEEP SUMMARY =====================" | tee -a "$LOG"
python3 - "$arch" <<'PY' | tee -a "$LOG"
import glob, json, os, sys, re
arch = sys.argv[1]
rows = []
for sf in sorted(glob.glob(os.path.join(arch, "*_preserved.json"))):
    base = os.path.basename(sf)[:-len("_preserved.json")]
    def load(p):
        try: return json.load(open(p))
        except Exception: return {}
    pj = load(sf); sj = load(sf.replace("_preserved.json", "_shredded.json"))
    m = re.match(r"(\w+)_nb(\d+)", base)
    design, nb = (m.group(1), m.group(2)) if m else (base, "?")
    p = pj.get("cycles_per_second_median"); s = sj.get("cycles_per_second_median")
    ratio = f"{p/s:.2f}x" if (p and s) else "-"
    rows.append((design, nb,
                 f"{p:,.0f}" if p else ("FAIL" if not pj.get("ok") else "-"),
                 f"{s:,.0f}" if s else ("FAIL" if not sj.get("ok") else "-"),
                 ratio))
print(f"{'design':<10} {'blocks':>6} {'preserved c/s':>15} {'shredded c/s':>15} {'ratio':>8}")
print("-" * 60)
for r in rows:
    print(f"{r[0]:<10} {r[1]:>6} {r[2]:>15} {r[3]:>15} {r[4]:>8}")
print()
print("ratio > 1.00x  => preserving macros is faster than shredding to AIG")
PY

echo | tee -a "$LOG"
echo "archived per-config JSON: benchmark-results/partd-sweep/" | tee -a "$LOG"
echo "done -- send partd_sweep.log" | tee -a "$LOG"
