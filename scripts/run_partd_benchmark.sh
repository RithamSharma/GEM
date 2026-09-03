#!/usr/bin/env bash
# ===========================================================================
#  Part D  --  simulated cycles/second and AIG graph size:
#              macros PRESERVED (heterogeneous V2 engine)
#           vs macros SHREDDED  (ordinary GEM AIG, V1 engine)
#              on identical RTL and identical stimulus.
#
#    ./scripts/run_partd_benchmark.sh                          # bench_mac, 4000 cycles
#    ./scripts/run_partd_benchmark.sh <design.sv> <top> [cycles] [num_blocks]
#
#  Stimulus: by default a bundled testbench `tests/hetero/tb_<top>.sv` drives it.
#  For a design with no such testbench (a hidden benchmark), point
#  GEM_PARTD_STIM at a .vcd or a numeric .csv/.txt stimulus table and its scope:
#    GEM_PARTD_STIM=path/to/stim.csv  GEM_PARTD_SCOPE=tb/uut \
#      ./scripts/run_partd_benchmark.sh design.v top 4000
#
#  Requires: yosys, iverilog, python3, CUDA toolkit + NVIDIA GPU.
#  Output:   benchmark-results/partd_summary.txt
#            benchmark-results/partd_preserved.json
#            benchmark-results/partd_shredded.json
#
#  Correctness: the preserved flow is run once WITH --check-with-cpu before any
#  timing; timing reps then run without the CPU gate so the number is CUDA-only.
# ===========================================================================
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
chmod +x scripts/*.sh 2>/dev/null || true

DESIGN=${1:-tests/hetero/bench_mac.sv}
TOP=${2:-bench_mac}
CYCLES=${3:-2000}
NB=${4:-0}
WARMUP=${WARMUP:-2}
REPS=${REPS:-5}
if [[ "$NB" == 0 ]]; then
    SM=$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader -i 0 2>/dev/null | head -1 || echo 8)
    [[ "$SM" =~ ^[0-9]+$ ]] || SM=8
    # one cooperative block per SM is always within the occupancy limit
    # (blocks <= SM * maxActiveBlocksPerSM, and that factor is >= 1). Cap at 64
    # as a sanity ceiling; override with arg 4.
    NB=$(( SM < 64 ? SM : 64 ))
fi
[[ "$NB" =~ ^[0-9]+$ && "$NB" -ge 1 ]] || NB=1

out="$repo/benchmark-results"; mkdir -p "$out"
work="$repo/build/partd"; rm -rf "$work"; mkdir -p "$work"

echo "design=$DESIGN  top=$TOP  cycles=$CYCLES  num_blocks=$NB  (warmup=$WARMUP reps=$REPS)"

# ---- 0. one stimulus VCD, shared by both flows ------------------------------
if [[ -n "${GEM_PARTD_STIM:-}" ]]; then
    SCOPE="${GEM_PARTD_SCOPE:-tb/uut}"
    case "$GEM_PARTD_STIM" in
        *.vcd) echo "[0] using stimulus VCD $GEM_PARTD_STIM"; cp "$GEM_PARTD_STIM" "$work/stim.vcd" ;;
        *)     echo "[0] converting numeric stimulus $GEM_PARTD_STIM -> VCD"
               python3 scripts/stim_to_vcd.py --stim "$GEM_PARTD_STIM" --out "$work/stim.vcd" --scope "$SCOPE" ;;
    esac
    [[ -s "$work/stim.vcd" ]] || { echo "FATAL: could not prepare stimulus from GEM_PARTD_STIM"; exit 1; }
else
    echo "[0] generating ${CYCLES}-cycle stimulus VCD from tb_${TOP}"
    [[ -f "tests/hetero/tb_${TOP}.sv" ]] || {
        echo "FATAL: no tests/hetero/tb_${TOP}.sv — set GEM_PARTD_STIM=<stim.vcd|csv> for a design without a bundled testbench"; exit 1; }
    iverilog -g2012 -s "tb_${TOP}" -o "$work/stim.vvp" \
        tests/hetero/behavioral_zenith_macros.sv "$DESIGN" "tests/hetero/tb_${TOP}.sv"
    # run from $work with a RELATIVE vcd name: a long absolute path can be truncated
    # by the testbench's fixed-width %s string register.
    ( cd "$work" && vvp stim.vvp "+CYCLES=$CYCLES" "+VCD=stim.vcd" ) >"$work/stim.log" 2>&1
    [[ -s "$work/stim.vcd" ]] || { echo "FATAL: stimulus VCD not produced (see $work/stim.log)"; exit 1; }
    SCOPE="tb_${TOP}/uut"
fi

cargo build -q --release --features v2 --bin cuda_test --bin cut_map_interactive || {
    echo "FATAL: cargo build failed"; exit 1; }

# ---- timing helper --------------------------------------------------------
time_flow() {   # time_flow <label> <gatelevel.gv> <gemparts> <params.json> <v2flag>
    local label=$1 gv=$2 parts=$3 params=$4 v2=$5
    local extra=(); [[ "$v2" == v2 ]] && extra=(--v2)
    python3 - "$label" "$gv" "$parts" "$params" "$work/${label}.vcd" \
              "$SCOPE" "$NB" "$WARMUP" "$REPS" "$out/partd_${label}.json" "${extra[@]}" <<'PY'
import json, os, re, statistics, subprocess, sys
label, gv, parts, params, ovcd, scope, nb, warmup, reps, outjson, *extra = sys.argv[1:]
warmup, reps, nb = int(warmup), int(reps), int(nb)
bin_ = "target/release/cuda_test"
env = os.environ.copy(); env["GEM_PARAMS_FILE"] = params
base = [bin_, gv, parts, os.path.join(os.path.dirname(ovcd), "stim.vcd"),
        ovcd, str(nb), "--input-vcd-scope", scope] + extra
E = re.compile(r"simulation, Elapsed=([0-9.]+)ms")
C = re.compile(r"total number of cycles: (\d+)")
def run(cmd):
    r = subprocess.run(cmd, env=env, text=True, capture_output=True)
    return r.stdout + r.stderr, r.returncode
# one correctness-gated pass before timing (both engines have a CPU gate)
combined, rc = run(base + ["--check-with-cpu"])
gate = ("V2 CPU sanity test passed!" in combined) or ("sanity test passed!" in combined)
if rc != 0 or not gate:
    tail = "\n".join(combined.splitlines()[-40:])
    json.dump({"label": label, "ok": False, "reason": "correctness gate failed",
               "returncode": rc, "tail": tail}, open(outjson, "w"), indent=2)
    print(f"  {label}: CORRECTNESS GATE FAILED (rc={rc})"); print(tail); sys.exit(3)
cm = C.search(combined)
if cm is None:
    tail = "\n".join(combined.splitlines()[-40:])
    json.dump({"label": label, "ok": False, "reason": "no 'total number of cycles' line",
               "returncode": rc, "tail": tail}, open(outjson, "w"), indent=2)
    print(f"  {label}: simulator produced no cycle-count line"); print(tail); sys.exit(6)
cyc = int(cm.group(1))
samples = []
for i in range(warmup + reps):
    combined, rc = run(base)          # timing reps: no CPU gate
    if rc != 0:
        print(f"  {label}: timing run {i} failed rc={rc}")
        print("\n".join(combined.splitlines()[-30:])); sys.exit(4)
    m = E.search(combined)
    if not m:
        print(f"  {label}: no 'Elapsed=' line in run {i}"); sys.exit(5)
    if i >= warmup:
        ms = float(m.group(1))
        samples.append({"elapsed_ms": ms, "cycles_per_second": cyc * 1000.0 / ms})
ms = [s["elapsed_ms"] for s in samples]
cps = [s["cycles_per_second"] for s in samples]
rep = {"label": label, "ok": True, "engine": "V2 heterogeneous" if extra else "V1 AIG",
       "simulated_cycles": cyc, "cuda_blocks": nb, "warmup": warmup, "reps": reps,
       "samples": samples,
       "elapsed_ms_median": statistics.median(ms),
       "cycles_per_second_median": statistics.median(cps),
       "cycles_per_second_stdev": statistics.pstdev(cps),
       "command": base}
json.dump(rep, open(outjson, "w"), indent=2)
print(f"  {label}: {statistics.median(cps):,.0f} cycles/s  "
      f"(median {statistics.median(ms):.2f} ms over {reps} reps, {cyc} cycles)")
PY
}

graph_size() {  # graph_size <map.log> <field>   field: "aig pins" | "and gates"
    local n
    n=$(grep -oE "[0-9]+ $2" "$1" 2>/dev/null | tail -1 | grep -oE '^[0-9]+')
    echo "${n:-?}"
}

# ---- Flow A : macros preserved, V2 engine --------------------------------
echo
echo "=================  PRESERVED (heterogeneous V2)  ================="
PA_OK=1
bash scripts/run_synth_zenith.sh "$DESIGN" "$TOP" "$work/preserved.gv" >"$work/preserved.yosys.log" 2>&1 \
    || { echo "FATAL: preserved synthesis failed (see $work/preserved.yosys.log)"; PA_OK=0; }
if [[ $PA_OK == 1 ]]; then
    python3 scripts/sanitize_gv.py "$work/preserved.gv" >/dev/null 2>&1 || true
    cp "$work/params.json" "$work/preserved.params.json" 2>/dev/null || \
        echo '{}' >"$work/preserved.params.json"
    GEM_PARAMS_FILE="$work/preserved.params.json" \
        cargo run -q --release --features cuda --bin cut_map_interactive -- \
        "$work/preserved.gv" "$work/preserved.gemparts" \
        --v2-parts --v2-num-partitions "$NB" >"$work/preserved.map.log" 2>&1 \
        || { echo "FATAL: preserved cut_map failed (see $work/preserved.map.log)"; PA_OK=0; }
fi
[[ $PA_OK == 1 ]] && { time_flow preserved "$work/preserved.gv" "$work/preserved.gemparts" \
    "$work/preserved.params.json" v2 || PA_OK=0; }
PA_PINS=$(graph_size "$work/preserved.map.log" "aig pins")
PA_ANDS=$(graph_size "$work/preserved.map.log" "and gates")

# ---- Flow B : macros shredded to gates, V1 engine -----------------------
echo
echo "=================  SHREDDED (ordinary GEM AIG, V1)  ================="
PB_OK=1
bash scripts/run_synth_baseline.sh "$DESIGN" "$TOP" "$work/shredded.gv" >"$work/shredded.yosys.log" 2>&1 \
    || { echo "FATAL: baseline synthesis failed (see $work/shredded.yosys.log)"; PB_OK=0; }
if [[ $PB_OK == 1 ]]; then
    python3 scripts/sanitize_gv.py "$work/shredded.gv" >/dev/null 2>&1 || true
    cp "$work/params.json" "$work/shredded.params.json" 2>/dev/null || \
        echo '{}' >"$work/shredded.params.json"
    GEM_PARAMS_FILE="$work/shredded.params.json" \
        cargo run -q --release --features cuda --bin cut_map_interactive -- \
        "$work/shredded.gv" "$work/shredded.gemparts" >"$work/shredded.map.log" 2>&1 \
        || { echo "FATAL: shredded cut_map failed (see $work/shredded.map.log)"; PB_OK=0; }
fi
[[ $PB_OK == 1 ]] && { time_flow shredded "$work/shredded.gv" "$work/shredded.gemparts" \
    "$work/shredded.params.json" v1 || PB_OK=0; }
PB_PINS=$(graph_size "$work/shredded.map.log" "aig pins")
PB_ANDS=$(graph_size "$work/shredded.map.log" "and gates")

# ---- summary -----------------------------------------------------------
{
  echo "Part D  --  $DESIGN ($TOP), $CYCLES cycles, num_blocks=$NB"
  echo "host: $(uname -sr)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  get() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d.get('cycles_per_second_median',0):,.0f}\" if d.get('ok') else 'FAILED')" "$1" 2>/dev/null || echo "n/a"; }
  PCPS=$(get "$out/partd_preserved.json")
  SCPS=$(get "$out/partd_shredded.json")
  printf '%-24s %16s %14s %12s\n' "flow" "cycles/sec (med)" "AIG pins" "AND gates"
  printf '%s\n' "----------------------------------------------------------------------"
  printf '%-24s %16s %14s %12s\n' "preserved (V2 hetero)" "$PCPS" "$PA_PINS" "$PA_ANDS"
  printf '%-24s %16s %14s %12s\n' "shredded  (V1 AIG)"    "$SCPS" "$PB_PINS" "$PB_ANDS"
  echo
  python3 - "$out/partd_preserved.json" "$out/partd_shredded.json" \
           "$PA_PINS" "$PB_PINS" "$PA_ANDS" "$PB_ANDS" <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
    if p.get("ok") and s.get("ok"):
        a = p["cycles_per_second_median"]; b = s["cycles_per_second_median"]
        if b > 0:
            tag = "preserved FASTER" if a > b else "preserved slower"
            print(f"throughput ratio (preserved / shredded): {a/b:.2f}x  [{tag}]")
except Exception as e:
    print(f"(throughput ratio unavailable: {e})")
def ratio(x, y, label):
    if x.isdigit() and y.isdigit() and int(x) > 0:
        print(f"{label}: {int(y)/int(x):.1f}x")
ratio(sys.argv[3], sys.argv[4], "AIG-pin reduction  (shredded / preserved)")
ratio(sys.argv[5], sys.argv[6], "AND-gate reduction (shredded / preserved)")
PY
  echo
  echo "per-flow JSON: benchmark-results/partd_preserved.json, benchmark-results/partd_shredded.json"
  echo "logs:          build/partd/*.log"
} | tee "$out/partd_summary.txt"

# non-zero exit only if BOTH flows failed (partial result is still useful)
[[ $PA_OK == 1 || $PB_OK == 1 ]]
