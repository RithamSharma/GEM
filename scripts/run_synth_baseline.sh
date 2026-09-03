#!/usr/bin/env bash
# Baseline synthesis wrapper: SHRED the macros to gates (unmodified-GEM
# behaviour). Mirror of scripts/run_synth_zenith.sh but with synth_baseline.ys
# and no "macros must survive" check (they must NOT survive here).
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <design.sv> <top-module> <output.gv>" >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
design=$(realpath -- "$1")
top=$2
output=$(realpath -m -- "$3")
template="$repo_dir/synth_baseline.ys"
generated=$(mktemp --suffix=.ys)
trap 'rm -f -- "$generated"' EXIT

case "$design$output$top" in
    *$'\n'*|*'"'*|*'\\'*)
        echo "error: paths/top containing quotes, backslashes, or newlines are unsupported" >&2
        exit 2
        ;;
esac

python3 - "$template" "$generated" "$design" "$top" "$output" <<'PY'
import os, pathlib, sys

template, generated, design, top, output = sys.argv[1:]
text = pathlib.Path(template).read_text()
# GEM_SYNTH_READ overrides the design read command (e.g. "read_slang" if the
# yosys-slang plugin is installed and a hidden benchmark needs it). Default is
# Yosys 0.68's built-in IEEE-1800-2012 frontend.
text = text.replace("@@GEM_READ@@", os.environ.get("GEM_SYNTH_READ", "read_verilog -sv"))
text = text.replace("@@GEM_DESIGN@@", f'"{design}"')
text = text.replace("@@GEM_TOP@@", top)
text = text.replace("@@GEM_OUT@@", f'"{output}"')
if "@@GEM_" in text:
    raise SystemExit("unexpanded synthesis template token")
pathlib.Path(generated).write_text(text)
PY

mkdir -p -- "$(dirname -- "$output")"
cd -- "$repo_dir"
yosys -s "$generated"
[[ -s "$output" ]] || { echo "error: baseline synthesis produced no netlist" >&2; exit 1; }

# Confirm the macros really were shredded (the baseline's whole purpose).
if grep -qE '\b(DSP48E2|CARRY4|SRLC32E)\b' "$output"; then
    echo "warning: baseline netlist still contains a macro instance name" >&2
fi
