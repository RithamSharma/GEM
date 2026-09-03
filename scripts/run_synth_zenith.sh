#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <design.sv> <top-module> <output.gv>" >&2
    exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
design=$(realpath -- "$1")
top=$2
output=$(realpath -m -- "$3")
template="$repo_dir/synth_zenith.ys"
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

counts=$(python3 - "$design" "$output" <<'PY'
import pathlib
import re
import sys

sources = [pathlib.Path(path).read_text() for path in sys.argv[1:]]
for macro in ("DSP48E2", "CARRY4", "SRLC32E"):
    pattern = re.compile(
        rf"\b{macro}\s*(?:#\s*\(.*?\)\s*)?(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\(",
        re.DOTALL,
    )
    print(macro, *(len(pattern.findall(source)) for source in sources))
PY
)

while read -r macro before after; do
    printf '%-9s input=%s output=%s\n' "$macro" "$before" "$after"
    if (( after < before )); then
        echo "error: synthesis lost $macro instances ($before -> $after)" >&2
        exit 1
    fi
done <<< "$counts"
