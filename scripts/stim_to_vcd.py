#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a numeric stimulus table into a VCD that GEM's simulators can read.

The judging panel supplies a design plus a "database file with numbers for each
signal per simulation step". GEM reads stimulus as VCD, so this bridges the two.

Accepted stimulus formats (auto-detected, override with --format):

  csv/tsv/wsv : one row per cycle. Columns are either
                - named by a header row:  clk,a,b,...   or   a[26:0],b[17:0],...
                - or positional, matched to --ports in order.
                Values are decimal (default) or hex (--radix hex, or 0x-prefixed).
  vcd         : already a VCD -> copied through unchanged.

Clocking:
  If a signal named by --clk (default "clk") is NOT among the columns, a clock is
  synthesised: it is 0 while inputs settle, then rises once per row. If it IS a
  column, its values are used verbatim and no edges are added.

Example
  python3 scripts/stim_to_vcd.py --stim stim.csv --out stim.vcd \\
      --ports clk:1,a:27,b:18 --radix hex --scope tb/uut
"""
import argparse
import csv
import io
import re
import sys
from pathlib import Path


def parse_ports(spec):
    """'clk:1,a:27,b[17:0]' -> [('clk',1),('a',27),('b',18)]"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\w+)\s*(?::\s*(\d+)|\[\s*(\d+)\s*:\s*(\d+)\s*\])?", tok)
        if not m:
            sys.exit(f"stim_to_vcd: cannot parse port spec token {tok!r}")
        name = m.group(1)
        if m.group(2):
            w = int(m.group(2))
        elif m.group(3) is not None:
            w = abs(int(m.group(3)) - int(m.group(4))) + 1
        else:
            w = 1
        out.append((name, w))
    return out


def header_to_ports(cols):
    return parse_ports(",".join(cols))


def sniff_delim(sample):
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return None  # whitespace


def read_rows(path, delim):
    text = Path(path).read_text().splitlines()
    rows = []
    for ln in text:
        ln = ln.strip()
        if not ln or ln.startswith(("#", "//")):
            continue
        if delim is None:
            rows.append(ln.split())
        else:
            rows.append(next(csv.reader([ln], delimiter=delim)))
    return rows


def to_int(tok, radix):
    tok = tok.strip()
    if tok in ("", "x", "X", "z", "Z", "-"):
        return None
    if tok.lower().startswith("0x"):
        return int(tok, 16)
    if tok.lower().startswith("0b"):
        return int(tok, 2)
    return int(tok, 16 if radix == "hex" else 10)


def bits(val, width):
    if val is None:
        return "x" * width
    val &= (1 << width) - 1
    return format(val, "0{}b".format(width))


def ident(i):
    # printable VCD identifier codes: 33..126
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 94)
        s = chr(33 + r) + s
    return s


def emit_vcd(ports, rows, clk_name, scope, out):
    codes = {name: ident(k) for k, (name, _) in enumerate(ports)}
    widths = {name: w for name, w in ports}
    synth_clk = clk_name not in widths

    w = io.StringIO()
    w.write("$timescale 1ns $end\n")
    for s in scope.split("/"):
        if s:
            w.write(f"$scope module {s} $end\n")
    for name, wd in ports:
        w.write(f"$var wire {wd} {codes[name]} {name} $end\n")
    if synth_clk:
        codes[clk_name] = ident(len(ports))
        w.write(f"$var wire 1 {codes[clk_name]} {clk_name} $end\n")
    for s in scope.split("/"):
        if s:
            w.write("$upscope $end\n")
    w.write("$enddefinitions $end\n")

    def dump(name, valbits):
        if len(valbits) == 1:
            w.write(f"{valbits}{codes[name]}\n")
        else:
            w.write(f"b{valbits} {codes[name]}\n")

    t = 0
    w.write(f"#{t}\n$dumpvars\n")
    prev = {}
    for r, row in enumerate(rows):
        # settle inputs at an even time; pulse the synthetic clock high 5ns later.
        vals = {}
        for (name, wd), tok in zip(ports, row):
            vals[name] = bits(to_int(tok, args.radix), wd)
        w.write(f"#{t}\n")
        for name, _ in ports:
            if vals[name] != prev.get(name):
                dump(name, vals[name])
                prev[name] = vals[name]
        if synth_clk:
            if prev.get(clk_name) != "0":
                dump(clk_name, "0")
                prev[clk_name] = "0"
            w.write(f"#{t + 5}\n")
            dump(clk_name, "1")
            prev[clk_name] = "1"
            t += 10
        else:
            t += 10
    w.write(f"#{t}\n")
    out.write(w.getvalue())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stim", required=True, help="numeric stimulus file (or a .vcd to pass through)")
    ap.add_argument("--out", required=True, help="output VCD path")
    ap.add_argument("--ports", help="positional column spec, e.g. 'clk:1,a:27,b:18' (omit if the file has a header)")
    ap.add_argument("--format", choices=["auto", "csv", "tsv", "wsv", "vcd"], default="auto")
    ap.add_argument("--radix", choices=["dec", "hex"], default="dec")
    ap.add_argument("--clk", default="clk", help="clock signal name (synthesised if not a column)")
    ap.add_argument("--scope", default="tb/uut", help="VCD scope path for --input-vcd-scope")
    ap.add_argument("--has-header", choices=["auto", "yes", "no"], default="auto")
    args = ap.parse_args()

    raw = Path(args.stim).read_text()
    fmt = args.format
    if fmt == "auto":
        fmt = "vcd" if raw.lstrip().startswith(("$", "#")) and "$var" in raw else "csv"

    if fmt == "vcd":
        Path(args.out).write_text(raw)
        print(f"stim_to_vcd: {args.stim} is already a VCD -> copied to {args.out}")
        sys.exit(0)

    first = next((l for l in raw.splitlines() if l.strip() and not l.strip().startswith(("#", "//"))), "")
    if args.format == "auto":
        delim = sniff_delim(first)
    else:
        delim = {"csv": ",", "tsv": "\t", "wsv": None}[args.format]
    rows = read_rows(args.stim, delim)
    if not rows:
        sys.exit("stim_to_vcd: no data rows")

    has_header = args.has_header
    if has_header == "auto":
        has_header = "yes" if any(re.search(r"[A-Za-z_]", c) for c in rows[0]) and not args.ports else "no"

    if has_header == "yes":
        ports = header_to_ports(rows[0])
        rows = rows[1:]
    elif args.ports:
        ports = parse_ports(args.ports)
    else:
        sys.exit("stim_to_vcd: need either a header row or --ports")

    for r in rows:
        if len(r) != len(ports):
            sys.exit(f"stim_to_vcd: row has {len(r)} columns, expected {len(ports)}: {r}")

    with open(args.out, "w") as fh:
        emit_vcd(ports, rows, args.clk, args.scope, fh)
    print(f"stim_to_vcd: {len(rows)} cycles, {len(ports)} signals -> {args.out}  (scope {args.scope})")
