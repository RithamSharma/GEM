#!/usr/bin/env python3
"""Host-Side Macro Memory Formatter plan, Phase 7.

Drives `formatter_gpu_test` under Nsight Compute and Nsight Systems and checks
the V2 selector layout against the theoretical coalesced-transfer minimum.

For one contiguous B-byte value per lane in a full 32-thread warp the ideal
number of 32-byte sectors requested is  ideal = 32 * B / 32 = B.  The V2
selector words are u64 (B = 8), so a fully coalesced warp load of one
(field, bit) row should request ~8 sectors.  Instance-AoS packing would request
up to 32.

Nothing here fabricates numbers: it only runs the tools and reports what they
say.  Requires a working NVIDIA GPU + `ncu` (and optionally `nsys`).
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def ncu_metrics(binary, out_prefix):
    """Collect global-load sector metrics for the formatter kernels."""
    metrics = ",".join(
        [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "gpu__time_duration.sum",
        ]
    )
    csv = run(
        [
            "ncu", "--csv", "--target-processes", "all",
            "-k", "regex:formatter_",
            "--metrics", metrics,
            str(binary),
        ]
    ).stdout
    (pathlib.Path(out_prefix + ".ncu.csv")).write_text(csv)
    # also keep the full report for the submission
    run(
        [
            "ncu", "--set", "full", "--target-processes", "all",
            "-k", "regex:formatter_", "-f", "-o", out_prefix + "_full",
            str(binary),
        ]
    )
    return csv


def parse_sectors_per_request(csv_text):
    rows = [r for r in csv_text.splitlines() if r and not r.startswith('"ID"')]
    sectors = requests = None
    for r in rows:
        cells = [c.strip('"') for c in r.split('","')]
        joined = " ".join(cells)
        if "sectors_pipe_lsu_mem_global_op_ld.sum" in joined:
            sectors = _last_number(cells)
        if "requests_pipe_lsu_mem_global_op_ld.sum" in joined:
            requests = _last_number(cells)
    if sectors and requests:
        return sectors / requests
    return None


def _last_number(cells):
    for c in reversed(cells):
        c = c.replace(",", "")
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", c):
            return float(c)
    return None


def nsys_transfers(binary, out_prefix):
    """One immutable program upload, none inside a loop (there is no cycle loop
    in the self-check, but this is the check the real V2 kernel must also pass)."""
    try:
        run(["nsys", "profile", "-o", out_prefix + "_nsys", "-f", "true",
             "--stats", "true", str(binary)])
        stats = run(["nsys", "stats", "--report", "gpumemtimesum",
                     out_prefix + "_nsys.nsys-rep"]).stdout
        (pathlib.Path(out_prefix + ".nsys.txt")).write_text(stats)
        return stats
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"nsys unavailable: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", dest="binary", type=pathlib.Path,
                    default=pathlib.Path("target/release/formatter_gpu_test"))
    ap.add_argument("--out", type=str, default="benchmark-results/formatter_v2")
    args = ap.parse_args()
    if not args.binary.exists():
        ap.error(f"build it first: cargo build --release --features v2 --bin "
                 f"formatter_gpu_test  (missing {args.binary})")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1. correctness gate: the binary must exit 0 (all self-check flags set).
    proc = subprocess.run([str(args.binary)], text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0:
        sys.exit(f"formatter_gpu_test self-check FAILED; not collecting metrics.\n{proc.stderr}")

    csv = ncu_metrics(args.binary, args.out)
    spr = parse_sectors_per_request(csv)
    nsys = nsys_transfers(args.binary, args.out)

    summary = {
        "binary": str(args.binary),
        "self_check": "pass",
        "sectors_per_global_load_request": spr,
        "ideal_sectors_for_u64_per_lane": 8,
        "verdict": (
            "at/near ideal" if spr is not None and spr <= 10
            else "investigate stride" if spr is not None
            else "ncu metric names vary by arch; resolve with `ncu --query-metrics`"
        ),
        "nsys_gpu_mem_time_summary": nsys.splitlines()[-20:] if isinstance(nsys, str) else nsys,
    }
    out_json = pathlib.Path(args.out + ".summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nartifacts: {args.out}.ncu.csv  {args.out}_full.ncu-rep  {out_json}")


if __name__ == "__main__":
    main()
