#!/usr/bin/env python3
"""Correctness-gated GEM timing and optional Nsight Compute collection."""

import argparse
import json
import pathlib
import platform
import shutil
import statistics
import subprocess
import time


def checked(command, *, stdout=None):
    return subprocess.run(command, check=True, text=True, stdout=stdout)


def simulator_command(args, output):
    command = [
        str(args.binary), str(args.gatelevel), str(args.gemparts),
        str(args.input_vcd), str(output), str(args.num_blocks),
        "--check-with-cpu",
    ]
    if args.input_scope:
        command.extend(["--input-vcd-scope", args.input_scope])
    if args.output_scope:
        command.extend(["--output-vcd-scope", args.output_scope])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", dest="binary", type=pathlib.Path, required=True)
    parser.add_argument("--gatelevel", type=pathlib.Path, required=True)
    parser.add_argument("--gemparts", type=pathlib.Path, required=True)
    parser.add_argument("--input-vcd", type=pathlib.Path, required=True)
    parser.add_argument("--input-scope")
    parser.add_argument("--output-scope")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--report-dir", type=pathlib.Path, default=pathlib.Path("benchmark-results"))
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()

    for path in (args.binary, args.gatelevel, args.gemparts, args.input_vcd):
        if not path.exists():
            parser.error(f"missing required input: {path}")
    if args.repetitions < 3 or args.warmup < 0:
        parser.error("use at least 3 repetitions and a non-negative warmup")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / "gem_output.vcd"
    command = simulator_command(args, output)

    # A mismatch aborts the benchmark instead of publishing invalid timing.
    for _ in range(args.warmup):
        checked(command, stdout=subprocess.DEVNULL)
    samples = []
    for _ in range(args.repetitions):
        start = time.perf_counter_ns()
        checked(command, stdout=subprocess.DEVNULL)
        samples.append((time.perf_counter_ns() - start) / 1e6)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report = {
        "correctness_gate": "passed (--check-with-cpu on every run)",
        "git_revision": revision,
        "host": platform.platform(),
        "command": command,
        "warmup_runs": args.warmup,
        "measured_runs": args.repetitions,
        "wall_time_ms": {
            "samples": samples,
            "median": statistics.median(samples),
            "mean": statistics.mean(samples),
            "stdev": statistics.stdev(samples),
            "min": min(samples),
            "max": max(samples),
        },
    }

    if args.ncu:
        ncu = shutil.which("ncu")
        if not ncu:
            raise SystemExit("--ncu requested but ncu is not installed")
        ncu_report = args.report_dir / "profile"
        checked([
            ncu, "--set", "full", "--force-overwrite", "--export", str(ncu_report),
            *command,
        ])
        report["nsight_compute_report"] = str(ncu_report.with_suffix(".ncu-rep"))

    report_path = args.report_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
