#!/usr/bin/env python3
"""Correctness-gated repeated timing for the V2 dependency-wave CUDA path."""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess


LINE = re.compile(
    r"benchmark: repetitions=(?P<reps>\d+) elapsed_ms=(?P<ms>[0-9.]+) "
    r"kernel_executions_per_s=(?P<exec>[0-9.]+) operation_evaluations_per_s=(?P<operation>[0-9.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=pathlib.Path, default=pathlib.Path("target/release/formatter_gpu_test"))
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--kernel-repetitions", type=int, default=10_000)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("benchmark-results/v2_runtime.json"))
    args = parser.parse_args()
    if args.samples < 3 or args.kernel_repetitions < 1:
        parser.error("use at least three samples and one kernel repetition")

    env = os.environ.copy()
    env["GEM_V2_BENCH_REPS"] = str(args.kernel_repetitions)
    samples = []
    for _ in range(args.samples):
        run = subprocess.run([str(args.bin)], text=True, capture_output=True, check=True, env=env)
        if "PASS: formatter and dependency-wave CUDA execution match CPU V2" not in run.stdout:
            raise SystemExit("correctness gate did not pass")
        match = LINE.search(run.stdout)
        if not match:
            raise SystemExit("benchmark line missing")
        samples.append({key: float(value) for key, value in match.groupdict().items()})

    exec_rates = [x["exec"] for x in samples]
    operation_rates = [x["operation"] for x in samples]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], text=True, capture_output=True, check=True
        ).stdout.strip()
    )
    report = {
        "correctness_gate": "CPU V2 == CUDA V2 on every sample",
        "git_revision": revision,
        "working_tree_dirty": dirty,
        "gpu": gpu,
        "samples": samples,
        "kernel_executions_per_second": {
            "median": statistics.median(exec_rates),
            "mean": statistics.mean(exec_rates),
            "stdev": statistics.stdev(exec_rates),
            "min": min(exec_rates),
            "max": max(exec_rates),
        },
        "operation_evaluations_per_second": {
            "median": statistics.median(operation_rates),
            "mean": statistics.mean(operation_rates),
            "stdev": statistics.stdev(operation_rates),
            "min": min(operation_rates),
            "max": max(operation_rates),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
