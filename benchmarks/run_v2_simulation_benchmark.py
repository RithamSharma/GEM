#!/usr/bin/env python3
"""Correctness-gated whole-simulation throughput benchmark for CUDA V2."""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess


ELAPSED = re.compile(r"simulation, Elapsed=([0-9.]+)ms")
CYCLES = re.compile(r"total number of cycles: (\d+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=pathlib.Path, required=True)
    parser.add_argument("--gatelevel", type=pathlib.Path, required=True)
    parser.add_argument("--gemparts", type=pathlib.Path, required=True)
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--input-vcd", type=pathlib.Path, required=True)
    parser.add_argument("--input-scope", required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmup < 0:
        parser.error("use at least three repetitions and a non-negative warmup")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_vcd = args.out.with_suffix(".vcd")
    command = [
        str(args.bin), str(args.gatelevel), str(args.gemparts),
        str(args.input_vcd), str(output_vcd), str(args.num_blocks),
        "--input-vcd-scope", args.input_scope, "--check-with-cpu", "--v2",
    ]
    env = os.environ.copy()
    env["GEM_PARAMS_FILE"] = str(args.params)

    samples = []
    cycles = None
    for run_index in range(args.warmup + args.repetitions):
        run = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        combined = run.stdout + run.stderr
        if "V2 CPU sanity test passed!" not in combined:
            raise SystemExit("correctness gate did not pass")
        elapsed = ELAPSED.search(combined)
        cycle_match = CYCLES.search(combined)
        if elapsed is None or cycle_match is None:
            raise SystemExit("simulator timing/cycle line missing")
        cycles = int(cycle_match.group(1))
        if run_index >= args.warmup:
            milliseconds = float(elapsed.group(1))
            samples.append({
                "elapsed_ms": milliseconds,
                "simulation_cycles_per_second": cycles * 1000.0 / milliseconds,
            })

    rates = [sample["simulation_cycles_per_second"] for sample in samples]
    times = [sample["elapsed_ms"] for sample in samples]
    report = {
        "correctness_gate": "CPU V2 == CUDA V2 on every measured run",
        "simulation_cycles": cycles,
        "cuda_blocks": args.num_blocks,
        "warmup_runs": args.warmup,
        "measured_runs": args.repetitions,
        "samples": samples,
        "elapsed_ms": {
            "median": statistics.median(times),
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times),
            "min": min(times),
            "max": max(times),
        },
        "simulation_cycles_per_second": {
            "median": statistics.median(rates),
            "mean": statistics.mean(rates),
            "stdev": statistics.stdev(rates),
            "min": min(rates),
            "max": max(rates),
        },
        "command": command,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
