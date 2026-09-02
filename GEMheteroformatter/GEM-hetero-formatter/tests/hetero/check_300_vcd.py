#!/usr/bin/env python3
"""Compare GEM's scalarized output VCD against the 300-cycle HDL oracle."""

import csv
import re
import sys
from pathlib import Path


OUTPUT_WIDTHS = {"p": 48, "o": 4, "co": 4, "q": 1, "q31": 1}


def read_expected(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 300:
        raise AssertionError(f"expected 300 oracle rows, found {len(rows)}")
    return rows


def read_actual(path: Path, sample_times):
    declarations = {}
    values = {}
    snapshots = {}
    current_time = None

    def finish_timestamp():
        if current_time in sample_times:
            assembled = {}
            for signal, width in OUTPUT_WIDTHS.items():
                value = 0
                for bit in range(width):
                    key = (signal, bit)
                    if key not in values:
                        raise AssertionError(
                            f"missing {signal}[{bit}] at timestamp {current_time}"
                        )
                    value |= values[key] << bit
                assembled[signal] = value
            snapshots[current_time] = assembled

    with path.open() as stream:
        for raw_line in stream:
            line = raw_line.strip()
            match = re.match(r"\$var\s+\S+\s+1\s+(\S+)\s+(\S+)\s+\$end", line)
            if match:
                identifier, name = match.groups()
                bit_match = re.fullmatch(r"(p|o|co)\[(\d+)\]", name)
                if bit_match:
                    declarations[identifier] = (bit_match.group(1), int(bit_match.group(2)))
                elif name in ("q", "q31"):
                    declarations[identifier] = (name, 0)
                continue
            if line.startswith("#"):
                finish_timestamp()
                current_time = int(line[1:])
                continue
            if len(line) >= 2 and line[0] in "01xz" and line[1:] in declarations:
                if line[0] not in "01":
                    raise AssertionError(
                        f"GEM produced {line[0]} for {declarations[line[1:]]} at {current_time}"
                    )
                values[declarations[line[1:]]] = int(line[0])
    finish_timestamp()
    return snapshots


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_300_vcd.py <expected.csv> <gem-output.vcd>")
    expected = read_expected(Path(sys.argv[1]))
    # flatten_test intentionally labels a propagated result with the previous
    # active edge (see its two-timestamp pipeline). Inputs applied on the
    # preceding negedge and captured at oracle time T therefore appear in the
    # GEM output VCD at T-10 ns for this testbench.
    output_time = lambda row: int(row["time"]) - 10_000
    sample_times = {output_time(row) for row in expected}
    actual = read_actual(Path(sys.argv[2]), sample_times)

    failures = []
    for row in expected:
        timestamp = output_time(row)
        got = actual.get(timestamp)
        if got is None:
            failures.append(f"cycle {row['cycle']}: missing GEM timestamp {timestamp}")
            continue
        want = {
            "p": int(row["p"], 16),
            "o": int(row["o"], 16),
            "co": int(row["co"], 16),
            "q": int(row["q"]),
            "q31": int(row["q31"]),
        }
        if got != want:
            failures.append(
                f"cycle {row['cycle']} time {timestamp}: expected {want}, got {got}"
            )
    if failures:
        for failure in failures[:20]:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(f"300-cycle regression failed with {len(failures)} mismatches")
    print("PASS: GEM matched the independent HDL oracle for all 300 cycles")


if __name__ == "__main__":
    main()
