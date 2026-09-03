#!/usr/bin/env python3
"""Compare the direct two-CARRY4 chain against the independent HDL oracle."""

import csv
import re
import sys
from pathlib import Path


def read_actual(path, sample_times):
    declarations, bits, snapshots = {}, {}, {}
    current_time = None

    def snapshot():
        if current_time in sample_times:
            values = {}
            for name in ("o", "co"):
                values[name] = sum(bits[(name, bit)] << bit for bit in range(8))
            snapshots[current_time] = values

    with path.open() as stream:
        for raw in stream:
            line = raw.strip()
            match = re.match(r"\$var\s+\S+\s+1\s+(\S+)\s+(o|co)\[(\d+)\]\s+\$end", line)
            if match:
                declarations[match.group(1)] = (match.group(2), int(match.group(3)))
            elif line.startswith("#"):
                snapshot()
                current_time = int(line[1:])
            elif len(line) >= 2 and line[1:] in declarations:
                if line[0] not in "01":
                    raise AssertionError(f"unknown output at {current_time}: {line}")
                bits[declarations[line[1:]]] = int(line[0])
    snapshot()
    return snapshots


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_carry_chain_vcd.py <expected.csv> <gem-output.vcd>")
    with Path(sys.argv[1]).open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    output_time = lambda row: int(row["time"]) - 10_000
    actual = read_actual(Path(sys.argv[2]), {output_time(row) for row in expected})
    for row in expected:
        timestamp = output_time(row)
        want = {"o": int(row["o"], 16), "co": int(row["co"], 16)}
        got = actual.get(timestamp)
        if got != want:
            raise SystemExit(
                f"CARRY chain mismatch at vector {row['cycle']}: expected {want}, got {got}"
            )
    print(f"PASS: direct CARRY4-to-CARRY4 dependency matched HDL for {len(expected)} vectors")


if __name__ == "__main__":
    main()
