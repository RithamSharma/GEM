#!/usr/bin/env python3
"""Compare the synthesized SRAM V2 output with the independent HDL oracle."""

import csv
import re
import sys
from pathlib import Path


def read_actual(path, sample_times):
    declarations, bits, snapshots = {}, {}, {}
    current_time = None

    def snapshot():
        if current_time in sample_times:
            if len(bits) != 32:
                raise AssertionError(f"missing q bits at {current_time}: {len(bits)}/32")
            snapshots[current_time] = sum(bits[i] << i for i in range(32))

    with path.open() as stream:
        for raw in stream:
            line = raw.strip()
            match = re.match(r"\$var\s+\S+\s+1\s+(\S+)\s+q\[(\d+)\]\s+\$end", line)
            if match:
                declarations[match.group(1)] = int(match.group(2))
            elif line.startswith("#"):
                snapshot()
                current_time = int(line[1:])
            elif len(line) >= 2 and line[1:] in declarations:
                if line[0] not in "01":
                    raise AssertionError(f"unknown q bit at {current_time}: {line}")
                bits[declarations[line[1:]]] = int(line[0])
    snapshot()
    return snapshots


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_sram_vcd.py <expected.csv> <gem-output.vcd>")
    with Path(sys.argv[1]).open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    # The oracle samples one nanosecond after the posedge; GEM labels the
    # propagated result with the active posedge itself.
    output_time = lambda row: int(row["time"]) - 1_000
    actual = read_actual(Path(sys.argv[2]), {output_time(row) for row in expected})
    for row in expected:
        timestamp = output_time(row)
        got = actual.get(timestamp)
        want = int(row["q"], 16)
        if got != want:
            raise SystemExit(
                f"SRAM mismatch at HDL {row['time']} / GEM {timestamp}: "
                f"expected {want:08x}, got {got!r}"
            )
    print(f"PASS: V2 SRAM matched independent HDL for {len(expected)} sampled edges")


if __name__ == "__main__":
    main()
