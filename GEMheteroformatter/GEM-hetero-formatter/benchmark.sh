#!/usr/bin/env bash
set -euo pipefail

echo "benchmark.sh no longer changes branches in your working tree."
echo "Build each revision in a separate git worktree, then run:"
echo "  python3 benchmarks/run_benchmarks.py --help"
