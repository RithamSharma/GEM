# Part D — throughput: macros preserved vs shredded

## One command

```bash
./scripts/run_partd_benchmark.sh                       # bench_mac, 4000 cycles
./scripts/run_partd_benchmark.sh <design.sv> <top> [cycles] [num_blocks]
```

It runs the **same RTL two ways on identical stimulus**:

| flow | synthesis | engine |
|---|---|---|
| **preserved** | `scripts/run_synth_zenith.sh` — DSP48E2 / CARRY4 / SRLC32E kept native | V2 heterogeneous (`cut_map_interactive --v2-parts` → `cuda_test --v2`) |
| **shredded**  | `scripts/run_synth_baseline.sh` — macros lowered to AIG gates + FFs | V1 Boomerang (`cut_map_interactive` → `cuda_test`, i.e. unmodified GEM) |

Both are run once **with `--check-with-cpu`** as a correctness gate; the timed
repetitions (2 warm-ups + 7 measured, median reported) then run without the CPU
gate so the number is CUDA-only. Timing is parsed from the simulator's own
`simulation, Elapsed=<ms>` line against `total number of cycles: <n>`.

## Default fixture

`tests/hetero/bench_mac.sv` — macro-dense by design:

- 16 independent `DSP48E2` multiply-accumulate lanes (`P += A*B`, OPMODE
  `9'h025`). Each lowers to ~1.5k AIG gates when shredded, so the baseline
  netlist is ~24k gates vs 16 preserved macro nodes.
- an 8-stage `CARRY4` adder with the classic `CO[3] -> CI` chain, `S` driven by
  XOR glue logic (exercises the scheduler and the per-wave AIG fold).
- all sequential state lives in the DSP `PREG` registers — no plain flip-flops
  outside the macros, so it synthesizes cleanly through `synth_zenith.ys`.

`tests/hetero/preservation_top.sv` (1 DSP + 1 CARRY4 + 1 SRLC32E, fully
verified) is a safe fallback:

```bash
./scripts/run_partd_benchmark.sh tests/hetero/preservation_top.sv preservation_top 4000
```

## Output

```
benchmark-results/partd_summary.txt        # human-readable table + speedup + graph reduction
benchmark-results/partd_preserved.json     # per-rep samples, median, stdev, command
benchmark-results/partd_shredded.json
build/partd/*.log                          # yosys / map / gem logs for both flows
```

`partd_summary.txt` reports, for each flow, median cycles/second and AIG cell
count, then:

```
throughput speedup (preserved / shredded): <x>x
AIG graph reduction (shredded / preserved): <x>x
```

The script exits non-zero only if **both** flows fail; a partial result (e.g.
the shredded flow plus both AIG cell counts) is still emitted and is a
defensible Part D claim on its own.

## Measured (NVIDIA GTX 1650, 2026-09-03)

Full table + analysis in **`benchmark-results/partd_measured.md`**; raw JSON in
`benchmark-results/partd-sweep/`. Headline:

| design | blocks | shredded V1 | preserved V2 | V2 / V1 |
|---|--:|--:|--:|--:|
| `hetero_farm` (96 independent macros) | 1 | 3,365 cyc/s | 15,546 cyc/s | **4.62×** |
| `hetero_farm` | 4 | 9,726 cyc/s | 16,874 cyc/s | **1.73×** |
| `hetero_farm` | 8 | 17,932 cyc/s | 17,046 cyc/s | 0.95× |
| `bench_mac` (8-deep serial CARRY4) | 1 | 48,817 cyc/s | 15,622 cyc/s | 0.32× |

Graph size: `hetero_farm` preserved = 1,995 AIG pins / **0** AND gates vs
shredded 153,604 / 150,905 — **77× fewer nodes**. `bench_mac`: 17× / 54×.

V2's per-cycle cost is barrier-bound and near-constant (~16–17k cyc/s); V1
scales with block count. So preserving wins big when the shredded design starves
the GPU and reaches parity once ~150k gates fill all 14 SMs. Deep serial macro
chains are V2's weak case (still evaluated correctly, unlike batched V1).

## Nsight

```bash
sudo scripts/profile_v2_ncu.sh     # integrated simulate_v2_cycles_kernel report
```

Needs GPU performance-counter permission. The earlier formatter/macro-kernel
profile and its interpretation are in `benchmark-results/part_b_v2.ncu-rep` and
`benchmark-results/part_b_v2_summary.md` (0% excessive global sectors, 99.6%
branch efficiency).
