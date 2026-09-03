# V2 selector layout: coalescing analysis (Host-Side Formatter plan, Phase 7)

## Layout

Every macro-type selector section is stored **field/bit-major** (`src/format_v2.rs::transpose_selectors`):

```
word(section_base + field_base + bit * padded_count + instance)
padded_count = round_up(n_instances, 32)
```

The GPU evaluates macro instance `i` on lane `i` of a warp. At a fixed
`(field, bit)`, lanes `0..31` read

```
base + k          for k = 0..31
```

— 32 **consecutive** `u64` words = one naturally-aligned 256-byte region.

## Theoretical minimum

For one contiguous `B`-byte value per lane in a full 32-lane warp, the ideal
number of 32-byte sectors a global load requests is

```
ideal_sectors = 32 lanes * B bytes / 32 bytes-per-sector = B
```

| value width per lane | ideal sectors/request |
|---|---|
| `u32` selector            | 4  |
| **`u64` selector (V2)**   | **8**  |
| `uint4` / 16-byte vector  | 16 |

The V2 selector word is `u64`, so a fully-coalesced warp read of one
`(field, bit)` row targets **8 sectors / request** and
`bytes_per_sector ≈ 100%`.

## What instance-AoS would cost

The pre-plan layout packed each bus instance-major:

```
instance0.A[0..27], instance1.A[0..27], ...
lane i reads  base + i*27 + bit   -> stride 27 u64 words between lanes
```

A warp then touches 32 addresses `≥ 27*8 = 216` bytes apart → up to **32
sectors / request**, `bytes_per_sector ≈ 8/256 ≈ 3%`. That is the "only 26.2 of
32 bytes per sector utilized … stride between threads" pattern in the baseline
Nsight report.

## How to measure (no numbers are invented here)

```bash
cargo build --release --features v2 --bin formatter_gpu_test
python3 benchmarks/formatter_coalescing.py --bin target/release/formatter_gpu_test
```

The script:

1. runs `formatter_gpu_test` and **aborts if the self-check fails** (magic /
   version / section alignment / device-vs-host `content_hash`);
2. `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,
   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,
   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct, …` over the
   `formatter_coalesced_probe` kernel;
3. computes `sectors / request` and compares to the ideal 8;
4. `nsys` confirms exactly **one** H2D transfer of `program_bytes()` and none
   in a loop;
5. writes `benchmark-results/formatter_v2.summary.json` +
   `formatter_v2_full.ncu-rep`.

Metric identifiers vary by architecture/toolkit — resolve on the target GPU
with `ncu --query-metrics | grep global_op_ld` before a final run.

## Acceptance (plan Phase 7)

- [ ] `formatter_coalesced_probe` global loads at ~8 sectors/request,
  `bytes_per_sector` ≥ 90 %.
- [ ] shared-memory bank conflicts zero, or quantitatively explained.
- [ ] exactly one immutable-program upload; none inside the cycle loop.
- [ ] a deliberately-retained instance-AoS reference section measurably worse
  on the same GPU with identical output (the A/B the plan asks for).
- [ ] `compute-sanitizer` memcheck / racecheck / synccheck clean.

All five require the target GPU; none can be produced in the authoring
environment (no `nvcc`, no driver).
