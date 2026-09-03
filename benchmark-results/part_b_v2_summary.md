# Nsight Compute analysis: Part B V2 macro-wave kernel

## Measurement identity

- GPU: NVIDIA GeForce GTX 1650, compute capability 7.5, 14 SMs.
- Kernel: `evaluate_v2_macro_waves`.
- Launch: 1 block x 256 threads.
- Fixture: two dependency waves and four total macro evaluations.
- Raw report: `part_b_v2.ncu-rep`.

## Key metrics

| Area | Metric | Result | Interpretation |
|---|---:|---:|---|
| Launch | Grid coverage | 0.04 waves | One block cannot fill a 14-SM GPU. |
| Timing | Profiled duration | 92.42 us | Instrumented duration; not normal throughput. |
| Compute | Compute throughput | 0.18% | The tiny fixture supplies almost no parallel work. |
| Memory | Memory throughput | 0.81% | The kernel is not bandwidth-bound. |
| Scheduler | No eligible warp | 97.42% | Schedulers are idle or waiting almost all the time. |
| Scheduler | Eligible warps/scheduler | 0.03 | Insufficient independent work is available. |
| Synchronization | CTA-barrier stall | 62.1 cycles; about 80% | The dependency-wave barrier dominates this tiny launch. |
| Branching | Branch efficiency | 99.60% | Type-homogeneous warp dispatch has very little branch divergence. |
| Warp use | Active threads/warp | 12.89 of 32 | Small type queues leave many lanes inactive. |
| Registers | Registers/thread | 104 | Register pressure limits theoretical occupancy to 50%. |
| Occupancy | Achieved occupancy | 25.13% | Only one block was launched, below the theoretical limit. |
| Memory safety | Local/shared spills | None | No spill traffic was reported. |
| Cache | L1 / L2 hit rate | 57.72% / 72.49% | Cache behavior is acceptable for this small diagnostic. |
| Coalescing | Global-load bytes/sector | 3.0 of 32 on 37.5% of L1 misses | Sparse queues waste memory sectors. |
| Coalescing | Global-store bytes/sector | 8.4 of 32 on 55.6% of L1 misses | Small output queues also underfill sectors. |

## What the result proves

1. Nsight can profile the V2 CUDA kernel successfully on the target GPU.
2. Type-homogeneous warp control is working: branch efficiency is 99.60%.
3. The kernel has no local-memory spill or shared-bank-conflict problem in this
   fixture.
4. Correctness checks still pass while the profiled kernel executes two direct
   dependency waves.

## What it does not prove

1. It does not establish whole-GEM speedup; the normal GEM path still uses V1
   for Boomerang AIG work.
2. It does not measure representative GPU saturation because only one block
   and four macros are launched.
3. It does not demonstrate full-warp coalescing. The formatter is field-major,
   but queues of only one or two instances cannot fill a 32-lane transaction.
4. The printed 14,627.9 kernel executions/s is collected under 34 profiling
   passes and must not be used as production throughput.

## Valid timing evidence

Use `carry_chain2_v2_runtime.json`, collected with CUDA events outside Nsight:

- median kernel executions/s: 90,171;
- median macro evaluations/s: 180,342.

This is a two-CARRY microbenchmark, not an original-versus-modified GEM
comparison.

## Required next performance experiment

After unified AIG/macro integration, benchmark the same synthesized designs on
V1 and V2 with enough partitions/blocks to occupy all 14 SMs. Include queue
sizes 1, 31, 32, 33, 256 and a large real design. Report CUDA-event median,
p95, simulated cycles/s, macro evaluations/s, and Nsight scheduler, barrier,
occupancy, and memory-sector metrics. Correctness against CPU/HDL must gate
every timing sample.
