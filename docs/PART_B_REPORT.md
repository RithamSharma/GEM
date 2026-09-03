# GEM heterogeneous CUDA execution — Part B report

## 1. Problem solved

Original GEM schedules Boolean AIG work efficiently, but a preserved FPGA
macro is a word-level operation with many pins, different arithmetic, and
state semantics. Treating macro outputs as ordinary old-state leaves breaks a
same-cycle chain such as `CARRY4_A.CO[3] -> CARRY4_B.CI`. Running all AIG work,
then all DSPs, then all carries also breaks arbitrary mixed dependencies.

The V2 design makes AIG regions and preserved macros peers in one dependency
graph and executes that graph directly on the GPU.

## 2. Scheduling mathematics

Let `G = (V, E_same ∪ E_next)`. `V` contains maximal AIG regions and one node
per preserved macro. `E_same` contains values visible in the current cycle.
`E_next` contains registered state transitions and does not contribute to the
combinational topological order.

```text
level(v) = 0                                      if pred_same(v) is empty
level(v) = 1 + max(level(u) : (u,v) in E_same)   otherwise
```

Kahn levelization rejects the design if the same-cycle subgraph is cyclic. The
required invariant is:

```text
for every (u,v) in E_same: level(u) < level(v)
```

Each level is split into `Q[level, partition, type]`, where type is AIG region,
CARRY4, DSP48E2, or SRLC32E. A warp therefore never switches primitive type
per lane. A direct carry chain occupies successive waves; no fake AND node is
inserted.

## 3. Host formatter and memory mapping

Every operand is encoded as a 64-bit selector:

```text
SourceSel = (space, word-index, bit, invert, valid)
space = Constant | PreviousState | CurrentStage | LocalShared
```

The scheduled edge selects the space: registered/cycle inputs use previous
state; an earlier local wave uses shared memory; a same-cycle cross-block value
uses current-stage global memory; and literals use the constant space.

Selectors are field-major. For field bit `b`, padded instance count `N`, and
instance `i`, `address(b,i) = section_base + b*N + i`. Thus lanes 0–31 load
consecutive `u64` words. Every section is 8/16-byte aligned and validated.
Mutable DSP/SRLC/SRAM state is separate from the immutable program.

## 4. GPU architecture

| Data | GPU location | Reason |
|---|---|---|
| Header, wave descriptors, selectors | Global memory | Immutable, uploaded once, coalesced access |
| Previous/next cycle state | Global memory | Persistent across cycles |
| Cross-block current-cycle values | Global `CurrentStage` arena | Visible after grid barrier |
| Gathered words and local live values | Shared memory | Low-latency block reuse |
| Primitive operands/results | Registers | Direct GPU ALU execution |
| Wave fields | Warp registers via `__shfl_sync` | One load/broadcast, uniform control |

For every wave, the kernel broadcasts queue ranges, executes the AIG region
and type-homogeneous macro queues, publishes outputs, and synchronizes at the
real dependency boundary. Sequential endpoints commit only after the
combinational schedule settles.

## 5. Main file changes

| File | Change |
|---|---|
| `src/aig.rs`, `src/aigpdk.rs` | Preserve macro descriptors, pins, outputs, parameters, and endpoints |
| `src/schedule.rs` | Unified DAG, AIG-region fusion, typed waves, cycle rejection |
| `src/hetero_parts.rs` | Versioned legacy-compatible `.gemparts` wrapper and placement queues |
| `src/format_v2.rs` | Stable 64-bit ABI, sections, validation, hashing |
| `src/format_v2_build.rs` | State layout, selector resolution, liveness allocation, coalesced serialization, cross-block remap, SRAM operations |
| `src/format_v2_cpu.rs` | Independent CPU interpreter and SRAM/state oracle |
| `src/format_v2_gpu.rs` | Device-backed program transport |
| `csrc/format_v2_abi.h`, `csrc/format_v2_decode.cuh` | Shared host/device ABI and selector decoder |
| `csrc/hetero_primitives.cuh` | Fixed-width native CUDA primitive models |
| `csrc/kernel_v2.cu` | Integrated cooperative multi-block evaluator and state/SRAM commit |
| `src/bin/cut_map_interactive.rs` | Generate V2 placement |
| `src/bin/cuda_test.rs` | End-to-end V2 CLI, VCD flow, CPU gate, V1/V2 artifact compatibility |
| `synth_zenith.ys`, `scripts/run_synth_zenith.sh` | Preserve macros and retain synchronous RAM mapping |

## 6. Correctness evidence

- 49 Rust unit tests pass.
- The single-CARRY model is exhaustive over all 1,024 inputs.
- A synthesized direct two-CARRY chain passes 1,024 independent HDL vectors,
  and CPU/CUDA comparison using two CUDA blocks.
- Mixed DSP/CARRY4/SRLC32E/AIG execution matches independent HDL for all 300
  checked cycles and matches CPU V2 state word-for-word.
- Synthesized SRAM matches HDL for synchronous read and read-before-write.
- Compute Sanitizer reports zero memory, race, and synchronization errors.

## 7. Throughput analysis

For `C` simulated cycles and measured interval `T_ms`:

```text
throughput = 1000*C/T_ms cycles/s
speedup(A over B) = throughput_A/throughput_B
```

GTX 1650 results in `benchmark-results/`:

| Test | Median result |
|---|---:|
| 10,000-launch V2 microbenchmark | 16,111.832 kernel executions/s |
| Same microbenchmark | 96,670.991 scheduled operation evaluations/s |
| 302-cycle mixed test, 1 block | 11,252.378 cycles/s |
| 302-cycle mixed test, 14 blocks | 8,748.681 cycles/s |

The tiny fixture's 14-block rate is 0.7775x the one-block rate because a few
operations cannot amortize cooperative-grid barriers. This proves multi-block
correctness, not a speedup. A defensible original-GEM gain requires the
organizer's representative large netlists under the identical benchmark; this
report deliberately does not fabricate one.

## 8. Profiling scope

The included `part_b_v2.ncu-rep` covers the formatter/macro kernel. The final
integrated profile command is automated by `scripts/profile_v2_ncu.sh`; Linux
must grant GPU performance-counter access. Functional CUDA execution and all
Compute Sanitizer modes pass without that privilege.
