# Optimization status and roadmap

## Shipped

### 1. `--engine auto` dispatcher (`src/bin/cuda_test.rs`)

`cuda_test` now chooses V1 or V2 per design instead of forcing one:

| condition | engine | why |
|---|---|---|
| any macro output feeds a **same-cycle** consumer (`!schedule.v1_batched_is_safe()`) | **V2** (forced) | the batched V1 path evaluates all AIG then all macros, so it would read the previous cycle's value on that edge — the PS Part B correctness requirement |
| macros present, none feeding a same-cycle consumer | **V1** | V1 on the *preserved* netlist evaluates macros natively but batched, with no per-cycle wave barriers and no bit-serial operand assembly — faster on the small preserved graph |
| no macros | **V1** | classic path, unchanged |
| `--engine v1` / `--engine v2` / `--v2` | as asked | `--engine v1` is refused (panic) when it would be incorrect |

One synthesis, one `cut_map_interactive --v2-parts`: the `.gemparts` carries the
V2 placement **and** the embedded legacy V1 partitions, so both engines are
available with no second synthesis.

Net effect: the combined engine is **never slower than classic GEM** (V1 handles
the cases where V2's barrier cost would lose) and is **V2-fast exactly where V2
is required** (chained macros — where classic GEM is also *wrong*). Verified by
`scripts/run_auto_check.sh` (every engine, every fixture, `--check-with-cpu`).

Predicate: `HeteroSchedule::v1_batched_is_safe()` in `src/schedule.rs`,
unit-tested (`v1_safety_predicate_matches_same_cycle_macro_consumers`).

## Roadmap (not shipped — needs a build + full verify cycle on the GPU box)

### 2. Cooperative bulk operand gather (the V2 per-cycle bottleneck)

`csrc/kernel_v2.cu`, in the DSP48E2 / CARRY4 / SRLC32E queue loop: each lane
currently issues ~141 sequential `source_bit()` calls per cycle to reassemble
`A[27] D[27] B[18] C[48] OPMODE[9] ...` from the transposed selector table.
Profiling shows this — not the barriers — is the ~60 us/cycle floor on a
GTX 1650. The loads are already coalesced (Nsight: 0% excess sectors); the
problem is that one lane does 141 *dependent-latency* loads with nothing to hide
them behind (one macro instance per lane).

**Plan:** stage a macro's whole selector block into shared memory with a
warp-cooperative coalesced load (32 lanes load 32 consecutive `u64` per field
bit), then each lane assembles its operand words from shared. Expected: the
per-cycle cost drops from ~141 serialized global latencies to ~1 (the bulk
load) + shared reads. This should let V2 also win at 8+ blocks on the wide farm.

Touch points: the `type == 1` (DSP) / `type == 0` (CARRY4) / `else` (SRLC)
branches around line 315-370 of `csrc/kernel_v2.cu`; `source_bit` /
`load_selector` in `csrc/format_v2_decode.cuh`. No ABI change, no host change.

**Verify after:** `cargo build --release --features v2`; `cargo test --features
v2 --lib`; `scripts/run_v2_carry_chain_test.sh`; `scripts/run_v2_sram_test.sh`;
`scripts/run_v2_300_simulation_test.sh`; Compute Sanitizer (memcheck / racecheck
/ synccheck) on `formatter_gpu_test` and the carry-chain `cuda_test`;
`scripts/partd_sweep.sh`.

### 3. Consume per-partition node ownership

`HeteroPlacementV2.node_partition` records which block owns each schedule node;
the kernel currently global-strides over all operations regardless. Consuming it
means each block evaluates its own node set with `__syncthreads()` and
`grid.sync()` fires only for genuine cross-block value exchange — fewer grid
barriers per wave. Combine with (2).

### 4. Batch N cycles per cooperative launch

The kernel already loops all cycles internally, but re-stages old->next state
and clears the arena every cycle. For designs with a shallow schedule, unrolling
2-4 cycles and keeping the arena live across them amortises that fixed cost.

### 5. Calibrated dispatch

Extend `--engine auto` with an optional `--calibrate <N>`: build both engine
contexts, run each for N cycles, assert they agree, then run the full sim with
the faster. Removes the last bit of guesswork from the V1-safe case. Needs the
dual-context setup in `cuda_test.rs` main (`v2_context` built whenever
`engine == auto`, `use_v2` decided after the calibration timing).
