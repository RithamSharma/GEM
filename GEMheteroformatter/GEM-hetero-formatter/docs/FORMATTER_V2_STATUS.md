# Host-Side Macro Memory Formatter plan — status in this tree

The plan was written for `RithamSharma/GEM-Heterogeneous-Simulator` @ `d9e8518`
(`src/hetero.rs`, `MacroBuffers::pack()`, `HeteroDag`, `PartitionSchedule`,
`Placement`, `StateLayout`). **None of those exist here.** This tree
(`RithamSharma/GEM`, branch `staged-aig-release`) has `src/aig.rs`
(`EndpointGroup::{DSPBlock,Carry4Block,Srlc32eBlock}`), `src/primitive_models.rs`,
`src/flatten.rs::FlattenedScriptV1`, and the `kernel_v1_impl.cuh` batched macro
evaluator. Everything below was rebuilt against those real types.

## Phase-by-phase

| plan phase | file(s) | status |
|---|---|---|
| **0** freeze contract | `src/format_v2_build.rs::StateLayout` (frozen persistent `u32` image: primary inputs / DFF Q / SRAM RD / DSP P bits, then per-DSP 48-bit P, then per-SRLC 32-bit storage), doc comments | **done** for this tree's types. The plan's upstream `HeteroDag`/`Placement` inputs don't exist; `src/schedule.rs::HeteroSchedule` is the real stand-in. |
| **1** source/dest selectors + 64-bit encoding + decoder + **net→selector resolution** | `src/format_v2.rs` (`SourceSel`/`DestinationSel`, `encode_*`/`decode_*`, named masks, `FormatError`); `src/format_v2_build.rs::resolve_source` | **done, tested.** `resolve_source` uses the *scheduled edge*: const → `Constant`; primary input / DFF Q / SRAM RD / DSP P → `PreviousState`; producer in an earlier local wave → `LocalShared`; earlier major stage → `CurrentStage` (single-stage path exercised; multi-stage wired but not fixture-covered). Exhaustive round-trip + reserved-bit/bad-space rejection. |
| **2** versioned sectioned ABI + validator | `src/format_v2.rs` (`ScriptV2Header`, `Section`, `ProgramLayoutV2`, `validate`, `content_hash`, `V2_HEADER_WORDS = 32`); `csrc/format_v2_abi.h` (shared constants + `static_assert`) | **done, tested.** golden header; monotonic / non-overlap / 8- & 16-byte alignment / in-bounds / bad-magic / bad-version. `abi_constants_match_the_shared_header` cross-checks the CUDA header. `.gemparts` file untouched (V1 baseline preserved). |
| **3** field/bit-major warp-coalesced transpose | `src/format_v2.rs::transpose_selectors`, `selector_word_index`, `padded_count` | **done, tested** for `n = 0,1,31,32,33,63,64,255,256`: `padded_count = round_up(n,32)`, adjacent active lanes exactly 1 `u64` apart at fixed `(field,bit)`, padding lanes zero, fields disjoint. |
| **4** coalesced state-gather + liveness allocator | `src/format_v2_build.rs::{allocate_shared_slots, build_gather_plan, apply_gather}` | **done, tested.** live-range shared-slot allocator (`peak_shared_words` = true simultaneous max, not the operand+output sum; reuse verified on a 4-deep carry chain → 1 word). Gather dedups `PreviousState` words and groups them into ≤32-word coalesced rounds (one word used by 20 macros → 1 round of 1 word). |
| **5** `UVec`/UCC upload + V2 CUDA wrapper + build.rs + `formatter_gpu_test` | `src/format_v2_gpu.rs::FlattenedScriptV2` (one immutable `UVec<u64>`, one upload); `csrc/kernel_v2.cu` (`formatter_gpu_selfcheck_cuda`, `formatter_coalesced_probe_cuda`); `build.rs` (`--features v2`, separate lib `gemcu_v2`); `src/bin/formatter_gpu_test.rs` | **Rust half done** (`FlattenedScriptV2::{from_resolved, validates, program_bytes}`, compiles in-tree). **CUDA half written, uncompiled** — no `nvcc` in the authoring env. `--features v2` is additive: the `--features cuda` submission build never touches `kernel_v2.cu`. |
| **6** one authoritative CPU + CUDA decoder | `src/format_v2.rs::decode_selector_section` (independent); `src/format_v2_cpu.rs::interpret_cycle` (CPU V2); `csrc/format_v2_decode.cuh` (CUDA) | **CPU half done, tested end-to-end.** `logical AIG+schedule → build_resolved_program → decode_selector_section → interpret_cycle` agrees with `primitive_models` for a single CARRY4, a **same-cycle CARRY4 chain** (`CO[3]→CI`, correct in one pass — the exact V1 defect), SRLC32E post-edge taps, DSP48E2 multiply-only, and a 3-macro program. CUDA decoder shares `format_v2_abi.h`; **uncompiled**. |
| **7** Nsight coalescing proof | `benchmarks/formatter_coalescing.py`, `docs/FORMATTER_V2_COALESCING.md` | **harness + analysis only.** Theoretical target: 8 sectors/request for one `u64`/lane (vs up to 32 for instance-AoS). No measured number — no GPU. |

## Verify

```bash
cd hetero_verify_crate && cargo test        # 42 passing (no mt-kahypar needed)
```

In-tree (needs a machine that can build `mt-kahypar`):

```bash
cargo test schedule:: macro_layout:: format_v2:: format_v2_build:: format_v2_cpu::
cargo build --features v2 --bin formatter_gpu_test          # needs nvcc
cargo run  --release --features v2 --bin formatter_gpu_test # needs an NVIDIA GPU
```

## What the report can and cannot claim

**Can claim (host-verified, 42 tests):**

- macros are formatted from **scheduled nodes**, not raw net ids — every
  operand bit carries a typed `SourceSel` chosen by its scheduled edge;
- a versioned, sectioned, 64-bit, alignment-validated ABI with an independent
  decoder and a byte-fold `content_hash`;
- field/bit-major selector layout with **proven 1-word warp stride** (the
  literal "coalesced global memory bandwidth" property, at the layout level);
- a coalesced gather plan + live-range shared allocator so state loads are
  deduped and shared use is the true peak;
- immutable program vs mutable macro state are **separate** buffers;
- one `UVec<u64>` carries the whole program in one upload;
- a CPU interpreter of the *identical bytes* reproduces `primitive_models`,
  including a same-cycle macro chain in one pass.

**Cannot claim (needs the GPU box):**

- that `kernel_v2.cu` / `format_v2_decode.cuh` compile (no `nvcc` here);
- `formatter_gpu_test` PASS (device-vs-host checksum, byte-for-byte H2D/D2H);
- `compute-sanitizer` clean;
- any measured sectors/request or transfer count/volume.

Run the three in-tree commands above on the standardized machine before the PS
report states Part A bullet 2 is "fully implemented and optimized".
