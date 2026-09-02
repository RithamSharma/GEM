# Heterogeneous macro work — changes on top of `staged-aig-release`

Base: `RithamSharma/GEM` @ `ceed2de` (branch `staged-aig-release`), working tree
clean. **Additive**: only `Cargo.toml`, `build.rs` and `src/lib.rs` are edited
(feature + module declarations); nothing is committed. The `--features cuda`
submission build is byte-for-byte unaffected.

## Modified (3 files)

| file | change |
|---|---|
| `src/lib.rs` | `pub mod schedule / macro_layout / format_v2 / format_v2_build / format_v2_cpu / format_v2_gpu;` |
| `Cargo.toml` | `+ [features] v2 = ["cuda"]` ; `+ [[bin]] formatter_gpu_test` (`required-features = ["v2"]`) |
| `build.rs` | `#[cfg(feature = "v2")]` block compiling `csrc/kernel_v2.cu` into a **separate** static lib `gemcu_v2` + its ucc binding. No effect without `v2`. |

## New host source — pure logic, no GPU, no `mt-kahypar` (42 unit tests)

| file | Host-Side Formatter plan phase | tests |
|---|---|---|
| `src/schedule.rs` | (companion plan) canonical heterogeneous same-cycle DAG over AIG regions + CARRY4/DSP48E2/SRLC32E; Kahn levelization; type-homogeneous waves; combinational-loop rejection. | 8 |
| `src/macro_layout.rs` | logical macro field tables + per-bit net provenance + schedule-driven instance order | 5 |
| `src/format_v2.rs` | **1-3, 6(host)**: `SourceSel`/`DestinationSel` + one 64-bit device encoding (named masks, checked convert, independent decoder); `ProgramLayoutV2` + `validate` (monotonic / non-overlap / 8- & 16-byte aligned / in-bounds); `transpose_selectors` field/bit-major with proven 1-word warp stride; `content_hash`; `assemble_macro_program` | 13 |
| `src/format_v2_build.rs` | **0, 1, 3, 4**: `StateLayout` (frozen persistent image); `resolve_source` (scheduled-edge → typed selector); `allocate_shared_slots` (live-range shared allocator); `build_gather_plan` (dedup + coalesced rounds); `build_resolved_program` — the production entry point | 6 |
| `src/format_v2_cpu.rs` | **6**: `interpret_cycle` — CPU V2 interpreter of the identical bytes via `primitive_models`; agrees with the reference model on a single CARRY4, a **same-cycle CARRY4 chain** (`CO[3]→CI` correct in one pass), SRLC32E post-edge taps, DSP48E2 multiply-only, and a 3-macro program | 7 |
| `src/format_v2_gpu.rs` | **5 (Rust half)**: `FlattenedScriptV2` — one immutable `UVec<u64>` (header + all sections), one upload; `program_bytes`, `validates` | compiles in-tree |

## New CUDA — written against the ABI, **uncompiled** (no `nvcc` in authoring env)

| file | |
|---|---|
| `csrc/format_v2_abi.h` | single source of truth for the ABI constants + `static_assert`s; `src/format_v2.rs` cross-checks it (`abi_constants_match_the_shared_header`) |
| `csrc/format_v2_decode.cuh` | device selector/header decoder, includes `format_v2_abi.h` |
| `csrc/kernel_v2.cu` | `formatter_gpu_selfcheck_cuda` (magic/version/align/device-vs-host `content_hash`) + `formatter_coalesced_probe_cuda` (Nsight sectors/request) |
| `csrc/kernel_v2_impl.skeleton.cuh` | level-driven macro evaluator structure (companion execution-engine plan; skeleton) |

## New bin / benchmark / docs

| file | |
|---|---|
| `src/bin/formatter_gpu_test.rs` | Phase 5 harness: builds a 3-macro design, formats it, one `UVec` upload, runs the GPU self-check, exit 0 iff all flags set |
| `benchmarks/formatter_coalescing.py` | Phase 7: drives `formatter_gpu_test` under `ncu`/`nsys`, computes sectors/request vs the ideal 8, gated on the self-check passing |
| `docs/FORMATTER_V2_STATUS.md` | **read first** — phase-by-phase status |
| `docs/FORMATTER_V2_COALESCING.md` | Phase 7 theoretical analysis + how to measure |
| `docs/V2_SCHEDULING.md` | Part-E scheduling equations |
| `docs/HETERO_PLAN_ANALYSIS.md` | the plans vs. this repo |

## Verification crate

`hetero_verify_crate/` — standalone (3 deps, not in the main build), stubs
`src/aig.rs` field-for-field so every new host module compiles and tests
without `mt-kahypar`.

```bash
cd hetero_verify_crate && cargo test      # 42 passing, 0 warnings from cargo build/test
```

In-tree, on a machine that can build the crate:

```bash
cargo test schedule:: macro_layout:: format_v2:: format_v2_build:: format_v2_cpu::
cargo build  --features v2 --bin formatter_gpu_test           # needs nvcc
cargo run -r --features v2 --bin formatter_gpu_test           # needs an NVIDIA GPU
python3 benchmarks/formatter_coalescing.py                    # needs ncu
```

## Honest status

- **Compiled + unit-tested in isolation:** all six host modules (42 tests).
- **Compiles in-tree, not built here:** `format_v2_gpu.rs` (`mt-kahypar` fails
  under MSVC on the authoring machine — unrelated to this code).
- **Written, not compiled:** `csrc/kernel_v2.cu`, `format_v2_decode.cuh`,
  `format_v2_abi.h` — no `nvcc`.
- **Not done (needs a GPU):** `formatter_gpu_test` PASS, `compute-sanitizer`
  clean, any measured sectors/request or transfer count. The committed V1
  kernel still batches macros — V2 is a parallel, opt-in path, not a
  replacement.
