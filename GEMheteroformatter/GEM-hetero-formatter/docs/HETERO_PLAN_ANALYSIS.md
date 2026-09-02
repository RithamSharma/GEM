# "Complete Heterogeneous CUDA Integration Plan" — analysis & execution

## TL;DR

The plan is written against a **different repo layout** (`RithamSharma/GEM-
Heterogeneous-Simulator`, files `src/hetero.rs`, `csrc/hetero_macros.cuh`,
`work/gem_ps_submission/`). This branch (`staged-aig-release`, commit
`474e06b` "Add heterogeneous FPGA primitive simulation support" + follow-ups)
already implements **most of Phases 0–5** in GEM's own idiom — but with one
architectural gap the team's own `CURRENT_AUDIT.md` already names:

> *"production topological traversal and Boomerang placement still branch only
> on `DriverType::AndGate` … A same-cycle combinational macro consumer can
> therefore sample the producer's old global-state value."*

The plan's real payload is fixing exactly that. Its Phases 3–4 (a versioned V2
script ABI + a brand-new `kernel_v2_impl.cuh` level-driven interpreter) are the
"correct" fix but are **multi-week, GPU-verification-gated work** — not
completable or provable before the Sept 3 deadline without risking the working
submission.

**Recommendation:** ship the working V1-integrated submission, plus the
**tested `src/schedule.rs`** (the documented blocker's missing brain) wired
into the existing staging path as the *interim* correctness fix, plus the
Part-E scheduling-equations doc. Treat the full V2 ABI/kernel as the
post-deadline "proper" track and hand it in as a design with a skeleton.

---

## What each phase maps to in *this* repo

| Plan phase | This repo already has | Gap |
|---|---|---|
| **0** freeze semantics + baseline | `src/primitive_models.rs` (exact CARRY4/DSP/SRLC models + tests), `SUBMISSION.md`, `VERIFICATION_RESULTS.md`, `BASE_REVISION.txt`, 300-cycle HDL differential (`tests/hetero/`, `scripts/run_300_simulation_test.sh`) | none material |
| **1** canonical heterogeneous DAG (`src/schedule.rs`) | `primitive_models::build_typed_schedule` — a *contract* on abstract `(kind, edges)`, unit-tested, **not fed by the real AIG** | needs a builder that walks `aig.carry4s/dsps/srlc32es` + `DriverType::{CARRY4,SRLC32E,DSP}` and resolves real producers → **`src/schedule.rs` in this deliverable** |
| **2** fused AIG regions + typed waves | — | `schedule.rs` `AigRegion` (macro-cut equivalence classes) + `TypedWave` |
| **3** versioned Script V2 ABI (`FlattenedScriptV2`) | `FlattenedScriptV1` only; macros ride inside `blocks_data`'s `shared_writeouts` via the `sram_duplicate_permute` gather | new — see `src/macro_layout.rs` (the SoA 64-bit device buffer, PS Part-A bullet 2) + a V2 stream if you go full-V2 |
| **4** level-driven CUDA V2 interpreter (`kernel_v2_impl.cuh`) | `kernel_v1_impl.cuh` evaluates **all DSPs, then all CARRY4s, then all SRLC32Es, once, after the AIG Boomerang fold** (lines ~437–511) | new kernel — skeleton in `csrc/kernel_v2_impl.skeleton.cuh`, **needs GPU bring-up** |
| **5** explicit old/current/next state | `kernel_v1_impl.cuh` already: DSP `P` read from `input_state` (old), SRLC storage word gated by resolved clock, CARRY4 always-write; `flatten.rs` initial-state image incl. SRLC `INIT` | correct for the *batched* model; V2 needs the same discipline per wave |
| **6** independent 3-way correctness gates | `--check-with-cpu` CPU executor + Icarus 300-cycle differential | add `carry_chain2/8.sv`, `aig_carry_aig.sv`, `cross_partition_chain.sv` fixtures; add a chained-macro GPU differential |
| **7** perf + Nsight | `benchmarks/run_benchmarks.py`, `benchmark.sh`, `--ncu` | run on the target GPU only |
| **8** docs | `SUBMISSION.md`, `CURRENT_AUDIT.md` | add `docs/V2_SCHEDULING.md` (in this deliverable) + block diagrams |

## The concrete bug the plan exists to fix

`src/aig.rs::topo_traverse_generic` (and `staging.rs::from_split`'s `level_id`)
recurse only through `DriverType::AndGate`. Every `CARRY4`/`SRLC32E` output pin
is a **traversal leaf**, so it is scheduled like a primary input:

- `flatten.rs::make_inputs_outputs` puts every macro output pin in `input_map`
  (the *previous-cycle* / FF-Q map): `input_map.insert(o, global_offset)` for
  CARRY4 `o_out`/`co_out`, DSP `p_out`, SRLC `q_out`/`q31_out`.
- `kernel_v1_impl.cuh` gathers all macro operands into `shared_writeouts`, then
  runs one `if (threadIdx.x < num_dsps)` block, one `< num_carry4s`, one
  `< num_srlc32es` — **no inter-macro ordering, once per partition per cycle.**

So `CARRY_A.CO[3] → CARRY_B.CIN` in the same partition, same cycle: `CARRY_B`
reads `CARRY_A.CO[3]` from `input_state` = **last cycle's value**. For
`DSP48E2` this is *correct* (`PREG=1` ⇒ `P` is registered). For `CARRY4` and
`SRLC32E` `Q`/`Q31` it is wrong.

## Two ways to fix it

### Path A — interim, safe, deadline-fit (recommended for Sept 3)

Keep the V1 kernel. Use `schedule.rs` to **force a major-stage cut** between a
macro and any same-cycle consumer at a higher level. The existing staged-IO +
cooperative `grid.sync()` machinery then carries the value in the *current*
cycle (kernel reads it via the `idx >> 31` staged-input path from
`output_state`, not `input_state`).

Wiring (host only, no kernel change):

1. `lib.rs`: `pub mod schedule; pub mod macro_layout;`
2. In the mapper (`src/bin/cut_map_interactive.rs`) after the `AIG` is built:
   ```rust
   let hs = gem::schedule::build_schedule(&aig)
       .unwrap_or_else(|e| panic!("heterogeneous schedule rejected: {e:?}"));
   ```
   `build_schedule` **is your combinational-loop gate** (PS: "a genuine
   combinational loop is rejected").
3. Turn `hs.macro_levels()` into extra `level_split` points / endpoint-subset
   boundaries for `staging::build_staged_aigs` so that for every
   `(u, v) ∈ edges_same` with `level(u) < level(v)`, `u`'s outputs
   (`hs.cross_level_staged_pins()`) become `StagedAIG::primary_output_pins`
   of an earlier stage. Simplest concrete rule: if
   `hs.macro_to_macro_edges()` is non-empty, split major stages at each
   distinct macro level.
4. `Partition::build_one`: refuse to co-locate a macro and a strictly-later
   same-cycle consumer in one partition (add a check against `hs`).

Cost: one extra `grid.sync()` per macro dependency level. For the PS's chained-
carry benchmarks that is a handful of syncs per cycle — correct, and still far
ahead of CPU. Document it as the honest number.

### Path B — full V2 (post-deadline "proper")

Phases 3–4 verbatim: `FlattenedScriptV2` + `macro_layout.rs` SoA buffer +
`kernel_v2_impl.cuh` that runs `schedule.rs::waves` as intra-block
`__syncthreads()`-separated waves, macros in type-homogeneous warps with
`__ballot_sync` tail masks, grid sync only at partition cuts. Skeleton
provided; needs `compute-sanitizer` clean + the 3-way differential on real
hardware before any perf claim.

## Deliverable contents

| file | status | drop-in |
|---|---|---|
| `src/schedule.rs` | complete + 8 unit tests; **not compiled here** (this box can't build `mt-kahypar`) — run `cargo test schedule` on the build machine | `src/`, add `pub mod schedule;` to `lib.rs` |
| `src/macro_layout.rs` | complete + 5 unit tests; same caveat | `src/`, add `pub mod macro_layout;` |
| `docs/V2_SCHEDULING.md` | Part-E scheduling equations | `docs/` |
| `csrc/kernel_v2_impl.skeleton.cuh` | **skeleton**, compiles-intent only, needs GPU bring-up | reference for Path B |
| this file | analysis | `docs/` |

## Verification status (be precise in the report)

- `schedule.rs` / `macro_layout.rs`: algorithm hand-traced against every test;
  **`cargo test` not run in this environment** (`mt-kahypar` C++ dep fails to
  build under MSVC here — unrelated to these files). Must pass on your machine
  before the report cites them.
- No CUDA compiled (no `nvcc` here). No GPU run. No perf number is defensible
  until `nvidia-smi` + `--check-with-cpu` + repeated timing + Nsight all pass
  on the standardized machine — exactly as `SUBMISSION.md` already says.
