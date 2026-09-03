# Verification results

> **SUPERSEDED (2026-09-03).** This file records the *first* verification pass,
> run on 2026-09-01 on a machine with **no NVIDIA driver** and **before** the V2
> heterogeneous engine was wired into `cut_map_interactive` / `cuda_test`. Its
> "Remaining PS-level blocker" note (production kernel does not execute a mixed
> same-cycle AIG/macro DAG) **no longer applies**: the V2 path now does exactly
> that and has been verified on a real GPU.
>
> For the current status see **[FINAL_VERIFICATION.md](FINAL_VERIFICATION.md)**
> (GTX 1650: 49 tests, 300/300 HDL differential, 1024-vector carry chain, SRAM,
> Compute Sanitizer 3/3 clean), **[PART_B_IMPLEMENTATION_STATUS.md](PART_B_IMPLEMENTATION_STATUS.md)**,
> and **[MERGED_SUBMISSION.md](MERGED_SUBMISSION.md)**. It is kept for history.

Run date: 2026-09-01 (Asia/Kolkata)

Base: `updated_gem_branch` revision
`0764603add3601ce5c40f97ba7e66a410394673e`.

Command:

```bash
./scripts/verify_submission.sh
```

## Passed gates

| Gate | Result |
|---|---|
| Shell and Python syntax | Pass |
| Yosys preservation of DSP48E2, CARRY4, and SRLC32E | Pass; input/output counts are one each |
| Valid CARRY4-only synthesis | Pass; absent macro types are not falsely required |
| DSP and SRLC parameter sidecar | Pass; includes `PREG` and `INIT` |
| Independent primitive tests | Pass; 4/4 tests, including all 1,024 CARRY4 input combinations |
| Rust all-target compile check | Pass |
| Heterogeneous NetlistDB parse and partition mapping | Pass; 413 netlist pins, 197 AIG pins, 61 endpoints, one partition |
| CPU synth-to-VCD fixture | Pass through timestamps 0, 5, and 15; non-empty output VCD produced |
| Independent HDL differential simulation | Pass; all DSP48E2, CARRY4, and SRLC32E outputs matched for 300 deterministic cycles |
| CUDA release translation | Pass with installed NVCC; `cuda_test` built successfully |
| Patch whitespace validation | Pass (`git diff --check`) |

Compiler warnings are inherited style/lifetime warnings and do not fail the
build.

The 300-cycle test exposed and led to repairs for two bugs that the earlier
short fixture did not cover: omitted leading zeroes in VCD vector changes were
misaligned onto the most-significant bus pins, and SRLC32E taps incorrectly
reported pre-edge storage instead of settling after the edge update.

## Not executable in this environment

`nvidia-smi` cannot communicate with an NVIDIA driver on this machine.
Consequently, GPU kernel execution, CUDA-vs-CPU differential comparison, and
Nsight performance collection were not run. CUDA compilation is not a
substitute for those tests.

## Remaining PS-level blocker

The production serializer/kernel still does not execute a same-cycle mixed
AIG/macro dependency DAG. It batches macros after ordinary AIG evaluation, so
a combinational macro chain can observe stale state. The tested typed scheduler
in `src/primitive_models.rs` is a correctness contract, not yet wired into the
production Boomerang path. See `CURRENT_AUDIT.md` for the exact remediation.

For that reason this artifact is a verified repair candidate, not evidence of
a complete Part B solution or a measured speedup.
