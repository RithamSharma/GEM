# Documentation index

The repository entry point is **[`../README.md`](../README.md)** — read that first.
It covers what the project is, how to build it, and how to run any design.

This folder holds the deeper material.

## Graded submission (Parts A–E)

| file | contents |
|---|---|
| [`COMPLIANCE.md`](COMPLIANCE.md) | **point-by-point response to the PS rules** — partial grading, single-clock / zero-init constraints, Yosys 0.68 + SystemVerilog, the custom-kernel requirement, the RTX 5060 (Blackwell) target machine, and the deliverables checklist |
| [`SUBMISSION_STATUS.md`](SUBMISSION_STATUS.md) | **part-by-part grading tracker** — status + exact evidence for Parts A, B, C, D, E |
| [`PART_B_REPORT.md`](PART_B_REPORT.md) | Part B write-up: the problem, the scheduling maths, the GPU memory mapping, the file-by-file change list, correctness evidence |
| [`PART_D_HOWTO.md`](PART_D_HOWTO.md) | Part D: how to run the preserved-vs-shredded throughput benchmark, and the measured numbers (GTX 1650) |
| [`PART_E.md`](PART_E.md) | Part E: what each documentation bullet is and where it lives |
| `GEM_Heterogeneous_Macro_Technical_Report.pdf` | the full technical report (regenerate: `python build_report.py`) |
| `GEM_PartE_Documentation.pdf` | the Part E deliverable (regenerate: `python build_part_e.py`) |
| `GEM_Heterogeneous_Macro_Explainer.pdf` | plain-language explainer |

## Design deep-dives

| file | contents |
|---|---|
| [`V2_SCHEDULING.md`](V2_SCHEDULING.md) | the modified GEM scheduling equations for the heterogeneous DAG — exactly what `src/schedule.rs` computes |
| [`FORMATTER_V2_COALESCING.md`](FORMATTER_V2_COALESCING.md) | the 64-bit selector layout and why warp lanes read consecutive words |
| [`OPTIMIZATION_ROADMAP.md`](OPTIMIZATION_ROADMAP.md) | the shipped `--engine auto` dispatcher + the remaining V2 kernel optimisations, with a verify checklist |
| [`VERIFICATION_RECORD.md`](VERIFICATION_RECORD.md) | what was run on a real NVIDIA GTX 1650 and the results |

## Operations

| file | contents |
|---|---|
| [`UPSTREAM_GEM_USAGE.md`](UPSTREAM_GEM_USAGE.md) | the original NVIDIA GEM usage guide (synthesis kit, `cut_map_interactive`, `cuda_test`) |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | building on distros with a too-new host compiler (nvcc GCC ceiling), and other gotchas |

## Archive

[`archive/`](archive/) — superseded and historical notes, kept for the audit trail.
