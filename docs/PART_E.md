# Part E — Documentation & Reports (10 pts)

Full deliverable: **`docs/GEM_PartE_Documentation.pdf`** (11 pages, regenerate with
`python docs/build_part_e.py`).

| PS Part E bullet | Where it is | Status |
|---|---|---|
| Mathematical definition of the modified GEM scheduling equations for the heterogeneous DAG | PDF Part 1 (formal); `docs/V2_SCHEDULING.md` (markdown); realized in `src/schedule.rs` | **complete** — vertex set, macro-cut fusion rule, `E_same`/`E_next` partition with the edge table, the Kahn level recurrence, invariants P1/P2 with proofs, the type-homogeneous wave decomposition, the commit rule, complexity, and the worked `CO[3]→CIN` cascade. Every equation corresponds to tested code. |
| Architectural block diagrams mapping the FPGA primitives to the **specific** GPU memory hierarchy (Global vs Shared vs Registers) | PDF Part 2, §2.1–2.7, Figures 1–5 | **complete** — the three-tier table, a hardware-specific sub-table for the **RTX 5060 (Blackwell GB206)** evaluation GPU and the verified GTX 1650, the whole-cycle dataflow figure, one block diagram per primitive (DSP48E2 / CARRY4 / SRLC32E) showing exactly which datum lives in which tier and the per-cycle traffic, the wave-execution timeline, the barrier taxonomy, and the per-buffer placement rationale. Also condensed as §7.5 of the Technical Report. |
| Extensive numerical analysis of the throughput gains | PDF Part 3, §3.1–3.9 | **complete** — analytical shred-cost model (node/depth blow-up per primitive), the execution-time model `T_shred / T_native`, the memory-traffic model, a worked projection, the measurement protocol, the JSON schema, **and measured results** (§3.8, NVIDIA GTX 1650, two independent sweeps within 2%: 77× node reduction, 4.62× / 1.73× cycles/s at 1 / 4 blocks) plus threats to validity. Re-measure on the panel machine with `./scripts/partd_sweep.sh`. |

## Regenerating

```bash
pip install reportlab
python docs/build_part_e.py          # -> docs/GEM_PartE_Documentation.pdf
```
