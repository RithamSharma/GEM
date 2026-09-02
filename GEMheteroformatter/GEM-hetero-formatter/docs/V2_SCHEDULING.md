# Modified GEM scheduling equations for the heterogeneous DAG

This is the Part E "mathematical definition of the modified GEM scheduling
equations". It documents exactly what `src/schedule.rs` computes.

## 1. Unified graph

Let the synthesized, macro-preserved netlist define

```
G = (V, E_same ∪ E_next)
```

### Vertices

```
V = R ∪ M
R = { AIG regions }                     (maximal AND-gate runs, see §3)
M = M_carry ∪ M_dsp ∪ M_srl             (one vertex per preserved macro cell)
```

Every synthesized operation maps to exactly one vertex, and every vertex
carries exactly one `NodeKind` (`AigRegion`, `Carry4`, `Dsp48e2`, `Srlc32e`).
This is the "one producer identity, one scheduled occurrence" invariant.

### Value sources (cycle start)

```
S = PI ∪ Q_dff ∪ RD_sram ∪ P_dsp ∪ {const 0}
```

`P_dsp` is in `S` because the PS constrains `DSP48E2` to `PREG = 1` with every
other register combinational: `P` is the value clocked on the *previous* rising
edge, so a same-cycle reader of `P` observes old state, identical to a DFF `Q`.

### Same-cycle edges `E_same`

`(u, v) ∈ E_same` iff `v` consumes, in the current cycle, a net produced by
`u` in the current cycle:

| `u`            | `v`            | condition                                             |
|----------------|----------------|------------------------------------------------------|
| region `r`     | macro `m`      | a gate of `r` drives an `S`/`DI`/`CIN`/`CYINIT` (CARRY4), `D`/`CE`/`A` (SRLC32E), or `A`/`B`/`C`/`D`/`INMODE`/`OPMODE`/`ALUMODE`/`CEP`/`RSTP` (DSP48E2) pin of `m` |
| macro `m`      | region `r`    | an `O`/`CO` pin (CARRY4) or `Q`/`Q31` pin (SRLC32E) of `m` feeds a gate of `r` |
| macro `m`      | macro `m'`    | same, where the sink pin belongs to another macro (`CARRY_A.CO[3] → CARRY_B.CIN`) |
| region `r`     | region `r'`   | a gate of `r` feeds a gate of `r'` and the two gates carry different macro cuts (§3) |

`DSP48E2` contributes **no** outgoing `E_same` edge: `P` is old state.
`SRLC32E` **does**, because the frozen semantics evaluate the asynchronous
`Q`/`Q31` taps against post-shift storage in the same HDL timestamp.

### Cross-cycle edges `E_next`

```
E_next = { (D → Q) : DFF }
       ∪ { (P_next → P) : DSP48E2 }
       ∪ { (shift_next → storage) : SRLC32E }
       ∪ { (wr → rd) : sync SRAM }
```

`E_next` is **excluded** from the same-cycle in-degree. Registered feedback is
therefore always schedulable; the only rejected topology is a genuine
combinational cycle in `E_same`.

## 2. Levelization (Kahn recurrence)

```
level(v) = 0                                if  ∀u. (u, v) ∉ E_same
level(v) = 1 + max { level(u) : (u, v) ∈ E_same }   otherwise
```

Computed by Kahn peeling over `E_same`. If the peel visits fewer than `|V|`
vertices, `E_same` has a cycle → `ScheduleError::CombinationalCycle` naming the
residual vertices (`build_schedule`).

**Invariant P1** — `∀ (u, v) ∈ E_same : level(u) < level(v)` (strict).
**Invariant P2** — every `v ∈ V` appears in exactly one level.

## 3. AIG regions (barrier amortization)

Scheduling each AND gate as its own vertex would put one barrier between every
gate and destroy Boomerang throughput. Instead, for each AND-gate pin `g`:

```
cut(g) = ⋃  contrib(x)            over the two fan-in pins x of g
contrib(x) = cut(x)               if x is an AND gate
           = { node(x) }          if x is a CARRY4 / SRLC32E output pin
           = ∅                    if x is a source or a DSP48E2 P pin
```

`cut` is monotonic along AND edges, so a single ascending-pin-id pass (AIG pins
are built in topological order) computes it. **Region = equivalence class of
AND-gate pins under `cut` equality.** Each region depends on exactly the macros
in its cut, all of which are strictly upstream, so region↔macro cycles in the
node graph are impossible unless the underlying netlist has a real
combinational loop (which §2 rejects).

Trade-off: two AND-gate-adjacent gates whose cuts differ by one macro land in
different regions (an extra wave), and gates with equal cut but no wire between
them share a region vertex (a placement, not correctness, concern). A later
SCC-condensation refinement can merge more aggressively; the equivalence-class
version is what ships because it is provably safe without a compiler in the
loop.

## 4. Type-homogeneous waves

```
Wave(ℓ)   = { v ∈ V : level(v) = ℓ }
Wave(ℓ)   = Q_aig(ℓ) ⊎ Q_carry(ℓ) ⊎ Q_dsp(ℓ) ⊎ Q_srl(ℓ)
```

Each `Q_k(ℓ)` holds a single `NodeKind`, so a GPU warp draining a queue never
runs a per-instance `switch` over primitive kinds — the PS "type-homogeneous
warps" requirement.

## 5. Execution order (one simulated cycle)

```
for ℓ = 0 .. L-1:                       # L = num_levels
    evaluate Q_aig(ℓ)     with the 256-thread Boomerang fold
    evaluate Q_carry(ℓ)   one CARRY4 per lane, warp-stride
    evaluate Q_srl(ℓ)     one SRLC32E per lane (async taps only)
    evaluate Q_dsp(ℓ)     one DSP48E2 per lane, int64 ALU
    publish level ℓ:
        warp-local producer→consumer   →  __syncwarp(mask)
        block-local producer→consumer  →  __syncthreads()
        block-crossing producer        →  stage to current-cycle global + grid.sync()
commit wave (once, at the cycle boundary):
    DFF.D, DSP48E2.P_next, SRLC32E.shift_next, SRAM writes   → next-state image
```

Old state is immutable for the whole `ℓ` loop; next state is written exactly
once, in the commit wave. That is the "persistent old state is immutable during
combinational evaluation" invariant.

## 6. Mapping to the GPU memory hierarchy

| datum | space | rationale |
|---|---|---|
| macro operand words, arithmetic temporaries (`AD`, `M`, carry chain `c[]`) | **registers** | thread-private, never shared |
| descriptor / queue metadata broadcast to a warp | **registers** via `__shfl_sync` | one lane loads, broadcasts |
| Boomerang 8192-bit working set | **shared** (existing packed area) | unchanged |
| current-cycle cross-warp values (`Q`/`Q31`, `O`/`CO`, region outputs a later wave reads) | **shared** value arena | published under `__syncthreads()` |
| old-state image, next-state image, cross-block staged current-cycle values | **global** | persists across the grid barrier |
| macro operand/result **side buffer** (V2) | **global**, SoA, 64-bit aligned | one `uint64_t` per lane per field → coalesced 2 KB transactions (see `macro_layout.rs`) |

## 7. Correspondence to code

| symbol | `src/schedule.rs` |
|---|---|
| `V` | `HeteroSchedule::nodes` |
| `E_same` | `HeteroSchedule::edges_same` |
| `level(v)` | `HeteroNode::level` |
| `Wave(ℓ)` split by kind | `HeteroSchedule::waves[ℓ]` (`TypedWave`) |
| producer map (net → this-cycle vertex) | `HeteroSchedule::producer` |
| cycle rejection | `ScheduleError::CombinationalCycle` |
| `level(m)` per macro cell | `HeteroSchedule::macro_levels()` |
| direct `CO[3] → CIN` edges | `HeteroSchedule::macro_to_macro_edges()` |
