# Part D — measured throughput: macros preserved vs shredded

**Host:** NVIDIA GeForce GTX 1650 (14 SMs), Linux 7.1.11-zen, CUDA 13.3, 2026-09-03.
**Method:** `scripts/partd_sweep.sh` — same RTL synthesized two ways, identical
`$random` stimulus VCD, 2000 simulated cycles, median of 5 reps after 2 warm-ups,
**every configuration gated once with `--check-with-cpu`** (CPU interpreter ==
CUDA, word-for-word). Raw per-config JSON in `benchmark-results/partd-sweep/`.

- **preserved** = DSP48E2 / CARRY4 / SRLC32E kept native, evaluated on the V2
  heterogeneous wave engine.
- **shredded**  = the three macros lowered to AIG gates + flip-flops, evaluated
  on the unmodified-GEM V1 Boomerang engine.

## Results

| design | blocks | shredded V1 | preserved V2 | V2 / V1 |
|---|--:|--:|--:|--:|
| `hetero_farm` — 32×(DSP48E2 + CARRY4 + SRLC32E), mutually independent | 1 | 3,365 cyc/s | 15,546 cyc/s | **4.62×** |
| `hetero_farm` | 4 | 9,726 cyc/s | 16,874 cyc/s | **1.73×** |
| `hetero_farm` | 8 | 17,932 cyc/s | 17,046 cyc/s | 0.95× |
| `bench_mac` — 8-deep `CO[3]→CI` CARRY4 chain | 1 | 48,817 cyc/s | 15,622 cyc/s | 0.32× |

std dev across the 5 reps was < 1.5% in every cell; two independent sweeps agreed within 2%. Multi-block cooperative execution verified at 2 / 4 / 8 / 14 blocks.

## Graph size (paid once per design, unconditional)

| design | AIG pins preserved | AIG pins shredded | AND gates preserved | AND gates shredded | node reduction |
|---|--:|--:|--:|--:|--:|
| `hetero_farm` | 1,995 | 153,604 | **0** | 150,905 | **77×** |
| `bench_mac` | 303 | 5,286 | 96 | 5,143 | 17× pins / 54× gates |

The preserved `hetero_farm` netlist contains **no Boolean logic at all** — 96
native macro nodes and nothing else.

## Reading

1. **V2 cost is near-constant (~16–17k cyc/s)** across block count and design.
   The per-cycle path is dominated by **bit-serial operand assembly**: each
   DSP48E2 lane issues ~141 sequential selector loads/cycle to rebuild A[27],
   D[27], B[18], C[48], OPMODE[9] and the controls (CARRY4 / SRLC32E are the
   same shape, smaller). That is a per-macro *fixed* cost, so extra blocks only
   spread 96 macros over more idle lanes. The loads are already coalesced
   (Nsight: 0% excess sectors).
2. **V1 cost scales with parallelism**: 3.4k → 9.7k → 17.9k cyc/s as blocks go
   1 → 4 → 8, because ~150k independent shredded gates have work to spread.
3. **Crossover:** V2 wins 4.62× when one block is starved by the shredded
   blow-up; the margin shrinks as blocks feed V1's large parallel workload and
   reaches parity at 8 blocks on this 14-SM GPU. A bigger GPU, a bigger design,
   or more macros keeps V2 ahead longer (its cost stays flat; V1's grows).
4. **The open optimisation** (not a correctness gap): a cooperative *bulk*
   gather — all lanes load a macro's selector block into shared memory once,
   then each lane assembles its operand from shared — plus consuming the
   per-partition node ownership `HeteroPlacementV2` already computes. Multi-block
   cooperative execution itself is done and verified (2 / 4 / 8 / 14 blocks).
5. **Adverse case:** a *deep serial* macro chain forces one barrier-separated
   wave per level for ~1 macro of work each; if the shredded equivalent is also
   small, V1's bit-parallel fold wins (`bench_mac`, 0.32×). V2 favours *wide*
   macro parallelism. The chain is still evaluated **correctly** — unmodified
   GEM's batched macro path reads stale state for same-cycle macro→macro edges.

## Fixtures

- `tests/hetero/hetero_farm.sv` (+ `tb_hetero_farm.sv`) — the wide-parallel farm.
- `tests/hetero/bench_mac.sv` (+ `tb_bench_mac.sv`) — the deep serial chain.
- `synth_zenith.ys` preserves; `synth_baseline.ys` shreds. Both mirror the same
  pass order so the only difference is blackbox-vs-real macros.
