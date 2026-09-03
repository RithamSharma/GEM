# Examples

Worked end-to-end examples for the heterogeneous-macro flow. Each is a design
plus a numeric stimulus table — exactly the shape a judge supplies.

## `dsp_datapath` — all three macros + a forced-V2 dependency

| file | what it is |
|---|---|
| `dsp_datapath.v` | top module `judge_dsp_datapath` — a small DSP/carry datapath |
| `dsp_datapath_stim.csv` | 200 cycles of stimulus (`clk,rst,sample,addend`) |

What it exercises:

| block | role |
|---|---|
| `SRLC32E` | 32-cycle shift-register **delay line** on `sample[0]`; taps `Q` (position 5) and `Q31` combined into `prbs` |
| `DSP48E2` ×2 | tap 0 = `sample × 21` (OPMODE `9'h005`); tap 1 = accumulator `P += sample × 13` (OPMODE `9'h025`) |
| `CARRY4` ×2 | 8-bit ripple adder; **`c_lo.CO[3] → c_hi.CI` is a same-cycle macro→macro edge** |
| AIG glue | `mac = p0 + p1` adds the two DSP outputs in the same cycle |

> The SRLC32E has **no feedback** into its `D` pin. An `SRLC32E` (or `CARRY4`)
> whose data input is a combinational function of its own output is a
> combinational loop that GEM's scheduler cannot prove acyclic and will reject
> with `CombinationalCycle {...}`. Register such feedback through a flip-flop.

The CARRY4 chain and the DSP→adder path both make `schedule.v1_batched_is_safe()`
return `false`, so `--engine auto` is **forced to pick V2** — this is exactly the
case where the classic batched engine would read stale state.

### Run it

```bash
# from the repo root, after a successful build (see ../README.md)
./scripts/run_hidden.sh examples/dsp_datapath.v judge_dsp_datapath examples/dsp_datapath_stim.csv
```

Expected in the log:

```
engine=auto: N same-cycle macro->consumer edge(s) -> V2 (required for correctness)
selected simulation engine: V2 (heterogeneous)
total number of cycles: ~200
V2 CPU sanity test passed!
```

Output waveform: `build/hidden/output.vcd`. The `V2 CPU sanity test passed!`
line means every output value matched GEM's byte-accurate CPU reference for all
200 cycles.

### Stimulus format

One row per cycle. The header row names the design's input ports; values are
decimal or `0x`-hex. A `clk` column is used verbatim (here it is held 1, so
each row is one rising edge). See `scripts/stim_to_vcd.py --help` for headerless
tables, bit-slice column names, and radix control.
