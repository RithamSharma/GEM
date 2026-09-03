# Benchmark evidence

- `v2_macro_runtime_final.json`: nine correctness-gated CUDA-event samples of
  the integrated AIG/macro microfixture, 10,000 kernel launches per sample.
- `v2_300cycle_1block.json`: nine correctness-gated whole-simulation samples
  using one CUDA block.
- `v2_300cycle_14block.json`: the same design, VCD, and 302 simulated cycles
  using 14 cooperative CUDA blocks.
- `part_b_v2.ncu-rep`: earlier Nsight Compute macro/formatter profile. It is
  retained as supporting evidence but is not the final integrated profile.
- `part_b_v2_summary.md`: interpretation and scope of that earlier profile.

All published timing collectors abort unless CUDA matches the CPU V2 oracle.
The final integrated Nsight report requires privileged GPU performance-counter
access and is generated with `sudo scripts/profile_v2_ncu.sh`.
