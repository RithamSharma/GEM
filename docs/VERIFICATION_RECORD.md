# Final verification record

Date: 2026-09-03  
GPU: NVIDIA GeForce GTX 1650, driver 610.57.04

| Gate | Result |
|---|---|
| `cargo test --features v2 --lib` | 49 passed, 0 failed |
| release V2 CUDA build | passed |
| 300-cycle mixed independent HDL/CPU/CUDA | 300/300 matched |
| direct two-CARRY4 HDL/CPU/two-block CUDA | 1,024/1,024 matched |
| synthesized SRAM HDL/CPU/CUDA | 3/3 stateful samples matched |
| Compute Sanitizer memcheck | 0 errors |
| Compute Sanitizer racecheck | 0 hazards, 0 warnings |
| Compute Sanitizer synccheck | 0 errors |
| `git diff --check` | passed |

The integrated Nsight command was attempted and reached the CUDA process, but
Linux denied hardware performance counters with `ERR_NVGPUCTRPERM`. Run
`sudo scripts/profile_v2_ncu.sh` to collect that final optional artifact.
