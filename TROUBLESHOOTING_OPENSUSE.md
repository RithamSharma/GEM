# Troubleshooting Guide for Advanced Compilers (e.g. OpenSUSE Tumbleweed)

The NVIDIA GEM simulator depends heavily on legacy CUDA libraries and specific host compilers (GCC). If you are attempting to run this repository on an advanced Linux distribution with bleeding-edge packages (like GCC 15 on OpenSUSE), the standard `cargo build` command will fail because NVIDIA's `nvcc` strictly bans host compilers newer than the versions officially supported by the CUDA Toolkit.

Below are the safe, isolated steps to fix these build errors on your machine without permanently breaking the repository for standard Ubuntu distributions.

### 1. Missing `<cstdint>` Error (GCC 13+)
If you get an error that `uint8_t` or similar types are not defined in the `mt-kahypar` crate:
```bash
sed -i '1i #include <cstdint>' ~/.cargo/registry/src/index.crates.io-*/mt-kahypar-*/mt-kahypar-sc/mt-kahypar/utils/memory_tree.h
```

### 2. Unsupported Host Compiler Error
If `nvcc` throws a fatal error because it detects a compiler version > 13:
You must patch the `ucc` compilation crate to force it to bypass the version check.
```bash
sed -i 's/\.flag("-std=c++11")/.flag("-std=c++17").flag("-allow-unsupported-compiler")/' eda-infra-rs/ucc/src/compile.rs
```

### 3. Unsupported GPU Architecture Error (`sm_50`)
The `ucc` crate attempts to build for Pascal (`sm_50`) by default. If your CUDA Toolkit (e.g. CUDA 13) has removed support for `sm_50`, you must override the target architecture during the build step:
```bash
UCC_CUDA_GENCODE="80" UCC_CUDA_PTX="" cargo build --release --features cuda
```

### 4. "Too many blocks in cooperative launch" Error
If your consumer GPU (e.g. RTX 3050) has too few Streaming Multiprocessors (SMs) to support the default 108 cooperative block launch count hardcoded in the baseline benchmarks, you can lower the required block count directly in the benchmark script:
```bash
python3 benchmarks/run_benchmarks.py --bin ./target/release/cuda_test --gatelevel test_circuit/gatelevel.gv --gemparts test_circuit/result.gemparts --input_vcd test_circuit/golden_output.vcd --num_blocks 16
```
