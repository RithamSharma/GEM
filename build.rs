//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("Building cuda source files for GEM...");
    println!("cargo:rerun-if-changed=csrc");

    #[cfg(feature = "cuda")] {
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda.debug(false).opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();

        #[cfg(feature = "v2")] {
            let mut cl_v2 = ucc::cl_cuda();
            cl_v2.ccbin(false);
            cl_v2.flag("-lineinfo");
            cl_v2.flag("-maxrregcount=128");
            cl_v2.debug(false).opt_level(3)
                .include(&csrc_headers)
                .files(["csrc/kernel_v2.cu"]);
            cl_v2.compile("gemcu_v2");
            println!("cargo:rustc-link-lib=static=gemcu_v2");
            ucc::bindgen(["csrc/kernel_v2.cu"], "kernel_v2.rs");
            ucc::make_compile_commands(&[&cl_cuda, &cl_v2]);
        }
        #[cfg(not(feature = "v2"))]
        ucc::make_compile_commands(&[&cl_cuda]);
    }
}
