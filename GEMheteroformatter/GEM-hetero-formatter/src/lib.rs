// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
pub mod aigpdk;

pub mod aig;

pub mod staging;

pub mod repcut;

pub mod pe;

pub mod flatten;

pub mod primitive_models;

// ---------------------------------------------------------------------------
// Heterogeneous macro integration (added; see docs/FORMATTER_V2_STATUS.md).
// Pure host logic, no GPU dependency:
//   cargo test schedule:: macro_layout:: format_v2:: format_v2_build:: format_v2_cpu::
// ---------------------------------------------------------------------------
pub mod schedule;
pub mod macro_layout;
pub mod format_v2;
pub mod format_v2_build;
pub mod format_v2_cpu;
/// GPU-facing V2 buffer assembly (needs `ulib`; the kernel launch itself lives
/// in `src/bin/formatter_gpu_test.rs` behind `--features v2`).
pub mod format_v2_gpu;
