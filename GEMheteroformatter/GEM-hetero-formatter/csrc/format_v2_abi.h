// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Single source of truth for the V2 macro-program ABI, shared by
// csrc/format_v2_decode.cuh, csrc/kernel_v2*.cu and (by value, checked in a
// unit test) src/format_v2.rs. Plain C++/CUDA, no dependency on ulib types so
// it can be included from either side.
//
// >>> If you change a constant here, change src/format_v2.rs and re-run
// >>> `cargo test format_v2::tests::abi_constants_match_the_shared_header`.
#pragma once
#include <cstdint>

namespace gem_v2 {

// ---- 64-bit selector word bit-field (src/format_v2.rs: SEL_*) --------------
//   bits  0..=31 : index         (state word / shared slot / stage word)
//   bits 32..=37 : bit           (0..=63)
//   bits 38..=39 : space         (SourceSpace / DestSpace discriminant)
//   bit      40  : invert        (sources only)
//   bit      41  : valid         (0 => padding lane, must be an all-zero word)
//   bits 42..=63 : reserved, must be zero
static constexpr uint64_t SEL_INDEX_MASK    = 0x00000000FFFFFFFFull;
static constexpr uint32_t SEL_INDEX_SHIFT   = 0;
static constexpr uint32_t SEL_BIT_SHIFT     = 32;
static constexpr uint64_t SEL_BIT_MASK      = 0x0000003F00000000ull;
static constexpr uint32_t SEL_SPACE_SHIFT   = 38;
static constexpr uint64_t SEL_SPACE_MASK    = 0x000000C000000000ull;
static constexpr uint64_t SEL_INVERT_BIT    = 1ull << 40;
static constexpr uint64_t SEL_VALID_BIT     = 1ull << 41;
static constexpr uint64_t SEL_RESERVED_MASK = 0xFFFFFC0000000000ull;

enum SourceSpace : uint32_t {
    SRC_CONSTANT    = 0,
    SRC_PREV_STATE  = 1,
    SRC_CUR_STAGE   = 2,
    SRC_LOCAL       = 3,
};
enum DestSpace : uint32_t {
    DST_LOCAL       = 0,
    DST_CUR_STAGE   = 1,
    DST_NEXT_STATE  = 2,
};

// ---- program header (src/format_v2.rs: V2_MAGIC / V2_VERSION / header_words) --
static constexpr uint64_t V2_MAGIC        = 0x32565F4D45475F47ull; // "G_GEM_V2" LE
static constexpr uint32_t V2_VERSION      = 1;
static constexpr int      V2_HEADER_WORDS = 32;

// header word slots
static constexpr int HW_MAGIC = 0, HW_VERSION = 1, HW_TOTAL_WORDS = 2,
                     HW_MAJOR_STAGES = 3, HW_PARTITIONS = 4, HW_WAVES = 5,
                     HW_SHARED_WORDS = 6, HW_FEATURE_FLAGS = 7, HW_CONTENT_HASH = 8,
                     HW_SECTION_COUNT = 9, HW_SECTION_TABLE = 10; // 3 words / section

// ---- section kinds (src/format_v2.rs: SectionKind, declaration order) -------
enum SectionKind : uint32_t {
    SEC_BLOCKS_START = 0,
    SEC_WAVE_DESCRIPTORS = 1,
    SEC_QUEUES = 2,
    SEC_DSP_SRC = 3,
    SEC_DSP_DST = 4,
    SEC_DSP_CTL = 5,
    SEC_CARRY4_SRC = 6,
    SEC_CARRY4_DST = 7,
    SEC_SRLC32E_SRC = 8,
    SEC_SRLC32E_DST = 9,
    SEC_GATHER_PLAN = 10,
};

// ---- canonical contiguous field tables (src/format_v2_build.rs) ------------
// widths only; the flat-bit order is the concatenation in this order.
//   DSP  src: A27 D27 B18 C48 OPMODE9 ALUMODE4 INMODE5 CEP1 RSTP1  (= 140)
//   DSP  dst: P48
//   CARRY4 src: S4 DI4 CIN1 CYINIT1  (= 10)   dst: O4 CO4  (= 8)
//   SRLC   src: D1 CE1 A5  (= 7)               dst: Q1 Q31_1  (= 2)
static constexpr int DSP_SRC_BITS    = 140;
static constexpr int DSP_DST_BITS    = 48;
static constexpr int CARRY4_SRC_BITS = 10;
static constexpr int CARRY4_DST_BITS = 8;
static constexpr int SRLC_SRC_BITS   = 7;
static constexpr int SRLC_DST_BITS   = 2;

// warp-padded lane count: round_up(n, 32)
__host__ __device__ __forceinline__ uint32_t padded_count(uint32_t n) {
    return (n + 31u) & ~31u;
}

// field/bit-major address, in u64 words from the section base:
//   word = sum(width[f]*pc for f < field) + bit_in_field * pc + instance
__host__ __device__ __forceinline__ uint32_t selector_word_index(
    const int* field_widths, int field_count,
    uint32_t n_instances, int field_idx, uint32_t bit_in_field, uint32_t instance)
{
    uint32_t pc = padded_count(n_instances);
    uint32_t base = 0;
    for (int f = 0; f < field_idx; ++f) base += (uint32_t)field_widths[f] * pc;
    return base + bit_in_field * pc + instance;
}

#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
static_assert(sizeof(uint64_t) == 8, "u64");
static_assert(SEL_BIT_MASK == (0x3Full << SEL_BIT_SHIFT), "bit mask");
static_assert(SEL_SPACE_MASK == (0x3ull << SEL_SPACE_SHIFT), "space mask");
static_assert((SEL_INDEX_MASK | SEL_BIT_MASK | SEL_SPACE_MASK | SEL_INVERT_BIT
               | SEL_VALID_BIT | SEL_RESERVED_MASK) == ~0ull, "selector bits tile u64");
static_assert((SEL_INDEX_MASK & SEL_BIT_MASK) == 0, "no overlap");
static_assert(V2_HEADER_WORDS % 2 == 0, "16-byte aligned header end");
static_assert(DSP_SRC_BITS == 27+27+18+48+9+4+5+1+1, "dsp src field table");
#endif

} // namespace gem_v2
