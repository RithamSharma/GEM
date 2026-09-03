// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Device-side decoder for the V2 macro program. All constants come from
// format_v2_abi.h (the single source of truth, cross-checked against
// src/format_v2.rs by `cargo test format_v2::tests::abi_constants_match_the_shared_header`).
//
// Compiled only under `--features v2` (build.rs). Bring-up gates before this
// counts toward Part A / Part B:
//   * compiles under build.rs's nvcc flags, cc >= 7.0
//   * formatter_gpu_selfcheck (csrc/kernel_v2.cu) passes: magic/version/align/
//     per-section checksum == host content_hash, byte-for-byte H2D/D2H
//   * compute-sanitizer memcheck + racecheck + synccheck clean
//   * CPU format_v2 decoder == this decoder (Phase 6)
#pragma once
#include "format_v2_abi.h"

namespace gem_v2 {

struct SourceSel { uint32_t index; uint32_t bit; uint32_t space; uint32_t invert; uint32_t valid; };
struct DestSel   { uint32_t index; uint32_t bit; uint32_t space; uint32_t valid; };

__device__ __forceinline__ SourceSel decode_source(uint64_t w) {
    SourceSel s;
    s.valid  = (w & SEL_VALID_BIT) ? 1u : 0u;
    s.index  = (uint32_t)((w & SEL_INDEX_MASK) >> SEL_INDEX_SHIFT);
    s.bit    = (uint32_t)((w & SEL_BIT_MASK) >> SEL_BIT_SHIFT);
    s.space  = (uint32_t)((w & SEL_SPACE_MASK) >> SEL_SPACE_SHIFT);
    s.invert = (w & SEL_INVERT_BIT) ? 1u : 0u;
    // a set reserved bit is a corrupt program; the host validator already
    // rejected it, so on-device we only assert in debug builds.
    #ifndef NDEBUG
    assert((w & SEL_RESERVED_MASK) == 0);
    #endif
    return s;
}

__device__ __forceinline__ DestSel decode_dest(uint64_t w) {
    DestSel d;
    d.valid = (w & SEL_VALID_BIT) ? 1u : 0u;
    d.index = (uint32_t)((w & SEL_INDEX_MASK) >> SEL_INDEX_SHIFT);
    d.bit   = (uint32_t)((w & SEL_BIT_MASK) >> SEL_BIT_SHIFT);
    d.space = (uint32_t)((w & SEL_SPACE_MASK) >> SEL_SPACE_SHIFT);
    #ifndef NDEBUG
    assert((w & (SEL_RESERVED_MASK | SEL_INVERT_BIT)) == 0);
    #endif
    return d;
}

struct HeaderView {
    const uint64_t* h;
    __device__ uint64_t magic()            const { return h[HW_MAGIC]; }
    __device__ uint32_t version()          const { return (uint32_t)h[HW_VERSION]; }
    __device__ uint64_t total_words()      const { return h[HW_TOTAL_WORDS]; }
    __device__ uint32_t num_major_stages() const { return (uint32_t)h[HW_MAJOR_STAGES]; }
    __device__ uint32_t num_partitions()   const { return (uint32_t)h[HW_PARTITIONS]; }
    __device__ uint32_t num_waves()        const { return (uint32_t)h[HW_WAVES]; }
    __device__ uint32_t shared_words()     const { return (uint32_t)h[HW_SHARED_WORDS]; }
    __device__ uint64_t feature_flags()    const { return h[HW_FEATURE_FLAGS]; }
    __device__ uint64_t content_hash()     const { return h[HW_CONTENT_HASH]; }
    __device__ uint32_t section_count()    const { return (uint32_t)h[HW_SECTION_COUNT]; }
    __device__ uint64_t sec_meta(int i)    const { return h[HW_SECTION_TABLE + 3 * i]; }
    __device__ uint64_t sec_start(int i)   const { return h[HW_SECTION_TABLE + 3 * i + 1]; }
    __device__ uint64_t sec_end(int i)     const { return h[HW_SECTION_TABLE + 3 * i + 2]; }
    __device__ uint32_t sec_kind(int i)    const { return (uint32_t)(sec_meta(i) >> 48); }
    __device__ uint32_t sec_align(int i)   const { return (uint32_t)((sec_meta(i) >> 32) & 0xffffu); }
    __device__ int find_section(uint32_t kind) const {
        for (uint32_t i = 0; i < section_count(); ++i)
            if (sec_kind((int)i) == kind) return (int)i;
        return -1;
    }
};

// Phase 3 coalesced selector read: lane == macro instance, so lanes 0..31 at a
// fixed (field,bit) touch consecutive u64 words -> one warp transaction.
__device__ __forceinline__ uint64_t load_selector(
    const uint64_t* __restrict__ section_base,
    uint32_t field_base_words, uint32_t bit, uint32_t padded_count_v, uint32_t instance)
{
    return section_base[field_base_words + bit * padded_count_v + instance];
}

// Gather one operand bit given its decoded selector. After the formatter's
// gather pass every operand is SRC_CONSTANT or SRC_LOCAL; the other two cases
// are kept for a caller that skips pre-gathering.
__device__ __forceinline__ uint32_t gather_bit(
    SourceSel s,
    const uint32_t* __restrict__ old_state,
    const uint32_t* __restrict__ cur_stage,
    const uint64_t* __restrict__ shared_arena)
{
    if (!s.valid && s.space != SRC_CONSTANT) return 0u;
    uint32_t v;
    switch (s.space) {
        case SRC_CONSTANT:   v = 0u; break;
        case SRC_PREV_STATE: v = (uint32_t)((old_state[s.index]  >> s.bit) & 1u); break;
        case SRC_CUR_STAGE:  v = (uint32_t)((cur_stage[s.index]  >> s.bit) & 1u); break;
        default:             v = (uint32_t)((shared_arena[s.index] >> s.bit) & 1ull); break; // SRC_LOCAL
    }
    return v ^ s.invert;
}

} // namespace gem_v2
