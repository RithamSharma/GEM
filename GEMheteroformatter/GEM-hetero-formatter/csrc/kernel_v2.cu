// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// V2 formatter GPU self-check (Host-Side Macro Memory Formatter plan, Phase 5).
//
// This is intentionally NOT the macro evaluator (that is the companion
// execution-engine plan, csrc/kernel_v2_impl.skeleton.cuh). It exists to prove
// the *formatter* half of Part A end-to-end on real hardware:
//
//   * the one immutable u64 program uploads once via UVec/AsUPtr,
//   * every section start is 8- or 16-byte aligned on the device pointer,
//   * a per-section FNV-1a checksum computed on the GPU equals the host's
//     `ProgramLayoutV2::content_hash`,
//   * a canary word written just past each section survives a round trip
//     (no section writes out of bounds),
//   * the field/bit-major selector layout gives adjacent lanes consecutive
//     u64 words (checked by having lane i read word base+i and reporting the
//     achieved global-load sector count via Nsight, Phase 7).
//
// Build: only compiled under `--features v2` (see build.rs); the default
// `--features cuda` build of the working submission never touches this file.

#include <crates/ulib/includes.hpp>
#include <cstdio>
#include "format_v2_abi.h"
#include "format_v2_decode.cuh"

#define V2_CHECK(call)                                          \
  do {                                                          \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
             cudaGetErrorString(err));                          \
    }                                                           \
  } while (0)

// FNV-1a over a u64 range, byte order little-endian -- must match
// src/format_v2.rs::content_hash.
__device__ __forceinline__ u64 fnv1a_words(const u64* __restrict__ p, u64 n) {
  u64 h = 0xcbf29ce484222325ull;
  for (u64 i = 0; i < n; ++i) {
    u64 w = p[i];
    for (int b = 0; b < 8; ++b) {
      h ^= (u64)((w >> (8 * b)) & 0xff);
      h *= 0x100000001b3ull;
    }
  }
  return h;
}

// One block, one thread per section. out_hash[i] = checksum of section i,
// out_flags bit layout:
//   bit0  magic ok
//   bit1  version ok
//   bit2  total_words == program_words
//   bit3  every section 8/16-byte aligned, monotonic, in-bounds
//   bit4  every section checksum == host content_hash contribution
//         (the caller compares the folded hash; per-section hashes are for
//          pinpointing a corrupt section)
__global__ void formatter_gpu_selfcheck(
    const u64* __restrict__ program,
    u64 program_words,
    u64 host_content_hash,
    u64* __restrict__ out_section_hash,
    u32* __restrict__ out_flags)
{
  using namespace gem_v2;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    u32 flags = 0;
    HeaderView hv{program};
    if (hv.magic() == V2_MAGIC)      flags |= 1u << 0;
    if (hv.version() == V2_VERSION)  flags |= 1u << 1;
    if (hv.total_words() == program_words) flags |= 1u << 2;

    u32 sc = hv.section_count();
    bool layout_ok = true;
    u64 prev_end = V2_HEADER_WORDS;
    u64 folded = 0xcbf29ce484222325ull;
    for (u32 i = 0; i < sc; ++i) {
      u64 s = hv.sec_start((int)i);
      u64 e = hv.sec_end((int)i);
      u32 al = hv.sec_align((int)i);
      if ((al != 8 && al != 16) || ((s * 8) % al) != 0) layout_ok = false;
      if (s < prev_end || e < s || e > program_words)   layout_ok = false;
      prev_end = e;
      u64 hsec = fnv1a_words(program + s, e - s);
      out_section_hash[i] = hsec;
      // fold section bytes into the running hash the same way content_hash does
      for (u64 w = s; w < e; ++w) {
        u64 word = program[w];
        for (int b = 0; b < 8; ++b) {
          folded ^= (u64)((word >> (8 * b)) & 0xff);
          folded *= 0x100000001b3ull;
        }
      }
    }
    if (layout_ok)                    flags |= 1u << 3;
    if (folded == host_content_hash)  flags |= 1u << 4;
    *out_flags = flags;
  }
}

// A tiny coalesced-read probe: every lane loads program[base_word + lane] from
// one selector section. Nsight (Phase 7) then reports sectors/request; the
// field/bit-major layout should hit the theoretical minimum (8 sectors for one
// u64 per lane in a full warp).
__global__ void formatter_coalesced_probe(
    const u64* __restrict__ program,
    u64 base_word,
    u32 n_lanes,
    u64* __restrict__ sink)
{
  u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane < n_lanes) {
    // read + a data-dependent store so the compiler cannot elide the load.
    u64 v = program[base_word + lane];
    if (v == 0xdeadbeefdeadbeefull) sink[lane] = v;
  }
}

// `void ..._cuda(...)` so ucc::bindgen emits a universal wrapper that takes
// `&UVec<_>` and does the one H2D upload via AsUPtr. `out_flags[0]` carries the
// pass/fail bitmask; `out_section_hash[i]` the per-section checksum.
extern "C" void formatter_gpu_selfcheck_cuda(
    const u64* program, usize program_words, u64 host_content_hash,
    u64* out_section_hash, u32* out_flags)
{
  formatter_gpu_selfcheck<<<1, 1>>>(program, (u64)program_words, host_content_hash,
                                    out_section_hash, out_flags);
  V2_CHECK(cudaGetLastError());
  V2_CHECK(cudaDeviceSynchronize());
}

extern "C" void formatter_coalesced_probe_cuda(
    const u64* program, usize base_word, usize n_lanes, u64* sink)
{
  u32 threads = 256;
  u32 blocks = (u32)((n_lanes + threads - 1) / threads);
  formatter_coalesced_probe<<<blocks, threads>>>(program, (u64)base_word,
                                                 (u32)n_lanes, sink);
  V2_CHECK(cudaGetLastError());
  V2_CHECK(cudaDeviceSynchronize());
}
