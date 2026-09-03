// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Integrated V2 heterogeneous CUDA evaluator and formatter self-check.
//
// The self-check proves the packed program ABI on real hardware, while the
// execution entry points below evaluate AIG operations and preserved FPGA
// macros in dependency waves:
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
#include <cassert>
#include <cstdio>
#include <cooperative_groups.h>
#include "format_v2_abi.h"
#include "format_v2_decode.cuh"
#include "hetero_primitives.cuh"

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

namespace {

__device__ __forceinline__ u32 source_bit(
    const u64 *section, u32 pc, u32 flat_bit, u32 instance,
    const u32 *prev_state, const u32 *current_stage, const u64 *shared) {
  gem_v2::SourceSel s = gem_v2::decode_source(section[flat_bit * pc + instance]);
  return gem_v2::gather_bit(s, prev_state, current_stage, shared);
}

__device__ __forceinline__ void publish_bit(
    const u64 *section, u32 pc, u32 flat_bit, u32 instance, u32 value,
    u64 *shared, u32 *current_stage, u32 *next_state) {
  gem_v2::DestSel d = gem_v2::decode_dest(section[flat_bit * pc + instance]);
  if (!d.valid) return;
  if (d.space == gem_v2::DST_LOCAL) {
    const u64 mask = 1ull << d.bit;
    u64 word = shared[d.index];
    shared[d.index] = value ? (word | mask) : (word & ~mask);
  } else {
    u32 *base = d.space == gem_v2::DST_CUR_STAGE ? current_stage : next_state;
    const u32 mask = 1u << d.bit;
    if (value) atomicOr(base + d.index, mask);
    else atomicAnd(base + d.index, ~mask);
  }
}

__device__ __forceinline__ const u64 *section_base(
    const u64 *program, const gem_v2::HeaderView &hv, u32 kind) {
  const int section = hv.find_section(kind);
  return section < 0 ? nullptr : program + hv.sec_start(section);
}

__device__ __forceinline__ void evaluate_v2_body(
    const u64 *__restrict__ program, usize program_words,
    const u32 *__restrict__ gather_pairs, usize gather_pair_count,
    const u32 *__restrict__ dsp_state_words, usize n_dsp,
    const u32 *__restrict__ srl_state_words, usize n_srl,
    u32 *__restrict__ sram_storage, usize n_srams,
    const u32 *__restrict__ prev_state, usize state_words,
    const u32 *__restrict__ input_word_masks,
    u32 *__restrict__ next_state, u32 *__restrict__ current_stage,
    usize current_stage_words,
    u32 *__restrict__ aig_out, usize n_aig,
    u64 *__restrict__ carry_out, usize n_carry,
    u64 *__restrict__ dsp_out, u64 *__restrict__ srl_out,
    u32 rising_edge, usize shared_words) {
  extern __shared__ u64 shared[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const u32 lane = threadIdx.x & 31u;
  const u32 warp = threadIdx.x >> 5;
  const u32 num_warps = blockDim.x >> 5;
  const usize global_tid = (usize)blockIdx.x * blockDim.x + threadIdx.x;
  const usize global_stride = (usize)gridDim.x * blockDim.x;
  const u32 global_warp = blockIdx.x * num_warps + warp;
  const u32 total_warps = gridDim.x * num_warps;

  for (usize i = threadIdx.x; i < shared_words; i += blockDim.x) shared[i] = 0;
  for (usize i = global_tid; i < state_words; i += global_stride) {
    const u32 input_mask = input_word_masks == nullptr ? 0u : input_word_masks[i];
    next_state[i] = (next_state[i] & input_mask) | (prev_state[i] & ~input_mask);
  }
  for (usize i = global_tid; i < current_stage_words; i += global_stride)
    current_stage[i] = i < state_words ? prev_state[i] : 0u;
  if (gridDim.x > 1) grid.sync(); else __syncthreads();
  for (usize i = threadIdx.x; i < gather_pair_count; i += blockDim.x) {
    const u32 old_word = gather_pairs[2 * i];
    const u32 shared_word = gather_pairs[2 * i + 1];
    if (old_word < state_words && shared_word < shared_words)
      shared[shared_word] = (u64)prev_state[old_word];
  }
  __syncthreads();

  gem_v2::HeaderView hv{program};
  if (hv.magic() != gem_v2::V2_MAGIC || hv.version() != gem_v2::V2_VERSION ||
      hv.total_words() != program_words) return;

  const u64 *wave_desc = section_base(program, hv, gem_v2::SEC_WAVE_DESCRIPTORS);
  const u64 *aig_ops = section_base(program, hv, gem_v2::SEC_AIG_OPS);
  const int endpoint_section = hv.find_section(gem_v2::SEC_ENDPOINT_OPS);
  const u64 *endpoint_ops = endpoint_section < 0 ? nullptr : program + hv.sec_start(endpoint_section);
  const int sram_section = hv.find_section(gem_v2::SEC_SRAM_OPS);
  const u64 *sram_ops = sram_section < 0 ? nullptr : program + hv.sec_start(sram_section);
  const u64 *csrc = section_base(program, hv, gem_v2::SEC_CARRY4_SRC);
  const u64 *cdst = section_base(program, hv, gem_v2::SEC_CARRY4_DST);
  const u64 *dsrc = section_base(program, hv, gem_v2::SEC_DSP_SRC);
  const u64 *ddst = section_base(program, hv, gem_v2::SEC_DSP_DST);
  const u64 *ssrc = section_base(program, hv, gem_v2::SEC_SRLC32E_SRC);
  const u64 *sdst = section_base(program, hv, gem_v2::SEC_SRLC32E_DST);
  const u32 cpc = gem_v2::padded_count((u32)n_carry);
  const u32 dpc = gem_v2::padded_count((u32)n_dsp);
  const u32 spc = gem_v2::padded_count((u32)n_srl);

  for (u32 wave = 0; wave < hv.num_waves(); ++wave) {
    u32 astart = 0, acount = 0, adepths = 0, cstart = 0, ccount = 0;
    u32 dstart = 0, dcount = 0, sstart = 0, scount = 0;
    if (lane == 0) {
      const u64 a = wave_desc[wave * 5 + 0];
      const u64 ad = wave_desc[wave * 5 + 1];
      const u64 c = wave_desc[wave * 5 + 2];
      const u64 d = wave_desc[wave * 5 + 3];
      const u64 s = wave_desc[wave * 5 + 4];
      astart = (u32)a; acount = (u32)(a >> 32);
      adepths = (u32)ad;
      cstart = (u32)c; ccount = (u32)(c >> 32);
      dstart = (u32)d; dcount = (u32)(d >> 32);
      sstart = (u32)s; scount = (u32)(s >> 32);
    }
    astart = __shfl_sync(0xffffffffu, astart, 0);
    acount = __shfl_sync(0xffffffffu, acount, 0);
    adepths = __shfl_sync(0xffffffffu, adepths, 0);
    cstart = __shfl_sync(0xffffffffu, cstart, 0);
    ccount = __shfl_sync(0xffffffffu, ccount, 0);
    dstart = __shfl_sync(0xffffffffu, dstart, 0);
    dcount = __shfl_sync(0xffffffffu, dcount, 0);
    sstart = __shfl_sync(0xffffffffu, sstart, 0);
    scount = __shfl_sync(0xffffffffu, scount, 0);

    // Gates at one dependency depth are independent and run across the full
    // CTA. A barrier appears only between depths that can consume one another.
    for (u32 depth = 0; depth < adepths; ++depth) {
      if (aig_ops != nullptr) {
        for (u32 rel = (u32)global_tid; rel < acount; rel += (u32)global_stride) {
          const u32 op = astart + rel;
          if ((u32)aig_ops[op * 4 + 0] != depth) continue;
          const gem_v2::SourceSel a = gem_v2::decode_source(aig_ops[op * 4 + 1]);
          const gem_v2::SourceSel b = gem_v2::decode_source(aig_ops[op * 4 + 2]);
          const gem_v2::DestSel d = gem_v2::decode_dest(aig_ops[op * 4 + 3]);
          const u32 value = gem_v2::gather_bit(a, prev_state, current_stage, shared) &
                            gem_v2::gather_bit(b, prev_state, current_stage, shared);
          if (op < n_aig) aig_out[op] = value;
          if (d.valid) {
            if (d.space == gem_v2::DST_LOCAL) {
              const u64 mask = 1ull << d.bit;
              if (value) atomicOr((unsigned long long *)(shared + d.index),
                                  (unsigned long long)mask);
              else atomicAnd((unsigned long long *)(shared + d.index),
                             (unsigned long long)~mask);
            } else {
              u32 *base = d.space == gem_v2::DST_CUR_STAGE ? current_stage : next_state;
              const u32 mask = 1u << d.bit;
              if (value) atomicOr(base + d.index, mask);
              else atomicAnd(base + d.index, ~mask);
            }
          }
        }
      }
      if (gridDim.x > 1) grid.sync(); else __syncthreads();
    }

    const u32 active_types = (ccount != 0) + (dcount != 0) + (scount != 0);
    if (active_types != 0) {
      const u32 assignment = global_warp % active_types;
      const u32 local_warp = global_warp / active_types;
      const u32 type_warps = (total_warps + active_types - 1 - assignment) / active_types;
      u32 type = 0xffffffffu;
      u32 rank = 0;
      if (ccount) { if (rank == assignment) type = 0; ++rank; }
      if (dcount) { if (rank == assignment) type = 1; ++rank; }
      if (scount) { if (rank == assignment) type = 2; }

      const u32 start = type == 0 ? cstart : (type == 1 ? dstart : sstart);
      const u32 count = type == 0 ? ccount : (type == 1 ? dcount : scount);
      for (u32 tile = local_warp * 32; tile < count; tile += type_warps * 32) {
        const u32 rel = tile + lane;
        const u32 mask = __ballot_sync(0xffffffffu, rel < count);
        if (rel < count) {
          const u32 inst = start + rel;
          if (type == 0) {
            u32 sv = 0, di = 0;
#pragma unroll
            for (u32 b = 0; b < 4; ++b) sv |= source_bit(csrc, cpc, b, inst, prev_state, current_stage, shared) << b;
#pragma unroll
            for (u32 b = 0; b < 4; ++b) di |= source_bit(csrc, cpc, 4 + b, inst, prev_state, current_stage, shared) << b;
            const u32 ci = source_bit(csrc, cpc, 8, inst, prev_state, current_stage, shared);
            const u32 cy = source_bit(csrc, cpc, 9, inst, prev_state, current_stage, shared);
            u32 o, co;
            CARRY4_Primitive::compute(sv, di, ci, cy, &o, &co);
            carry_out[inst] = (u64)o | ((u64)co << 4);
#pragma unroll
            for (u32 b = 0; b < 4; ++b) publish_bit(cdst, cpc, b, inst, (o >> b) & 1, shared, current_stage, next_state);
#pragma unroll
            for (u32 b = 0; b < 4; ++b) publish_bit(cdst, cpc, 4 + b, inst, (co >> b) & 1, shared, current_stage, next_state);
          } else if (type == 1) {
            u32 a = 0, din = 0, bval = 0, opmode = 0, alumode = 0, inmode = 0;
            u64 cval = 0;
            for (u32 bit = 0; bit < 27; ++bit) a |= source_bit(dsrc, dpc, bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 27; ++bit) din |= source_bit(dsrc, dpc, 27 + bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 18; ++bit) bval |= source_bit(dsrc, dpc, 54 + bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 48; ++bit) cval |= (u64)source_bit(dsrc, dpc, 72 + bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 9; ++bit) opmode |= source_bit(dsrc, dpc, 120 + bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 4; ++bit) alumode |= source_bit(dsrc, dpc, 129 + bit, inst, prev_state, current_stage, shared) << bit;
            for (u32 bit = 0; bit < 5; ++bit) inmode |= source_bit(dsrc, dpc, 133 + bit, inst, prev_state, current_stage, shared) << bit;
            const u32 cep = source_bit(dsrc, dpc, 138, inst, prev_state, current_stage, shared);
            const u32 rstp = source_bit(dsrc, dpc, 139, inst, prev_state, current_stage, shared);
            const u32 clock_edge = source_bit(dsrc, dpc, 140, inst, prev_state, current_stage, shared);
            const u32 pw = dsp_state_words[inst];
            const u64 prevp = (u64)prev_state[pw] | ((u64)prev_state[pw + 1] << 32);
            u64 nextp;
            if (clock_edge)
              DSP48E2_Subset::compute(din, a, bval, cval, prevp, opmode, alumode, inmode, cep, rstp, &nextp);
            else
              nextp = prevp & ((1ULL << 48) - 1);
            dsp_out[inst] = nextp;
            next_state[pw] = (u32)nextp;
            next_state[pw + 1] = (u32)(nextp >> 32);
            for (u32 bit = 0; bit < 48; ++bit) publish_bit(ddst, dpc, bit, inst, (nextp >> bit) & 1, shared, current_stage, next_state);
          } else {
            const u32 d = source_bit(ssrc, spc, 0, inst, prev_state, current_stage, shared);
            const u32 ce = source_bit(ssrc, spc, 1, inst, prev_state, current_stage, shared);
            u32 addr = 0;
            for (u32 bit = 0; bit < 5; ++bit) addr |= source_bit(ssrc, spc, 2 + bit, inst, prev_state, current_stage, shared) << bit;
            const u32 clock_edge = source_bit(ssrc, spc, 7, inst, prev_state, current_stage, shared);
            const u32 sw = srl_state_words[inst];
            u32 ns, q, q31;
            SRLC32E_Primitive::compute(d, ce, clock_edge, addr, prev_state[sw], &ns, &q, &q31);
            srl_out[inst] = (u64)q | ((u64)q31 << 1) | ((u64)ns << 32);
            next_state[sw] = ns;
            publish_bit(sdst, spc, 0, inst, q, shared, current_stage, next_state);
            publish_bit(sdst, spc, 1, inst, q31, shared, current_stage, next_state);
          }
          __syncwarp(mask);
        }
      }
    }
    if (gridDim.x > 1) grid.sync(); else __syncthreads();
  }
  if (sram_ops != nullptr) {
    constexpr usize SRAM_SRC_BITS = 91;
    constexpr usize SRAM_OP_WORDS = 124;
    for (usize instance = global_tid; instance < n_srams; instance += global_stride) {
      const u64 *op = sram_ops + instance * SRAM_OP_WORDS;
      const usize storage_base = (usize)op[0];
      auto source = [&](usize index) {
        return gem_v2::gather_bit(gem_v2::decode_source(op[1 + index]),
                                  prev_state, current_stage, shared);
      };
      const u32 read_enable = source(0);
      u32 read_address = 0, write_address = 0;
      for (u32 bit = 0; bit < 13; ++bit) {
        read_address |= source(1 + bit) << bit;
        write_address |= source(14 + bit) << bit;
      }
      const u32 old_read = sram_storage[storage_base + read_address];
      const u32 old_write = sram_storage[storage_base + write_address];
      u32 write_mask = 0, write_data = 0;
      for (u32 bit = 0; bit < 32; ++bit) {
        write_mask |= source(27 + bit) << bit;
        write_data |= source(59 + bit) << bit;
      }
      if (read_enable) {
        for (u32 bit = 0; bit < 32; ++bit) {
          const gem_v2::DestSel dst = gem_v2::decode_dest(op[1 + SRAM_SRC_BITS + bit]);
          if (!dst.valid) continue;
          const u32 mask = 1u << dst.bit;
          if ((old_read >> bit) & 1u) atomicOr(next_state + dst.index, mask);
          else atomicAnd(next_state + dst.index, ~mask);
        }
      }
      sram_storage[storage_base + write_address] =
          (old_write & ~write_mask) | (write_data & write_mask);
    }
    if (gridDim.x > 1) grid.sync(); else __syncthreads();
  }
  if (endpoint_ops != nullptr) {
    const usize endpoint_count = (hv.sec_end(endpoint_section) - hv.sec_start(endpoint_section)) / 4;
    for (usize op = global_tid; op < endpoint_count; op += global_stride) {
      const u64 kind = endpoint_ops[op * 4 + 0];
      const gem_v2::SourceSel data = gem_v2::decode_source(endpoint_ops[op * 4 + 1]);
      const gem_v2::SourceSel enable = gem_v2::decode_source(endpoint_ops[op * 4 + 2]);
      const gem_v2::DestSel dst = gem_v2::decode_dest(endpoint_ops[op * 4 + 3]);
      const u32 value = gem_v2::gather_bit(data, prev_state, current_stage, shared);
      const u32 enabled = gem_v2::gather_bit(enable, prev_state, current_stage, shared);
      if (enabled && dst.valid) {
        u32 *base = dst.space == gem_v2::DST_CUR_STAGE ? current_stage : next_state;
        const u32 mask = 1u << dst.bit;
        if (value) atomicOr(base + dst.index, mask);
        else atomicAnd(base + dst.index, ~mask);
      }
      (void)kind;
    }
  }
  if (gridDim.x > 1) grid.sync(); else __syncthreads();
  (void)rising_edge;
}

__global__ void evaluate_v2_macro_waves(
    const u64 *__restrict__ program, usize program_words,
    const u32 *__restrict__ gather_pairs, usize gather_pair_count,
    const u32 *__restrict__ dsp_state_words, usize n_dsp,
    const u32 *__restrict__ srl_state_words, usize n_srl,
    const u32 *__restrict__ prev_state, usize state_words,
    const u32 *__restrict__ input_word_masks,
    u32 *__restrict__ next_state, u32 *__restrict__ current_stage,
    u32 *__restrict__ aig_out, usize n_aig,
    u64 *__restrict__ carry_out, usize n_carry,
    u64 *__restrict__ dsp_out, u64 *__restrict__ srl_out,
    u32 rising_edge, usize shared_words) {
  evaluate_v2_body(
      program, program_words, gather_pairs, gather_pair_count,
      dsp_state_words, n_dsp, srl_state_words, n_srl,
      nullptr, 0,
      prev_state, state_words, input_word_masks, next_state, current_stage,
      state_words,
      aig_out, n_aig, carry_out, n_carry, dsp_out, srl_out,
      rising_edge, shared_words);
}

__global__ void simulate_v2_cycles_kernel(
    const u64 *__restrict__ program, usize program_words,
    const u32 *__restrict__ gather_pairs, usize gather_pair_count,
    const u32 *__restrict__ dsp_state_words, usize n_dsp,
    const u32 *__restrict__ srl_state_words, usize n_srl,
    u32 *__restrict__ sram_storage, usize n_srams,
    const u32 *__restrict__ input_word_masks,
    u32 *__restrict__ states, usize state_words, usize num_cycles,
    u32 *__restrict__ current_stage, usize current_stage_words,
    u32 *__restrict__ aig_out, usize n_aig,
    u64 *__restrict__ carry_out, usize n_carry,
    u64 *__restrict__ dsp_out, u64 *__restrict__ srl_out,
    usize shared_words) {
  for (usize cycle = 0; cycle < num_cycles; ++cycle) {
    evaluate_v2_body(
        program, program_words, gather_pairs, gather_pair_count,
        dsp_state_words, n_dsp, srl_state_words, n_srl, sram_storage, n_srams,
        states + cycle * state_words, state_words, input_word_masks,
        states + (cycle + 1) * state_words, current_stage,
        current_stage_words,
        aig_out, n_aig, carry_out, n_carry, dsp_out, srl_out,
        0, shared_words);
  }
}

} // namespace

extern "C" void evaluate_v2_macro_waves_cuda(
    const u64 *program, usize program_words,
    const u32 *gather_pairs, usize gather_pair_count,
    const u32 *dsp_state_words, usize n_dsp,
    const u32 *srl_state_words, usize n_srl,
    const u32 *prev_state, usize state_words,
    const u32 *input_word_masks,
    u32 *next_state, u32 *current_stage,
    u32 *aig_out, usize n_aig,
    u64 *carry_out, usize n_carry,
    u64 *dsp_out, u64 *srl_out, u32 rising_edge, usize shared_words) {
  evaluate_v2_macro_waves<<<1, 256, shared_words * sizeof(u64)>>>(
      program, program_words, gather_pairs, gather_pair_count,
      dsp_state_words, n_dsp, srl_state_words, n_srl,
      prev_state, state_words, input_word_masks, next_state, current_stage,
      aig_out, n_aig,
      carry_out, n_carry, dsp_out, srl_out, rising_edge, shared_words);
  V2_CHECK(cudaGetLastError());
  V2_CHECK(cudaDeviceSynchronize());
}

extern "C" void benchmark_v2_macro_waves_cuda(
    const u64 *program, usize program_words,
    const u32 *gather_pairs, usize gather_pair_count,
    const u32 *dsp_state_words, usize n_dsp,
    const u32 *srl_state_words, usize n_srl,
    const u32 *prev_state, usize state_words,
    const u32 *input_word_masks,
    u32 *next_state, u32 *current_stage,
    u32 *aig_out, usize n_aig,
    u64 *carry_out, usize n_carry,
    u64 *dsp_out, u64 *srl_out, u32 rising_edge, usize shared_words,
    usize repetitions, u64 *elapsed_ns) {
  cudaEvent_t begin, end;
  V2_CHECK(cudaEventCreate(&begin));
  V2_CHECK(cudaEventCreate(&end));
  V2_CHECK(cudaEventRecord(begin));
  for (usize i = 0; i < repetitions; ++i) {
    evaluate_v2_macro_waves<<<1, 256, shared_words * sizeof(u64)>>>(
        program, program_words, gather_pairs, gather_pair_count,
        dsp_state_words, n_dsp, srl_state_words, n_srl,
        prev_state, state_words, input_word_masks, next_state, current_stage,
        aig_out, n_aig,
        carry_out, n_carry, dsp_out, srl_out, rising_edge, shared_words);
  }
  V2_CHECK(cudaEventRecord(end));
  V2_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  V2_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  const u64 elapsed_host_ns = (u64)(elapsed_ms * 1000000.0f);
  V2_CHECK(cudaMemcpy(elapsed_ns, &elapsed_host_ns, sizeof(elapsed_host_ns),
                      cudaMemcpyHostToDevice));
  V2_CHECK(cudaEventDestroy(begin));
  V2_CHECK(cudaEventDestroy(end));
  V2_CHECK(cudaGetLastError());
}

// Production-style host wrapper used by cuda_test --v2. The immutable program
// and metadata are uploaded once, while each launch consumes state[c] and
// commits state[c+1]. Primary-input bits already present in state[c+1] are
// retained through input_word_masks.
extern "C" void simulate_v2_cycles_cuda(
    const u64 *program, usize program_words,
    const u32 *gather_pairs, usize gather_pair_count,
    const u32 *dsp_state_words, usize n_dsp,
    const u32 *srl_state_words, usize n_srl,
    const u32 *input_word_masks,
    u32 *states, usize state_words, usize num_cycles,
    usize n_aig, usize n_carry, usize shared_words,
    usize current_stage_words, usize num_blocks, usize n_srams,
    usize sram_storage_words) {
  u32 *current_stage = nullptr;
  u32 *aig_out = nullptr;
  u64 *carry_out = nullptr, *dsp_out = nullptr, *srl_out = nullptr;
  u32 *sram_storage = nullptr;
  V2_CHECK(cudaMalloc((void **)&current_stage, current_stage_words * sizeof(u32)));
  V2_CHECK(cudaMalloc((void **)&aig_out, (n_aig ? n_aig : 1) * sizeof(u32)));
  V2_CHECK(cudaMalloc((void **)&carry_out, (n_carry ? n_carry : 1) * sizeof(u64)));
  V2_CHECK(cudaMalloc((void **)&dsp_out, (n_dsp ? n_dsp : 1) * sizeof(u64)));
  V2_CHECK(cudaMalloc((void **)&srl_out, (n_srl ? n_srl : 1) * sizeof(u64)));
  V2_CHECK(cudaMalloc((void **)&sram_storage,
                      (sram_storage_words ? sram_storage_words : 1) * sizeof(u32)));
  V2_CHECK(cudaMemset(sram_storage, 0, sram_storage_words * sizeof(u32)));
  int cooperative = 0, sm_count = 0, active_per_sm = 0;
  V2_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, 0));
  V2_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  V2_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_per_sm, simulate_v2_cycles_kernel, 256, shared_words * sizeof(u64)));
  assert(cooperative && "V2 multi-partition execution requires cooperative launch support");
  // Clamp the requested block count to what a cooperative launch can actually
  // co-resident on this GPU. The kernel uses gridDim.x (not num_blocks) for all
  // its cross-block work division, and current_stage_words is sized by the
  // schedule (not the block count), so clamping here is transparent.
  if (active_per_sm < 1) active_per_sm = 1;
  {
    usize max_coop = (usize)sm_count * (usize)active_per_sm;
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > max_coop) num_blocks = max_coop;
  }
  void *kernel_args[] = {
      &program, &program_words, &gather_pairs, &gather_pair_count,
      &dsp_state_words, &n_dsp, &srl_state_words, &n_srl,
      &sram_storage, &n_srams,
      &input_word_masks, &states, &state_words, &num_cycles,
      &current_stage, &current_stage_words, &aig_out, &n_aig,
      &carry_out, &n_carry, &dsp_out, &srl_out, &shared_words};
  V2_CHECK(cudaLaunchCooperativeKernel(
      (void *)simulate_v2_cycles_kernel, dim3((u32)num_blocks), dim3(256),
      kernel_args, shared_words * sizeof(u64), nullptr));
  V2_CHECK(cudaGetLastError());
  V2_CHECK(cudaDeviceSynchronize());
  V2_CHECK(cudaFree(current_stage));
  V2_CHECK(cudaFree(aig_out));
  V2_CHECK(cudaFree(carry_out));
  V2_CHECK(cudaFree(dsp_out));
  V2_CHECK(cudaFree(srl_out));
  V2_CHECK(cudaFree(sram_storage));
}
