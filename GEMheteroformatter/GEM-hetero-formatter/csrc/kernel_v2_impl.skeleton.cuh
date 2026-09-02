// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ============================ SKELETON ONLY ============================
// Path B (full V2) level-driven heterogeneous interpreter. This file shows
// STRUCTURE and the partial-warp-mask discipline; it is NOT compiled or GPU-
// verified. Bring-up gates before it may be cited in the report:
//   * compiles under the repo's nvcc flags (build.rs), cc >= 7.0
//   * compute-sanitizer memcheck + racecheck + synccheck clean
//   * independent HDL oracle == CPU V2 == this kernel on every tests/hetero
//     fixture incl. carry_chain2/8.sv
// =====================================================================
//
// Consumes the V2 stream serialized from `src/schedule.rs::HeteroSchedule`:
//
//   struct WaveDesc {           // one per (partition, level)
//     u32 n_aig_regions;        // AIG-region sub-scripts follow inline (V1 fold)
//     u32 n_carry4, off_carry4; // indices into the SoA macro buffer
//     u32 n_dsp,    off_dsp;
//     u32 n_srl,    off_srl;
//     u32 publish_flags;        // bit0 need __syncthreads after this wave
//                               // bit1 need grid.sync (cross-block consumer)
//   };
//
// The SoA macro buffer is `macro_layout.rs::MacroDeviceLayout` uploaded as
// `const u64* macro_data`. Field array base:
//   macro_data[ header[off_field] + instance ]      // one u64 per lane, coalesced

#include "kernel_v1_impl.cuh"   // reuse VectorRead4, the Boomerang fold, hetero_macros

// ---- fixed-width primitive math: reuse, do NOT fork ----
// DSP48E2_Subset::compute / CARRY4_Primitive::compute / SRLC32E_Primitive::compute
// already live in kernel_v1_impl.cuh and match src/primitive_models.rs.

namespace cg = cooperative_groups;

// Evaluate one type-homogeneous CARRY4 queue for the current wave.
// `base_in` / `base_out` are u64* into the SoA In / Out field arrays.
__device__ __forceinline__ void v2_eval_carry4_wave(
    const u64* __restrict__ base_in, u64* __restrict__ base_out, u32 n)
{
    const u32 warp = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31;
    const u32 warps = blockDim.x >> 5;

    // warp-stride over instances; every collective uses the EXACT active mask.
    for (u32 tile = warp * 32; tile < n; tile += warps * 32) {
        u32 i = tile + lane;
        u32 active = __ballot_sync(0xffffffff, i < n);   // full-warp ballot ok: every lane reaches here
        if (i >= n) continue;

        u64 din = base_in[i];                            // coalesced 256B load
        u32 s      = (u32)(din >> 0) & 0xF;
        u32 di     = (u32)(din >> 4) & 0xF;
        u32 cin    = (u32)(din >> 8) & 1;
        u32 cyinit = (u32)(din >> 9) & 1;

        u32 o, co;
        CARRY4_Primitive::compute(s, di, cin, cyinit, &o, &co);
        base_out[i] = (u64)(o | (co << 4));

        // intra-warp cascade CO[3]->CIN of the NEXT lane is a schedule edge,
        // not an intra-wave dependency: schedule.rs guarantees chained CARRY4s
        // land in different waves, so no __shfl of `co` is needed here. If a
        // future relaxation co-schedules a cascade, exchange it with:
        //   u32 up_co = __shfl_up_sync(active, co, 1);
        (void)active;
    }
}

__device__ __forceinline__ void v2_eval_dsp_wave(
    const u64* __restrict__ A, const u64* __restrict__ D, const u64* __restrict__ B,
    const u64* __restrict__ C, const u64* __restrict__ prevP, const u64* __restrict__ Ctl,
    u64* __restrict__ nextP, u32 n)
{
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (u32 tile = warp * 32; tile < n; tile += warps * 32) {
        u32 i = tile + lane;
        if (i >= n) continue;
        u32 a = (u32)A[i], d = (u32)D[i], b = (u32)B[i];
        u64 c = C[i], pp = prevP[i];
        u32 ctl = (u32)Ctl[i];
        u32 op_state = ctl & 0x3;           // 0=C, 1=M, 2=P+M  (host pre-decoded OPMODE)
        u32 preadd   = (ctl >> 2) & 1;
        u32 cep      = (ctl >> 3) & 1;
        u32 rstp     = (ctl >> 4) & 1;

        // adapt to DSP48E2_Subset::compute's real-OPMODE signature:
        u32 opmode  = op_state == 0 ? 0x030 : (op_state == 1 ? 0x005 : 0x025);
        u32 inmode  = preadd ? 0x4 : 0x0;
        u64 np;
        DSP48E2_Subset::compute(d, a, b, c, pp, opmode, /*alumode=*/0, inmode, cep, rstp, &np);
        nextP[i] = np;
    }
}

__device__ __forceinline__ void v2_eval_srl_wave(
    const u64* __restrict__ In, u64* __restrict__ Out, u64* __restrict__ Storage,
    u32 rising_edge, u32 n)
{
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (u32 tile = warp * 32; tile < n; tile += warps * 32) {
        u32 i = tile + lane;
        if (i >= n) continue;
        u64 din = In[i];
        u32 d  = (u32)(din >> 0) & 1;
        u32 ce = (u32)(din >> 1) & 1;
        u32 a  = (u32)(din >> 3) & 0x1F;
        u32 cur = (u32)Storage[i];
        u32 nxt, q, q31;
        SRLC32E_Primitive::compute(d, ce, rising_edge, a, cur, &nxt, &q, &q31);
        Out[i] = (u64)(q | (q31 << 1));
        Storage[i] = (u64)nxt;             // committed at the cycle boundary by scatter
    }
}

// One partition, one cycle: iterate dependency waves.
__device__ void simulate_partition_v2(
    const u32* __restrict__ script, u32 script_size,
    const u64* __restrict__ macro_data, const u64* __restrict__ header,
    const u32* __restrict__ input_state, u32* __restrict__ output_state,
    u32 rising_edge,
    u32* shared_metadata, u32* shared_writeouts, u32* shared_state)
{
    u32 pi = 0;
    u32 num_waves = script[pi++];
    for (u32 w = 0; w < num_waves; ++w) {
        // --- WaveDesc ---
        u32 n_aig = script[pi++];
        u32 n_c4  = script[pi++], off_c4 = script[pi++];
        u32 n_dsp = script[pi++], off_dsp = script[pi++];
        u32 n_srl = script[pi++], off_srl = script[pi++];
        u32 flags = script[pi++];

        // --- 1. AIG regions: existing 256-thread Boomerang fold, inline ---
        for (u32 r = 0; r < n_aig; ++r) {
            // pi advanced by the V1 fold; shared_state / shared_writeouts as V1
            pi = boomerang_fold_one_region(script, pi, input_state, output_state,
                                           shared_metadata, shared_writeouts, shared_state);
        }
        __syncthreads();   // AIG outputs of this wave visible before macros read them

        // --- 2. type-homogeneous macro queues ---
        if (n_c4) {
            const u64* in  = macro_data + header[off_c4 + 0] /*In*/;
            u64* out       = (u64*)macro_data + header[off_c4 + 1] /*Out*/;
            v2_eval_carry4_wave(in, out, n_c4);
        }
        if (n_dsp) {
            const u64* A = macro_data + header[off_dsp + 0];
            // ... D,B,C,prevP,Ctl,nextP at off_dsp+1..6
            v2_eval_dsp_wave(A, A + n_dsp, A + 2*n_dsp, A + 3*n_dsp, A + 4*n_dsp,
                             A + 5*n_dsp, (u64*)A + 6*n_dsp, n_dsp);
        }
        if (n_srl) {
            const u64* in = macro_data + header[off_srl + 0];
            v2_eval_srl_wave(in, (u64*)in + n_srl, (u64*)in + 2*n_srl, rising_edge, n_srl);
        }

        // --- 3. publish ---
        // macro results land in the SoA Out arrays (global). A later same-block
        // wave that reads them needs them visible:
        if (flags & 1u) __syncthreads();
        // a later-partition (cross-block) consumer needs the grid barrier:
        if (flags & 2u) cg::this_grid().sync();
    }
    // assert(pi == script_size);   // enable in debug builds
    (void)script_size;
}
