// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <crates/ulib/includes.hpp>
#include <cstdio>
#include <cooperative_groups.h>

struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) {
    *this = *t;
  }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) {
    *this = *t;
  }
};

// --- Zenith Hardware Macro Implementations ---

struct alignas(8) DSP48E2_Subset {
    __device__ static void compute(
        u32 d,           // full 27-bit D input, right-aligned in the low bits
        u32 a,           // full 27-bit A input, right-aligned in the low bits
        u32 b,           // full 18-bit B input, right-aligned in the low bits
        uint64_t c_val,  // full 48-bit C input
        uint64_t prev_p, // the PREG's value from the previous clock edge
        u32 opmode,
        u32 alumode,
        u32 inmode,
        u32 cep,
        u32 rstp,
        uint64_t* next_p
    ) {
        if (rstp) {
            *next_p = 0;
            return;
        }
        if (!cep) {
            *next_p = prev_p & ((1ULL << 48) - 1);
            return;
        }
        // The PS subset supports ALUMODE=0000 and the real DSP48E2
        // OPMODE encodings C=0x030, M=0x005 and P+M=0x025.
        assert((alumode & 0xF) == 0);
        bool preadd = (inmode & 0x4) && !(inmode & 0x8);
        // Sign-extend the 27-bit two's-complement A/D fields (bit 26 is the sign bit)
        // by shifting the value up to the MSB of a 32-bit lane and back with an
        // arithmetic shift.
        int64_t a_val = (int32_t)(a << 5) >> 5;
        int64_t d_val = (int32_t)(d << 5) >> 5;
        int64_t ad_val = preadd ? (a_val + d_val) : a_val;
        // 2nd-audit D-02 fix: AD is a 27-bit register in real hardware, so
        // A+D must wrap/sign-extend back to 27 bits -- the sum of two
        // 27-bit signed values can need 28 bits, and that carry was
        // previously kept instead of being truncated away. Re-mask to the
        // low 27 bits and sign-extend again with the same shift trick used
        // on the inputs above.
        ad_val = (int64_t)(int32_t)(((uint32_t)ad_val & 0x7FFFFFF) << 5) >> 5;

        // Sign-extend the 18-bit signed B input the same way.
        int64_t b_val = (int32_t)(b << 14) >> 14;

        // Multiplier Logic
        int64_t m_val = ad_val * b_val;

        int64_t p_out;
        if (opmode == 0x030) { // C path
            p_out = (int64_t)c_val;
        } else if (opmode == 0x005) { // Multiply-only
            p_out = m_val;
        } else if (opmode == 0x025) { // Accumulate: P + M
            int64_t signed_p = (int64_t)(prev_p << 16) >> 16;
            p_out = signed_p + m_val;
        } else {
            // Unsupported controls are a netlist error, never a hold mode.
            assert(false);
            p_out = 0;
        }

        // Output is 48-bit
        *next_p = (uint64_t)p_out & ((1ULL << 48) - 1);
    }
};

struct alignas(4) CARRY4_Primitive {
    __device__ static void compute(u32 s, u32 di, u32 cin, u32 cyinit, u32* o, u32* co) {
        u32 c[5];
        c[0] = cyinit | cin;
        u32 out_o = 0;
        u32 out_co = 0;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            u32 s_i = (s >> i) & 1;
            u32 di_i = (di >> i) & 1;
            c[i+1] = (s_i & c[i]) | ((~s_i & 1) & di_i);
            out_o |= (s_i ^ c[i]) << i;
            out_co |= c[i+1] << i;
        }

        *o = out_o;
        *co = out_co;
    }
};

struct alignas(4) SRLC32E_Primitive {
    __device__ static void compute(u32 d, u32 ce, u32 rising_edge, u32 a, u32 current_state, u32* next_state, u32* q, u32* q31) {
        u32 state = current_state;
        if (rising_edge && ce) {
            state = (state << 1) | (d & 1);
        }
        // Q/Q31 are asynchronous taps. Once a rising edge updates storage,
        // the taps settle to the new value in that same HDL timestamp.
        *q = (state >> (a & 0x1F)) & 1;
        *q31 = (state >> 31) & 1;
        *next_state = state;
    }
};

__device__ void simulate_block_v1(
  const u32 *__restrict__ script,
  usize script_size,
  const u32 *__restrict__ input_state,
  u32 *__restrict__ output_state,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ shared_metadata,
  u32 *__restrict__ shared_writeouts,
  u32 *__restrict__ shared_state
  )
{
  int script_pi = 0;
  while(true) {
    VectorRead2 t2_1, t2_2;
    VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    script_pi += 256;
    t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
    __syncthreads();
    int num_stages = shared_metadata[0];
    if(!num_stages) {
      break;
    }
    int is_last_part = shared_metadata[1];
    int num_ios = shared_metadata[2];
    int io_offset = shared_metadata[3];
    int num_srams = shared_metadata[4];
    int sram_offset = shared_metadata[5];
    int num_global_read_rounds = shared_metadata[6];
    int num_output_duplicates = shared_metadata[7];
    // M-01/M-02 fix: these three used to only be read much later, inside
    // the "Heterogeneous Macro Evaluation" block below -- too late to use
    // for committing the gathered macro-input words into shared_writeouts
    // during the sram/duplicate commit step a few lines down. Reading them
    // here is safe: shared_metadata is fully loaded and __syncthreads()'d
    // before this point (see the metadata load a few lines up), same as
    // slots 0..7 above.
    int num_dsps = shared_metadata[8];
    int num_carry4s = shared_metadata[9];
    int num_srlc32es = shared_metadata[10];
    int macro_out_words = num_dsps * 2 + num_carry4s + num_srlc32es * 2;
    int macro_in_words = num_dsps * 5 + num_carry4s + num_srlc32es;
    // Local shared_writeouts layout, low to high:
    //   [0, num_normal_writeouts)                    ordinary AIG/DFF writeouts
    //   [num_normal_writeouts, +macro_out_words)      macro OUTPUTS (P, O/CO, Q/Q31+state)
    //   [..., +macro_in_words)                        macro INPUTS (gathered A/B/C/D/... operand words)
    //   [..., +num_output_duplicates)                 output-activation duplicates
    //   [..., +num_srams)                             SRAM read-data (num_ios - num_srams .. num_ios)
    // dup_start/macro_in_start are the boundaries between these regions;
    // flatten.rs's num_writeouts (== num_ios here) was extended to include
    // macro_in_words specifically so this arithmetic lines up on both sides.
    int dup_start = num_ios - num_srams - num_output_duplicates;
    int macro_in_start = dup_start - macro_in_words;
    int num_normal_writeouts = macro_in_start - macro_out_words;
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if(threadIdx.x % 2 == 0) {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    }
    else {
      writeout_hook_i = writeout_hook_i >> 16;
    }

    t4_1.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds) + threadIdx.x);
    t4_2.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4) + threadIdx.x);
    t4_3.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 2) + threadIdx.x);
    t4_4.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 3) + threadIdx.x);
    t4_5.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 4) + threadIdx.x);
    u32 t_global_rd_state = 0;
    for(int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
      u32 idx = t2_1.c1;
      u32 mask = t2_1.c2;
      script_pi += 256 * 2;
      t2_2.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }

      if(gr_i + 1 >= num_global_read_rounds) break;
      idx = t2_2.c1;
      mask = t2_2.c2;
      script_pi += 256 * 2;
      t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        if(idx >> 31) real_input_array = output_state - (1 << 31);
        else real_input_array = input_state;
        u32 value = real_input_array[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    __syncthreads();

    for(int bs_i = 0; bs_i < num_stages; ++bs_i) {
      u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;
#define GEMV1_SHUF_INPUT_K(k_outer, k_inner, t_shuffle) {           \
        u32 k = k_outer * 4 + k_inner;                              \
        u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);          \
        u32 t_shuffle_2_idx = t_shuffle >> 16;                      \
                                                                    \
        hier_input |= (shared_state[t_shuffle_1_idx >> 5] >>        \
                       (t_shuffle_1_idx & 31) & 1) << (k * 2);      \
        hier_input |= (shared_state[t_shuffle_2_idx >> 5] >>        \
                       (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);  \
      }
#define GEMV1_SHUF_INPUT_K_4(k_outer, t_shuffle) {    \
        GEMV1_SHUF_INPUT_K(k_outer, 0, t_shuffle.c1); \
        GEMV1_SHUF_INPUT_K(k_outer, 1, t_shuffle.c2); \
        GEMV1_SHUF_INPUT_K(k_outer, 2, t_shuffle.c3); \
        GEMV1_SHUF_INPUT_K(k_outer, 3, t_shuffle.c4); \
      }
      script_pi += 256 * 4 * 5;
      GEMV1_SHUF_INPUT_K_4(0, t4_1);
      t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(1, t4_2);
      t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(2, t4_3);
      t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(3, t4_4);
      t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_INPUT_K
#undef GEMV1_SHUF_INPUT_K_4
      hier_flag_xora = t4_5.c1;
      hier_flag_xorb = t4_5.c2;
      hier_flag_orb = t4_5.c3;
      t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      __syncthreads();

      // hier[0]
      if(threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
        shared_state[threadIdx.x] = ret;
      }
      __syncthreads();
      // hier[1..3]
      u32 tmp_cur_hi;
      for(int hi = 1; hi <= 3; ++hi) {
        int hier_width = 1 << (7 - hi);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          tmp_cur_hi = ret;
          shared_state[threadIdx.x] = ret;
        }
        __syncthreads();
      }
      // hier[4..7], within the first warp.
      if(threadIdx.x < 32) {
        for(int hi = 4; hi <= 7; ++hi) {
          int hier_width = 1 << (7 - hi);
          u32 hier_input_a = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
          u32 hier_input_b = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
          if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            tmp_cur_hi = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          }
        }
        u32 v1 = __shfl_down_sync(0xffffffff, tmp_cur_hi, 1);
        // hier[8..12]
        if(threadIdx.x == 0) {
          u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
          u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
          u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
          u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12 /* 0b1100 */;
          u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2 /* 0b10 */;
          tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
        }
        shared_state[threadIdx.x] = tmp_cur_hi;
      }
      __syncthreads();

      // write out
      if((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
      }
    }
    __syncthreads();

    // sram & duplicate permutation
    u32 sram_duplicate_t = 0;
#define GEMV1_SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                          \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);      \
      u32 t_shuffle_2_idx = t_shuffle >> 16;                  \
                                                              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>            \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>            \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);          \
    }
#define GEMV1_SHUF_SRAM_DUPL_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_SRAM_DUPL_K_4(0, t4_1);
    t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(1, t4_2);
    t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(2, t4_3);
    t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(3, t4_4);
    t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_SRAM_DUPL_K_4
#undef GEMV1_SHUF_SRAM_DUPL_K
    sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
    t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

    // sram read fires here.
    u32 *ram = nullptr;
    u32 r, w0;
    u32 port_w_addr_iv, port_w_wr_en, port_w_wr_data_iv;
    if(threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t;
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
        ? 0xffffffff : (0xffffffff >> (last_tid - num_srams * 4));
      port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);

      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13);
        // u32 sram_ed = sram_st + (1 << 13);
        u32 port_r_addr_iv = addrs & 0xffff;
        port_w_addr_iv = addrs >> 16;

        ram = sram_data + sram_st;
        r = ram[port_r_addr_iv];
        w0 = ram[port_w_addr_iv];
      }
    }
    // __syncthreads();

    // clock enable permutation
    u32 clken_perm = 0;
#define GEMV1_SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                      \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);  \
      u32 t_shuffle_2_idx = t_shuffle >> 16;              \
                                                          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>        \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>        \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);      \
    }
#define GEMV1_SHUF_CLKEN_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_CLKEN_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_CLKEN_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_CLKEN_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_CLKEN_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_CLKEN_K_4(0, t4_1);
    GEMV1_SHUF_CLKEN_K_4(1, t4_2);
    GEMV1_SHUF_CLKEN_K_4(2, t4_3);
    GEMV1_SHUF_CLKEN_K_4(3, t4_4);
#undef GEMV1_SHUF_CLKEN_K
#undef GEMV1_SHUF_CLKEN_K_4

    // sram commit
    if(threadIdx.x < num_srams * 4) {
      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_output_duplicates) {
      shared_writeouts[dup_start + (threadIdx.x - num_srams * 4)] = sram_duplicate_t;
    }
    // M-01 fix: this branch never existed before. flatten.rs already
    // generates valid gather permutation words for every macro input bit
    // (source thread range right after the output-activation duplicates,
    // see the matching macro_perm_base shift in flatten.rs), and every
    // thread in this range already computed a correct sram_duplicate_t
    // above -- it just never got copied into shared_writeouts, so the
    // macro evaluator below was reading whatever stale/unrelated contents
    // happened to be sitting at macro_in_start from a previous script
    // phase instead of the real gathered operand words.
    else if(threadIdx.x < num_srams * 4 + num_output_duplicates + macro_in_words) {
      shared_writeouts[macro_in_start + (threadIdx.x - num_srams * 4 - num_output_duplicates)] = sram_duplicate_t;
    }

    __syncthreads();

    // Heterogeneous Macro Evaluation
    // (num_dsps, num_carry4s, num_srlc32es, macro_out_words, macro_in_words,
    // dup_start, macro_in_start, num_normal_writeouts are all computed
    // earlier now -- see the M-01/M-02 comment above the metadata reads.)

    // Evaluate DSP48E2
    if (threadIdx.x < num_dsps) {
        int i = threadIdx.x;
        u32 d0 = shared_writeouts[macro_in_start + i * 5 + 0];
        u32 d1 = shared_writeouts[macro_in_start + i * 5 + 1];
        u32 d2 = shared_writeouts[macro_in_start + i * 5 + 2];
        u32 d3 = shared_writeouts[macro_in_start + i * 5 + 3];
        u32 d4 = shared_writeouts[macro_in_start + i * 5 + 4];

        u32 a = d0 & 0x7FFFFFF;        // full 27-bit A (was truncated to 16 bits)
        u32 b = (d0 >> 30) | ((d1 & 0xFFFF) << 2);
        uint64_t c_val = (d1 >> 16) | ((uint64_t)d2 << 16);
        u32 d_in = d3 & 0x7FFFFFF;     // full 27-bit D (was truncated to 16 bits)
        u32 opmode = (d3 >> 27) | ((d4 & 0xF) << 5);
        u32 alumode = (d4 >> 4) & 0xF;
        u32 inmode = (d4 >> 8) & 0x1F;
        u32 cep = (d4 >> 13) & 1;
        u32 rstp = (d4 >> 14) & 1;

        // PREG is the only clocked register in this subset: read back the P
        // value this DSP produced last cycle (already ping-ponged into
        // input_state by the generic writeout step below) so MAC mode
        // (state 2) accumulates onto the real previous P, not onto C.
        int dsp_out_idx = num_normal_writeouts + i * 2;
        uint64_t prev_p = (uint64_t)input_state[io_offset + dsp_out_idx]
            | ((uint64_t)input_state[io_offset + dsp_out_idx + 1] << 32);

        uint64_t next_p;
        DSP48E2_Subset::compute(
            d_in, a, b, c_val, prev_p, opmode, alumode, inmode, cep, rstp,
            &next_p);
        shared_writeouts[dsp_out_idx] = next_p & 0xFFFFFFFF;
        shared_writeouts[dsp_out_idx + 1] = next_p >> 32;
    }

    // Evaluate CARRY4
    if (threadIdx.x < num_carry4s) {
        int i = threadIdx.x;
        int carry4_start = macro_in_start + num_dsps * 5;
        u32 d0 = shared_writeouts[carry4_start + i];

        u32 di = (d0 & 1) | ((d0 >> 1) & 2) | ((d0 >> 2) & 4) | ((d0 >> 3) & 8);
        u32 s = ((d0 >> 1) & 1) | ((d0 >> 2) & 2) | ((d0 >> 3) & 4) | ((d0 >> 4) & 8);
        u32 cin = (d0 >> 8) & 1;
        u32 cyinit = (d0 >> 9) & 1;

        u32 o, co;
        CARRY4_Primitive::compute(s, di, cin, cyinit, &o, &co);
        int carry4_out_idx = num_normal_writeouts + num_dsps * 2 + i;
        shared_writeouts[carry4_out_idx] = o | (co << 4);
    }

    // Evaluate SRLC32E
    if (threadIdx.x < num_srlc32es) {
        int i = threadIdx.x;
        int srlc_start = macro_in_start + num_dsps * 5 + num_carry4s;
        u32 d0 = shared_writeouts[srlc_start + i];

        u32 d = d0 & 1;
        u32 ce = (d0 >> 1) & 1;
        u32 rising_edge = (d0 >> 2) & 1;
        u32 a = (d0 >> 3) & 0x1F;

        u32 next_state, q, q31;
        int srlc_out_idx = num_normal_writeouts + num_dsps * 2 + num_carry4s + i * 2;
        u32 current_state = input_state[io_offset + srlc_out_idx + 1];
        SRLC32E_Primitive::compute(d, ce, rising_edge, a, current_state, &next_state, &q, &q31);
        shared_writeouts[srlc_out_idx] = q | (q31 << 1);
        shared_writeouts[srlc_out_idx + 1] = next_state;
    }

    __syncthreads();
    u32 writeout_inv = shared_writeouts[threadIdx.x];

    clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
    writeout_inv ^= t4_5.c3;

    if(threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;
    }

    if(is_last_part) break;
  }
  assert(script_size == script_pi);
}

__global__ void simulate_v1_noninteractive_simple_scan(
  usize num_blocks,
  usize num_major_stages,
  const usize *__restrict__ blocks_start,
  const u32 *__restrict__ blocks_data,
  u32 *__restrict__ sram_data,
  usize num_cycles,
  usize state_size,
  u32 *__restrict__ states_noninteractive
  )
{
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 script_starts[32], script_sizes[32];
  assert(num_major_stages <= 32);
  if(threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] - script_starts[threadIdx.x];
  }
  __syncthreads();
  for(usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    for(usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(
        blocks_data + script_starts[stage_i],
        script_sizes[stage_i],
        states_noninteractive + cycle_i * state_size,
        states_noninteractive + (cycle_i + 1) * state_size,
        sram_data,
        shared_metadata, shared_writeouts, shared_state
        );
      cooperative_groups::this_grid().sync();
    }
  }
}
