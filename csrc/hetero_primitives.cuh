#pragma once

#include <crates/ulib/includes.hpp>
#include <cassert>
#include <cstdint>

struct alignas(8) DSP48E2_Subset {
  __device__ static void compute(
      u32 d, u32 a, u32 b, uint64_t c_val, uint64_t prev_p,
      u32 opmode, u32 alumode, u32 inmode, u32 cep, u32 rstp,
      uint64_t *next_p) {
    if (rstp) {
      *next_p = 0;
      return;
    }
    if (!cep) {
      *next_p = prev_p & ((1ULL << 48) - 1);
      return;
    }
    assert((alumode & 0xF) == 0);
    const bool preadd = (inmode & 0x4) && !(inmode & 0x8);
    int64_t a_val = (int32_t)(a << 5) >> 5;
    int64_t d_val = (int32_t)(d << 5) >> 5;
    int64_t ad_val = preadd ? (a_val + d_val) : a_val;
    ad_val = (int64_t)(int32_t)(((uint32_t)ad_val & 0x7FFFFFF) << 5) >> 5;
    const int64_t b_val = (int32_t)(b << 14) >> 14;
    const int64_t m_val = ad_val * b_val;

    int64_t p_out;
    if (opmode == 0x030) {
      p_out = (int64_t)c_val;
    } else if (opmode == 0x005) {
      p_out = m_val;
    } else if (opmode == 0x025) {
      const int64_t signed_p = (int64_t)(prev_p << 16) >> 16;
      p_out = signed_p + m_val;
    } else {
      assert(false);
      p_out = 0;
    }
    *next_p = (uint64_t)p_out & ((1ULL << 48) - 1);
  }
};

struct alignas(4) CARRY4_Primitive {
  __device__ static void compute(
      u32 s, u32 di, u32 cin, u32 cyinit, u32 *o, u32 *co) {
    u32 c[5];
    c[0] = cyinit | cin;
    u32 out_o = 0;
    u32 out_co = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const u32 s_i = (s >> i) & 1;
      const u32 di_i = (di >> i) & 1;
      c[i + 1] = (s_i & c[i]) | ((~s_i & 1) & di_i);
      out_o |= (s_i ^ c[i]) << i;
      out_co |= c[i + 1] << i;
    }
    *o = out_o;
    *co = out_co;
  }
};

struct alignas(4) SRLC32E_Primitive {
  __device__ static void compute(
      u32 d, u32 ce, u32 rising_edge, u32 a, u32 current_state,
      u32 *next_state, u32 *q, u32 *q31) {
    u32 state = current_state;
    if (rising_edge && ce) state = (state << 1) | (d & 1);
    *q = (state >> (a & 0x1F)) & 1;
    *q31 = (state >> 31) & 1;
    *next_state = state;
  }
};
