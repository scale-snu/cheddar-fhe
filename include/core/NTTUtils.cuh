#pragma once

#include "common/Basic.cuh"
#include "core/NTT.h"

namespace cheddar {
namespace kernel {

/**
 * @brief Compute a = a + bw, b = a - bw
 *
 * @tparam word
 * @param a in range [-q, q)
 * @param b in range [-q, q)
 * @param w in range [0, q)
 * @param q (prime number)
 * @param q_inv (montgomery const)
 */
template <typename word>
__device__ __inline__ void ButterflyNTT(make_signed_t<word> &a,
                                        make_signed_t<word> &b, const word w,
                                        const word q,
                                        const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;

  if (a < 0) a += q;  // [0, q) range

  signed_word mult = basic::detail::__mult_montgomery_lazy<word>(
      b, static_cast<signed_word>(w), q, q_inv);  // (-q, q)
  if (mult < 0) mult += q;                        // [0, q) range

  b = a - mult;        // (-q, q)
  a = (a - q) + mult;  // [-q, q)
}

/**
 * @brief Compute a = (a + b), b = (a - b)w
 *
 * @tparam word
 * @param a in range [-q, q)
 * @param b in range [-q, q)
 * @param w in range [0, q)
 * @param q (prime number)
 * @param q_inv (montgomery const)
 */
template <typename word>
__device__ __inline__ void ButterflyINTT(make_signed_t<word> &a,
                                         make_signed_t<word> &b, const word w,
                                         const word q,
                                         const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;

  if (a < 0) a += q;         // [0, q)
  if (b < 0) b += q;         // [0, q)
  signed_word diff = a - b;  // (-q, q)
  a = (a - q) + b;           // [-q, q)
  b = basic::detail::__mult_montgomery_lazy<word>(
      diff, static_cast<signed_word>(w), q,
      q_inv);  // (-q, q)
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTTFirst(make_signed_t<word> *local,
                                              int tw_idx, const word *w,
                                              const word q,
                                              const make_signed_t<word> q_inv) {
  // Recursive call
  if constexpr (stage > 1) {
    MultiRadixNTTFirst<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
  }

  constexpr int num_tw = 1 << (stage - 1);
  constexpr int stride = radix / (1 << stage);

  // Load twiddle factors
  word w_vec[num_tw];
  basic::VectorizedMove<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));

#pragma unroll
  for (int i = 0; i < num_tw; i++) {
#pragma unroll
    for (int j = 0; j < stride; j++) {
      ButterflyNTT(local[i * 2 * stride + j],
                   local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
    }
  }
}
template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTTLast(make_signed_t<word> *local,
                                              int tw_idx, const word *w,
                                              const word q,
                                              const make_signed_t<word> q_inv) {
  constexpr int num_tw = 1 << (stage - 1);
  constexpr int stride = radix / (1 << stage);

  // Load twiddle factors
  word w_vec[num_tw];
  basic::VectorizedMove<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));

#pragma unroll
  for (int i = 0; i < num_tw; i++) {
#pragma unroll
    for (int j = 0; j < stride; j++) {
      ButterflyINTT(local[i * 2 * stride + j],
                    local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
    }
  }

  // Recursive call
  if constexpr (stage > 1) {
    MultiRadixINTTLast<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
  }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTT(make_signed_t<word> *local, int tw_idx,
                                         const word *w, const word q,
                                         const make_signed_t<word> q_inv) {
  constexpr int num_tw = radix / (1 << stage);
  constexpr int stride = (1 << (stage - 1));

  // Load twiddle factors
  word w_vec[num_tw];
  basic::VectorizedMove<word, num_tw>(w_vec, w + tw_idx);

#pragma unroll
  for (int i = 0; i < num_tw; i++) {
#pragma unroll
    for (int j = 0; j < stride; j++) {
      ButterflyNTT(local[i * 2 * stride + j],
                   local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
    }
  }

  // Recursive call
  if constexpr (stage > 1) {
    MultiRadixNTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
  }
}
template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixNTT_OT(make_signed_t<word> *local,
                                            int tw_idx, const word *w,
                                            const word *w_msb, const word q,
                                            const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  int last_tw_idx = (1 << (stage - 1)) * tw_idx;
  int msbIdx = last_tw_idx / lsb_size;
  int lsbIdx = last_tw_idx % lsb_size;

  constexpr int num_outer_blocks = radix / (1 << stage);  // 2
  constexpr int accumed_tw_num = (1 << (stage)) - 1;      // 3
  word twiddle_factor_set[accumed_tw_num * num_outer_blocks];

  constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
  constexpr int offset = ((1 << (stage - 1)) - 1) * num_outer_blocks;
  // load last twiddle factor
  if constexpr (kExtendedOT) {
#pragma unroll
    for (int i = 0; i < num_tw_factor; i++) {
      twiddle_factor_set[i + offset] =
          basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
    }

    // compute all twiddle factors with the last stage twiddle factor
#pragma unroll
    for (int curr_stage = stage; curr_stage > 1; curr_stage--) {
      int src_offset = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
      int dst_offset = ((1 << (curr_stage - 2)) - 1) * num_outer_blocks;
      int curr_stage_tw_num = ((1 << (curr_stage - 1))) * num_outer_blocks;
#pragma unroll
      for (int i = 0; i < curr_stage_tw_num / 2; i++) {
        word operand = twiddle_factor_set[src_offset + i * 2];
        twiddle_factor_set[dst_offset + i] =
            basic::detail::__mult_montgomery_lazy<word>(
                operand, static_cast<signed_word>(operand), q, q_inv);
      }
    }

    // compute with the twiddle factor set
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = (1 << (stage - curr_stage + 1));
      int num_blocks = (radix / block_size);
      int tw_offset = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
#pragma unroll
      for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++) {
          ButterflyNTT(local[curr_block * block_size + i],
                       local[curr_block * block_size + i + stride],
                       twiddle_factor_set[tw_offset + curr_block], q, q_inv);
        }
      }
    }
  } else {
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = (1 << (stage - curr_stage + 1));
      int num_blocks = (radix / block_size);
      if (curr_stage == stage) {
        word OT_factors[num_tw_factor];
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++) {
          OT_factors[i] = basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
        }

#pragma unroll
        for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
          int stride = block_size / 2;
#pragma unroll
          for (int i = 0; i < stride; i++) {
            ButterflyNTT(local[curr_block * block_size + i],
                         local[curr_block * block_size + i + stride],
                         OT_factors[curr_block], q, q_inv);
          }
        }
        continue;
      }
#pragma unroll
      for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++) {
          ButterflyNTT(local[curr_block * block_size + i],
                       local[curr_block * block_size + i + stride],
                       w[(1 << (curr_stage - 1)) * tw_idx + curr_block], q,
                       q_inv);
        }
      }
    }
  }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTT(make_signed_t<word> *local,
                                          int tw_idx, const word *w,
                                          const word q,
                                          const make_signed_t<word> q_inv) {
  // Recursive call
  if constexpr (stage > 1) {
    MultiRadixINTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
  }

  constexpr int num_tw = radix / (1 << stage);
  constexpr int stride = (1 << (stage - 1));

  // Load twiddle factors
  word w_vec[num_tw];
  basic::VectorizedMove<word, num_tw>(w_vec, w + tw_idx);

#pragma unroll
  for (int i = 0; i < num_tw; i++) {
#pragma unroll
    for (int j = 0; j < stride; j++) {
      ButterflyINTT(local[i * 2 * stride + j],
                    local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
    }
  }
}
template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixINTT_OT(make_signed_t<word> *local,
                                             int tw_idx, const word *w,
                                             const word *w_msb, const word q,
                                             const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  int first_tw_idx = (1 << (stage - 1)) * tw_idx;
  int msbIdx = first_tw_idx / lsb_size;
  int lsbIdx = first_tw_idx % lsb_size;

  constexpr int num_outer_blocks = radix / (1 << stage);  // 4
  constexpr int accumed_tw_num = (1 << (stage)) - 1;      // 3
  word twiddle_factor_set[accumed_tw_num * num_outer_blocks];

  constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
  if constexpr (kExtendedOT) {
    // load first twiddle factor
#pragma unroll
    for (int i = 0; i < num_tw_factor; i++) {
      twiddle_factor_set[i] = basic::detail::__mult_montgomery_lazy<word>(
          w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
    }

    // compute all twiddle factors with the first stage twiddle factor
    int accum = 0;
#pragma unroll
    for (int curr_stage = 1; curr_stage < stage; curr_stage++) {
      int curr_stage_tw_num = num_outer_blocks * (1 << (stage - curr_stage));
#pragma unroll
      for (int i = 0; i < curr_stage_tw_num / 2; i++) {
        word operand = twiddle_factor_set[accum + i * 2];
        twiddle_factor_set[curr_stage_tw_num + accum + i] =
            basic::detail::__mult_montgomery_lazy<word>(
                operand, static_cast<signed_word>(operand), q, q_inv);
      }
      accum += curr_stage_tw_num;
    }
    accum = 0;
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = (1 << curr_stage);
      int num_blocks = (radix / block_size);
      // int tw_offset = accum;
#pragma unroll
      for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++) {
          ButterflyINTT(local[curr_block * block_size + i],
                        local[curr_block * block_size + i + stride],
                        twiddle_factor_set[accum + curr_block], q, q_inv);
        }
      }
      accum += num_blocks;
    }
  } else {
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = (1 << curr_stage);
      int num_blocks = (radix / block_size);
      if (curr_stage == 1) {
        word OT_factors[num_tw_factor];
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++) {
          OT_factors[i] = basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
        }
#pragma unroll
        for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
          int stride = block_size / 2;
#pragma unroll
          for (int i = 0; i < stride; i++) {
            ButterflyINTT(local[curr_block * block_size + i],
                          local[curr_block * block_size + i + stride],
                          OT_factors[curr_block], q, q_inv);
          }
        }
        continue;
      }
#pragma unroll
      for (int curr_block = 0; curr_block < num_blocks; curr_block++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++) {
          ButterflyINTT(local[curr_block * block_size + i],
                        local[curr_block * block_size + i + stride],
                        w[(1 << (stage - curr_stage)) * tw_idx + curr_block], q,
                        q_inv);
        }
      }
    }
  }
}

template <typename word>
__device__ __inline__ void NopFunc(make_signed_t<word> &result,
                                   const make_signed_t<word> a, const word b,
                                   const word prime,
                                   const make_signed_t<word> montgomery) {}
template <typename word>
__device__ __inline__ void MultConstLazy(make_signed_t<word> &result,
                                         const make_signed_t<word> a,
                                         const word b, const word prime,
                                         const make_signed_t<word> montgomery) {
  result = basic::detail::__mult_montgomery_lazy<word>(
      a, static_cast<make_signed_t<word>>(b), prime, montgomery);
}

template <typename word>
__device__ __inline__ void MultConst(make_signed_t<word> &result,
                                     const make_signed_t<word> a, const word b,
                                     const word prime,
                                     const make_signed_t<word> montgomery) {
  auto temp = basic::detail::__mult_montgomery_lazy<word>(
      a, static_cast<make_signed_t<word>>(b), prime, montgomery);
  if (temp < 0) temp += prime;
  result = temp;
}

template <typename word>
__device__ __inline__ void MultConstNormalize(
    make_signed_t<word> &result, const make_signed_t<word> a, const word b,
    const word prime, const make_signed_t<word> montgomery) {
  auto temp = basic::detail::__mult_montgomery_lazy<word>(
      a, static_cast<make_signed_t<word>>(b), prime, montgomery);
  if (temp < 0) temp += prime;
  if (temp > (prime >> 1)) temp -= prime;
  result = temp;
}

}  // namespace kernel

template <int log_degree, NTTType type, Phase phase>
struct NTTLaunchConfig {
  static constexpr int degree_ = 1 << log_degree;

  __host__ __device__ static constexpr int RadixStages() {
    if ((type == NTTType::NTT && phase == Phase::Phase1) ||
        (type == NTTType::INTT && phase == Phase::Phase2)) {
      if (log_degree == 16)
        return 7;
      else
        return log_degree - 9;
    }
    // else
    return 9;
  }
  __host__ __device__ static constexpr int StageMerging() {
    if ((type == NTTType::NTT && phase == Phase::Phase1) ||
        (type == NTTType::INTT && phase == Phase::Phase2)) {
      if (log_degree == 16) return 4;
    }
    // else
    return 3;
  }
  __host__ __device__ static constexpr int LogWarpBatching() {
    if ((type == NTTType::NTT && phase == Phase::Phase1) ||
        (type == NTTType::INTT && phase == Phase::Phase2)) {
      if (log_degree == 16) return 4;
    }
    // else
    return 0;
  }
  __host__ __device__ static constexpr int LsbSize() {
    if constexpr (log_degree == 16) {
      return 32;
    }
    // else
    return 32;
  }
  __host__ __device__ static constexpr bool OFTwiddle() { return true; }
  __host__ static constexpr int BlockDim() {
    return 1 << (RadixStages() - StageMerging() + LogWarpBatching());
  }
};

}  // namespace cheddar