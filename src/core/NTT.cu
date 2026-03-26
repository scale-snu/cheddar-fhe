#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/ConstantMemory.cuh"
#include "common/PrimeUtils.h"
#include "common/PtrList.h"
#include "core/NTT.h"
#include "core/NTTUtils.cuh"

namespace {
// https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <int Start, int End, int Inc = 1, class Func>
constexpr void constexpr_for(Func &&func) {
  if constexpr (Start < End) {
    func(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(std::forward<Func>(func));
  }
}
}  // namespace

namespace cheddar {
namespace kernel {
template <typename word, int log_degree>
__global__ void INTTPhase1(make_signed_t<word> *dst, const word *primes,
                           const make_signed_t<word> *inv_primes,
                           const word *twiddle_factors,
                           const word *twiddle_factors_msb, int tw_y_extra,
                           int num_q_primes,
                           const InputPtrList<make_signed_t<word>, 1> src) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase1>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
  }
  signed_word *dst_limb = dst + (y_idx << log_degree);

  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const signed_word *load_ptr = src_limb + (x_idx << kStageMerging);
  basic::VectorizedMove<signed_word, kPerThreadElems>(local, load_ptr);

  // INTT main
  int tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int sm_log_stride = 0;
  int sm_idx = batch_idx << kStageMerging;

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    if (i == 0) {
      if constexpr (kOFTwiddle) {
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                             prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT) {
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                             prime, inv_prime);
      }
    }

    // Store the results in shared memory and synchronize
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();

    // Adjust indices and strides for the next iteration
    if (i == num_main_iters - 1) {
      tw_idx >>= kTailStages;
      sm_log_stride += kTailStages;
    } else {
      tw_idx >>= kStageMerging;
      sm_log_stride += kStageMerging;
    }

    // Reload the data from shared memory
    sm_idx = (batch_idx & ((1 << sm_log_stride) - 1)) +
             ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);

  int dst_idx = batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
                (row_idx << kNumStages);
  for (int j = 0; j < kPerThreadElems; j++) {
    dst_limb[dst_idx + (j << (kNumStages - kStageMerging))] = local[j];
  }
}

template <typename word, int log_degree,
          elem_func_t<word> elem_func = NopFunc<word>>
__global__ void INTTPhase2(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    int tw_y_extra, int num_q_primes,
    const InputPtrList<make_signed_t<word>, 1> src,
    const InputPtrList<word, 1> src_const = InputPtrList<word, 1>()) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  // constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  // We do not use OF-Twiddle in this phase
  // constexpr int kOFTwiddle = false;
  // We use batching
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  // Indexing preparation
  // int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  int src_const_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
    src_const_idx += src_const.extra_;
  }
  word src_const_value =
      basic::StreamingLoadConst(src_const.ptrs_[0] + src_const_idx);
  signed_word *dst_limb = dst + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);

  // Load first input
  signed_word local[kPerThreadElems];
  constexpr int initial_log_stride = (log_degree - kNumStages);
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const signed_word *load_ptr =
      src_limb + (stage_group_idx << (initial_log_stride + kStageMerging)) +
      batch_idx + (blockIdx.x << kLogWarpBatching);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << initial_log_stride));
  }

  int tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int sm_log_stride = kLogWarpBatching;
  int sm_idx =
      (threadIdx.x & ((1 << sm_log_stride) - 1)) +
      ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                         prime, inv_prime);

    // Store the results in shared memory and synchronize
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();

    // Adjust indices and strides for the next iteration
    if (i == num_main_iters - 1) {
      tw_idx >>= kTailStages;
      sm_log_stride += kTailStages;
    } else {
      tw_idx >>= kStageMerging;
      sm_log_stride += kStageMerging;
    }

    // Reload the data from shared memory
    sm_idx =
        (threadIdx.x & ((1 << sm_log_stride) - 1)) +
        ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);

  int dst_idx = batch_idx + (stage_group_idx << initial_log_stride) +
                (blockIdx.x << kLogWarpBatching);

  for (int j = 0; j < kPerThreadElems; j++) {
    elem_func(local[j], local[j], src_const_value, prime, inv_prime);
    dst_limb[dst_idx + (j << (log_degree - kStageMerging))] = local[j];
  }
}

template <typename word, int log_degree>
__global__ void NTTPhase1(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    int tw_y_extra, int num_q_primes, int skip_start, int skip_end,
    const InputPtrList<make_signed_t<word>, 1> src,
    const InputPtrList<word, 1> src_const = InputPtrList<word, 1>()) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase1>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  // constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  // We do not use OF-Twiddle in this phase
  // constexpr int kOFTwiddle = false;
  // We use batching
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  // Indexing preparation
  int y_idx = blockIdx.y;
  if (y_idx >= skip_start) {
    y_idx += (skip_end - skip_start);
  }
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + (y_idx << log_degree);
  int src_const_idx = y_idx;
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
    src_const_idx += src_const.extra_;
  }
  signed_word *dst_limb = dst + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);

  // Load first input
  signed_word local[kPerThreadElems];
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const signed_word *load_ptr = src_limb + batch_idx +
                                (blockIdx.x << kLogWarpBatching) +
                                (stage_group_idx << (log_degree - kNumStages));
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad<signed_word>(
        load_ptr + (i << (log_degree - kStageMerging)));
  }

  if (src_const.ptrs_[0] != nullptr) {
    const word src_const_value =
        basic::StreamingLoadConst(src_const.ptrs_[0] + src_const_idx);
    for (int i = 0; i < kPerThreadElems; i++) {
      MultConstLazy<word>(local[i], local[i], src_const_value, prime,
                          inv_prime);
    }
  }

  int final_tw_idx = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = kNumStages - kStageMerging + kLogWarpBatching;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[threadIdx.x + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (threadIdx.x & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w, prime,
                                                        inv_prime);
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  int dst_idx =
      batch_idx +
      (stage_group_idx << ((log_degree - kNumStages) + kStageMerging)) +
      (blockIdx.x << kLogWarpBatching);

  for (int i = 0; i < kPerThreadElems; i++) {
    dst_limb[dst_idx + (i << (log_degree - kNumStages))] = local[i];
  }
}

template <typename word, int log_degree>
__global__ void NTTPhase2(make_signed_t<word> *dst, const word *primes,
                          const make_signed_t<word> *inv_primes,
                          const word *twiddle_factors,
                          const word *twiddle_factors_msb, int tw_y_extra,
                          int num_q_primes, int skip_start, int skip_end,
                          const InputPtrList<make_signed_t<word>, 1> src) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  if (y_idx >= skip_start) {
    y_idx += (skip_end - skip_start);
  }
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src.ptrs_[0] + (y_idx << log_degree);
  if (y_idx >= num_q_primes) {
    tw_y_idx += tw_y_extra;
    src_limb += src.extra_;
  }
  signed_word *dst_limb = dst + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int log_stride = kNumStages - kStageMerging;
  const signed_word *load_ptr =
      src_limb + batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
      (row_idx << kNumStages);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << log_stride));
  }

  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = log_stride;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[batch_idx + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (batch_idx & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    if (i == 0) {
      // last phase
      if constexpr (kOFTwiddle) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    }
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  // Lazy normalization
  for (int i = 0; i < kPerThreadElems; i++) {
    if (local[i] < 0) {
      local[i] += prime;
    }
  }

  signed_word *dst_ptr = dst_limb + (x_idx << kStageMerging);
  basic::VectorizedMove<signed_word, kPerThreadElems>(dst_ptr, local);
}

// We can safely assume that skip_start = 0, skip_end = 0, and all extra = 0
template <typename word, int log_degree>
__global__ void NTTPhase2ForModDown(
    make_signed_t<word> *dst, const word *primes,
    const make_signed_t<word> *inv_primes, const word *twiddle_factors,
    const word *twiddle_factors_msb, int src2_start, int src2_end,
    const make_signed_t<word> *src, const make_signed_t<word> *src2,
    const word *inv_p_prod, const word *src2_padding = nullptr) {
  // Shared memory initialization
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);
  signed_word *temp_orig = temp;

  // Parameters
  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase2>;
  constexpr int kNumStages = Config::RadixStages();
  constexpr int kStageMerging = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize = Config::LsbSize();
  constexpr int kMsbSize = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();
  int row_idx = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  // Indexing preparation
  int y_idx = blockIdx.y;
  word prime = basic::StreamingLoadConst(primes + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  int tw_y_idx = y_idx;
  const signed_word *src_limb = src + (y_idx << log_degree);
  signed_word *dst_limb = dst + (y_idx << log_degree);
  const word *w = twiddle_factors + (tw_y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (tw_y_idx * kMsbSize);

  // Load first input
  signed_word local[kPerThreadElems];
  int log_stride = kNumStages - kStageMerging;
  const signed_word *load_ptr =
      src_limb + batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
      (row_idx << kNumStages);
  for (int i = 0; i < kPerThreadElems; i++) {
    local[i] = basic::StreamingLoad(load_ptr + (i << log_stride));
  }

  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int tw_idx = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = log_stride;

  // First stage
  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(local, tw_idx, w,
                                                         prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++) {
    temp[batch_idx + (j << sm_log_stride)] = local[j];
  }
  __syncthreads();
  sm_log_stride -= kTailStages;

  // Subsequent stages
  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (batch_idx & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++) {
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    }

    int tw_idx = final_tw_idx >> (kStageMerging * i);
    if (i == 0) {
      if constexpr (kOFTwiddle) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    } else {
      if constexpr (kOFTwiddle && !kExtendedOT) {
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      } else {
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w,
                                                            prime, inv_prime);
      }
    }
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++) {
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    }
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  // Lazy normalization
  for (int i = 0; i < kPerThreadElems; i++) {
    if (local[i] < 0) {
      local[i] += prime;
    }
  }

  // SSA steps
  basic::VectorizedMove<signed_word, kPerThreadElems>(
      temp + batch_idx * kPerThreadElems, local);
  __syncthreads();

  int src2_y_index = y_idx - src2_start;
  int offset = (src2_y_index << log_degree) +
               (blockIdx.x << (kNumStages + kLogWarpBatching));
  const signed_word *src2_pos = src2 + offset;
  signed_word inv_p_prod_val = basic::StreamingLoadConst(inv_p_prod + y_idx);
  signed_word *dst_pos =
      dst_limb + (blockIdx.x << (kNumStages + kLogWarpBatching));

  signed_word src2_padding_val = 0;
  bool src2_exists = (src2_y_index >= 0 && y_idx < src2_end);
  if (src2_padding != nullptr && src2_exists) {
    src2_padding_val = basic::StreamingLoadConst(src2_padding + src2_y_index);
  }

  for (int i = threadIdx.x; i < blockDim.x * kPerThreadElems; i += blockDim.x) {
    signed_word res = 0;
    if (src2_exists) {
      res = src2_pos[i];
      if (src2_padding != nullptr) {
        res = basic::detail::__mult_montgomery_lazy<word>(res, src2_padding_val,
                                                          prime, inv_prime);
        if (res < 0) {
          res += prime;
        }
      }
    }
    res -= temp_orig[i];
    res = basic::detail::__mult_montgomery_lazy<word>(res, inv_p_prod_val,
                                                      prime, inv_prime);
    if (res < 0) {
      res += prime;
    }
    dst_pos[i] = res;
  }
}

}  // namespace kernel

// ----- template for each functions ------
template <typename word>
void NTTHandler<word>::NTT(DvView<word> &dst, const NPInfo &np,
                           const DvConstView<word> &src,
                           bool montgomery_conversion /*= false*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "NTT: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    if (montgomery_conversion) {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          src_ptr_list, src_const_ptr_list);
    } else {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          src_ptr_list);
    }
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, 0, 0, src_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::INTT(DvView<word> &dst, const NPInfo &np,
                            const DvConstView<word> &src,
                            bool montgomery_conversion /*= true*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "INTT: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = inv_twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, src_ptr_list);
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list;
  if (montgomery_conversion) {
    src_const_ptr_list.ptrs_[0] = inv_degree_.data() + ter_left;
  } else {
    src_const_ptr_list.ptrs_[0] = inv_degree_mont_.data() + ter_left;
  }
  src_const_ptr_list.extra_ = main_left;

  // Phase 2
  block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase2<word, j, kernel::MultConst<word>>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
            src_ptr_list, src_const_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::INTTAndMultConst(DvView<word> &dst, const NPInfo &np,
                                        const DvConstView<word> &src,
                                        const DvConstView<word> &src_const,
                                        bool normalize /*= false*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "INTTForModUp: Invalid dst size");

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;
  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = inv_twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, src_ptr_list);
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list(src_const);
  src_const_ptr_list.extra_ = src_const.QSize() - num_q_primes;

  // Phase 2
  block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    if (normalize) {
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list);
    } else {
      kernel::INTTPhase2<word, j, kernel::MultConst<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list);
    }
  });
}

template <typename word>
void NTTHandler<word>::NTTForModUp(DvView<word> &dst, const NPInfo &np,
                                   int skip_start, int skip_end,
                                   const DvConstView<word> &src) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "NTTForModUp: Invalid dst size");

  // Extra handling for skip primes
  AssertTrue(skip_start >= 0 && skip_start < num_q_primes &&
                 skip_end >= skip_start && skip_end <= num_q_primes,
             "NTTForModUp: Invalid skip primes");
  num_total_primes -= (skip_end - skip_start);

  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);
  int ter_left = param_.GetMaxNumTer() - np.num_ter_;
  int main_left = param_.GetMaxNumMain() - np.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());
  InputPtrList<signed_word, 1> src_ptr_list;
  src_ptr_list.ptrs_[0] = src_ptr;
  src_ptr_list.extra_ = src.QSize() - q_size;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    // montgomery_conversion is always false
    if constexpr (kFuseMontgomery) {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
          skip_start, skip_end, src_ptr_list);
    } else {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
          skip_start, skip_end, src_ptr_list, src_const_ptr_list);
    }
  });

  src_ptr_list.ptrs_[0] = dst_ptr;
  src_ptr_list.extra_ = 0;

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
        dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
        num_q_primes, skip_start, skip_end, src_ptr_list);
  });
}

template <typename word>
void NTTHandler<word>::NTTForModDown(
    DvView<word> &dst, const NPInfo &np_src1, const NPInfo &np_src2,
    const DvConstView<word> &src1, const DvConstView<word> &src2,
    const DvConstView<word> &inv_p_prod,
    const DvConstView<word> &src2_padding /*= DvConstView<word>(nullptr,
                                                              0)*/) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_q_primes = np_src1.GetNumQ();
  int q_size = num_q_primes * param_.degree_;
  int num_total_primes = np_src1.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "NTTForModUp: Invalid dst size");

  // Special restrictions for NTTForModDown
  AssertTrue(np_src1.num_aux_ == 0, "NTTForModDown: num_aux should be 0");

  int num_src2_primes = np_src2.GetNumQ();
  AssertTrue(num_src2_primes <= num_total_primes,
             "NTTForModDown: Invalid src2 size");
  AssertTrue(dst.data() != src2.data(),
             "NTTForModDown: dst and src2 should be different");
  int src2_start = np_src1.num_ter_ - np_src2.num_ter_;
  int src2_end = src2_start + num_src2_primes;
  AssertTrue(src2_end <= num_total_primes, "NTTForModDown: Invalid src2 size");

  const word *primes = param_.GetPrimesPtr(np_src1);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_src1);
  int ter_left = param_.GetMaxNumTer() - np_src1.num_ter_;
  int main_left = param_.GetMaxNumMain() - np_src1.num_main_;

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src1_ptr = reinterpret_cast<const signed_word *>(src1.data());
  auto src2_ptr = reinterpret_cast<const signed_word *>(src2.data());
  // We do in-place NTT of dst first
  InputPtrList<signed_word, 1> ntt_ptr_list;
  ntt_ptr_list.ptrs_[0] = src1_ptr;
  ntt_ptr_list.extra_ = 0;

  const word *tw_ptr = twiddle_factors_.data() + ter_left * param_.degree_;
  const word *tw_msb_ptr =
      twiddle_factors_msb_.data() + ter_left * GetMsbSize();

  // Phase 1
  int block_dim = GetBlockDim(NTTType::NTT, Phase::Phase1);
  int stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase1);
  dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                num_total_primes);
  int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    // montgomery_conversion is always false
    if constexpr (kFuseMontgomery) {
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          ntt_ptr_list);
    } else {
      InputPtrList<word, 1> src_const_ptr_list;
      src_const_ptr_list.ptrs_[0] = montgomery_converter_.data() + ter_left;
      src_const_ptr_list.extra_ = main_left;
      kernel::NTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes, 0, 0,
          ntt_ptr_list, src_const_ptr_list);
    }
  });

  // Phase 2
  block_dim = GetBlockDim(NTTType::NTT, Phase::Phase2);
  stage_merging = GetStageMerging(NTTType::NTT, Phase::Phase2);
  grid_dim =
      dim3(param_.degree_ / (1 << stage_merging) / block_dim, num_total_primes);
  shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (log_degree != j) return;
    kernel::NTTPhase2ForModDown<word, j>
        <<<grid_dim, block_dim, shared_mem_size>>>(
            dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, src2_start,
            src2_end, dst_ptr, src2_ptr, inv_p_prod.data(),
            src2_padding.data());
  });
}

// dst = INTT(src) * const_src
template <typename word>
void NTTHandler<word>::INTTForModDown(
    DvView<word> &dst, const NPInfo &np_src, const NPInfo &np_non_intt,
    const DvConstView<word> &src, const DvConstView<word> &src_const) const {
  using signed_word = make_signed_t<word>;
  int log_degree = param_.log_degree_;
  int num_total_primes = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  AssertTrue(dst.TotalSize() == num_total_primes * param_.degree_,
             "INTTForModDown: Invalid dst size");

  // Specific check for INTTForModDown
  AssertTrue(np_src.GetNumQ() * param_.degree_ == src.QSize(),
             "INTTForModDown: Invalid src size");
  AssertTrue(np_src.GetNumTotal() * param_.degree_ == src.TotalSize(),
             "INTTForModDown: Invalid src size");
  AssertTrue(np_non_intt.num_aux_ == 0,
             "INTTForModDown: num_aux should be 0 after moddown");
  AssertTrue(np_non_intt.IsSubsetOf(np_src),
             "INTTForModDown: Invalid np combination");
  AssertTrue(src.data() != dst.data(),
             "INTTForModDown: src and dst should be different");
  AssertTrue(src_const.AuxSize() == np_src.num_aux_,
             "INTTForModDown: Invalid src_const size");
  AssertTrue(
      src_const.TotalSize() == np_src.GetNumTotal() - np_non_intt.GetNumTotal(),
      "INTTForModDown: Invalid src_const size");

  // We either perform INTT on main primes or terminal primes (+ aux primes --
  // optional) and not both.
  // Also, it's possible that we don't perform INTT on any q primes.
  bool intt_on_main = np_src.num_main_ > np_non_intt.num_main_;
  bool intt_on_ter = np_src.num_ter_ > np_non_intt.num_ter_;
  AssertTrue(!intt_on_main || !intt_on_ter,
             "INTTForModDown: Invalid np combination");

  // unsafe conversion
  auto dst_ptr = reinterpret_cast<signed_word *>(dst.data());
  auto src_ptr = reinterpret_cast<const signed_word *>(src.data());

  // Preparing src_const
  InputPtrList<word, 1> src_const_ptr_list;
  src_const_ptr_list.ptrs_[0] = src_const.data();
  src_const_ptr_list.extra_ = 0;

  // Case 1: We only perform INTT on the upper part of src
  if (!intt_on_ter) {
    int num_q_primes = np_src.num_main_ - np_non_intt.num_main_;

    const word *primes = param_.GetPrimesPtr(np_src);
    const signed_word *inv_primes = param_.GetInvPrimesPtr(np_src);

    // We ignore lower part
    int num_src_offset_primes = np_src.num_ter_ + np_non_intt.num_main_;
    int num_tw_offset_primes = param_.GetMaxNumTer() + np_non_intt.num_main_;
    primes += num_src_offset_primes;
    inv_primes += num_src_offset_primes;

    int main_left = param_.GetMaxNumMain() - np_src.num_main_;

    InputPtrList<signed_word, 1> src_ptr_list;
    src_ptr_list.ptrs_[0] = src_ptr + num_src_offset_primes * param_.degree_;
    src_ptr_list.extra_ = 0;

    const word *tw_ptr =
        inv_twiddle_factors_.data() + num_tw_offset_primes * param_.degree_;
    const word *tw_msb_ptr =
        inv_twiddle_factors_msb_.data() + num_tw_offset_primes * GetMsbSize();

    // Phase 1
    int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
    int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
    dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                  num_total_primes);
    int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, main_left,
          num_q_primes, src_ptr_list);
    });

    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;

    // Phase 2
    block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
    stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
    grid_dim = dim3(param_.degree_ / (1 << stage_merging) / block_dim,
                    num_total_primes);
    shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, main_left, num_q_primes,
              src_ptr_list, src_const_ptr_list);
    });
  } else {  // Case 2. We perform INTT on some of the ter primes and all aux
            // primes
    int num_q_primes = np_src.num_ter_ - np_non_intt.num_ter_;
    int q_size = num_q_primes * param_.degree_;

    const word *primes =
        param_.__GetPrimesPtrModDownWithTerPrimes(np_src, np_non_intt);
    const signed_word *inv_primes =
        param_.__GetInvPrimesPtrModDownWithTerPrimes(np_src, np_non_intt);

    int ter_left = param_.GetMaxNumTer() - np_src.num_ter_;
    int main_left = param_.GetMaxNumMain() - np_src.num_main_;
    int tw_y_extra = param_.GetMaxNumMain() + np_non_intt.num_ter_;

    InputPtrList<signed_word, 1> src_ptr_list;
    src_ptr_list.ptrs_[0] = src_ptr;
    src_ptr_list.extra_ = src.QSize() - q_size;

    const word *tw_ptr =
        inv_twiddle_factors_.data() + ter_left * param_.degree_;
    const word *tw_msb_ptr =
        inv_twiddle_factors_msb_.data() + ter_left * GetMsbSize();

    // Phase 1
    int block_dim = GetBlockDim(NTTType::INTT, Phase::Phase1);
    int stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase1);
    dim3 grid_dim(param_.degree_ / (1 << stage_merging) / block_dim,
                  num_total_primes);
    int shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase1<word, j><<<grid_dim, block_dim, shared_mem_size>>>(
          dst_ptr, primes, inv_primes, tw_ptr, tw_msb_ptr, tw_y_extra,
          num_q_primes, src_ptr_list);
    });

    src_ptr_list.ptrs_[0] = dst_ptr;
    src_ptr_list.extra_ = 0;

    // Phase 2
    block_dim = GetBlockDim(NTTType::INTT, Phase::Phase2);
    stage_merging = GetStageMerging(NTTType::INTT, Phase::Phase2);
    grid_dim = dim3(param_.degree_ / (1 << stage_merging) / block_dim,
                    num_total_primes);
    shared_mem_size = block_dim * (1 << stage_merging) * sizeof(word);

    constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
      if (log_degree != j) return;
      kernel::INTTPhase2<word, j, kernel::MultConstNormalize<word>>
          <<<grid_dim, block_dim, shared_mem_size>>>(
              dst_ptr, primes, inv_primes, tw_ptr, tw_y_extra, num_q_primes,
              src_ptr_list, src_const_ptr_list);
    });
  }
}

template <typename word>
int NTTHandler<word>::GetLsbSize() const {
  int log_degree = param_.log_degree_;
  int lsb_size = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree)
      lsb_size = NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::LsbSize();
    return;
  });
  return lsb_size;
}

template <typename word>
int NTTHandler<word>::GetMsbSize() const {
  int log_degree = param_.log_degree_;
  int lsb_size = GetLsbSize();
  return (1 << log_degree) / lsb_size;
}

template <typename word>
int NTTHandler<word>::GetLogWarpBatching() const {
  int log_degree = param_.log_degree_;
  int log_warp_batching = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree)
      log_warp_batching =
          NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::LogWarpBatching();
    return;
  });
  return log_warp_batching;
}

template <typename word>
int NTTHandler<word>::GetStageMerging(NTTType type, Phase phase) const {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "GetStageMerging: Invalid log_degree");
  int stage_merging = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree) {
      if (type == NTTType::NTT) {
        if (phase == Phase::Phase1) {
          stage_merging =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::StageMerging();
        } else {
          stage_merging =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase2>::StageMerging();
        }
      } else {
        if (phase == Phase::Phase1) {
          stage_merging =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase1>::StageMerging();
        } else {
          stage_merging =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase2>::StageMerging();
        }
      }
    }
    return;
  });
  return stage_merging;
}

template <typename word>
int NTTHandler<word>::GetBlockDim(NTTType type, Phase phase) const {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "GetBlockDim: Invalid log_degree");
  int block_dim = 0;
  constexpr_for<min_log_degree_, max_log_degree_ + 1>([&](auto j) {
    if (j == log_degree) {
      if (type == NTTType::NTT) {
        if (phase == Phase::Phase1) {
          block_dim =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase1>::BlockDim();
        } else {
          block_dim =
              NTTLaunchConfig<j, NTTType::NTT, Phase::Phase2>::BlockDim();
        }
      } else {
        if (phase == Phase::Phase1) {
          block_dim =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase1>::BlockDim();
        } else {
          block_dim =
              NTTLaunchConfig<j, NTTType::INTT, Phase::Phase2>::BlockDim();
        }
      }
    }
    return;
  });
  return block_dim;
}

template <typename word>
NTTHandler<word>::NTTHandler(const Parameter<word> &param) : param_(param) {
  if (!cm_populated_) {
    PopulateConstantMemory(param_);
    cm_populated_ = true;
  }
  PopulateTwiddleFactors();
}

template <typename word>
void NTTHandler<word>::PopulateTwiddleFactors() {
  int log_degree = param_.log_degree_;
  AssertTrue(log_degree >= min_log_degree_ && log_degree <= max_log_degree_,
             "NTTHandler: Invalid log_degree");
  int degree = (1 << log_degree);
  NPInfo np = param_.LevelToNP(param_.max_level_, param_.alpha_);
  const auto &primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  int lsb_size = GetLsbSize();
  int msb_size = GetMsbSize();

  Hv h_psi_rev_mont(degree * num_total_primes, 0);
  Hv h_psi_inv_rev_mont(degree * num_total_primes, 0);
  Hv h_N_inv(num_total_primes, 0);
  Hv h_N_inv_mont(num_total_primes, 0);
  Hv h_mont_convert(num_total_primes, 0);

  Hv h_psi_rev_mont_msb(msb_size * num_total_primes, 0);
  Hv h_psi_inv_rev_mont_msb(msb_size * num_total_primes, 0);

  for (int i = 0; i < num_total_primes; i++) {
    std::vector<word> psi_rev(degree);
    std::vector<word> psi_inv_rev(degree);

    word p = primes[i];
    word psi = primeutil::FindPrimitiveMthRoot(2 * degree, p);
    word psi_inv = primeutil::InvMod<word>(psi, p);

    h_N_inv[i] = primeutil::InvMod<word>(degree, p);
    h_N_inv_mont[i] = primeutil::ToMontgomery<word>(h_N_inv[i], p);
    h_mont_convert[i] =
        primeutil::ToMontgomery(primeutil::ToMontgomery<word>(1, p), p);

    psi_rev[0] = 1;
    psi_inv_rev[0] = 1;
    for (int j = 1; j < degree; j++) {
      psi_rev[j] = primeutil::MultMod(psi_rev[j - 1], psi, p);
      psi_inv_rev[j] = primeutil::MultMod(psi_inv_rev[j - 1], psi_inv, p);
    }
    BitReverseVector(psi_rev);
    BitReverseVector(psi_inv_rev);

    for (int j = 0; j < degree; j++) {
      h_psi_rev_mont[i * degree + j] =
          primeutil::ToMontgomery<word>(psi_rev[j], p);
      h_psi_inv_rev_mont[i * degree + j] =
          primeutil::ToMontgomery<word>(psi_inv_rev[j], p);
    }

    // OFTwiddle computation
    for (int j = 0; j < msb_size; j++) {
      h_psi_rev_mont_msb[i * msb_size + j] =
          h_psi_rev_mont[i * degree + j * lsb_size];
      h_psi_inv_rev_mont_msb[i * msb_size + j] =
          h_psi_inv_rev_mont[i * degree + j * lsb_size];
    }
  }
  // Copy from host to device
  CopyHostToDevice<word>(twiddle_factors_, h_psi_rev_mont);
  CopyHostToDevice<word>(inv_twiddle_factors_, h_psi_inv_rev_mont);
  CopyHostToDevice<word>(inv_degree_, h_N_inv);
  CopyHostToDevice<word>(inv_degree_mont_, h_N_inv_mont);
  CopyHostToDevice<word>(montgomery_converter_, h_mont_convert);
  CopyHostToDevice<word>(twiddle_factors_msb_, h_psi_rev_mont_msb);
  CopyHostToDevice<word>(inv_twiddle_factors_msb_, h_psi_inv_rev_mont_msb);
}

template <typename word>
DvConstView<word> NTTHandler<word>::ImaginaryUnitConstView(
    const NPInfo &np) const {
  int ter_offset = (param_.GetMaxNumTer() - np.num_ter_) * param_.degree_;
  int q_size = param_.L_ * param_.degree_ - ter_offset;
  int aux_size = param_.alpha_ * param_.degree_;

  return DvConstView<word>(twiddle_factors_.data() + ter_offset + 1,
                           q_size + aux_size, aux_size);
}

template class NTTHandler<uint32_t>;
template class NTTHandler<uint64_t>;

}  // namespace cheddar