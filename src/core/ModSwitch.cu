#include <algorithm>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/ConstantMemory.cuh"
#include "common/DoubleWord.h"
#include "common/PrimeUtils.h"
#include "core/ModSwitch.h"

// kernel constants
#define kUnrollNumber 4
#define kLimbBatching 3
#define kNumThreadsPerBlock 256

#define kNumThreadsX 64
#define kNumThreadsY 4

#define kMaxNumAccum 4
namespace cheddar {
namespace kernel {

template <typename word>
__global__ void ModSwitchMatrixMult(word *dst, const word *primes,
                                    const make_signed_t<word> *inv_primes,
                                    const int src_len, const int dst_len,
                                    const int skip_start, const int skip_end,
                                    const make_signed_t<word> *src,
                                    const make_signed_t<word> *bconv_table) {
  // Load bconv_table into shared memory
  using signed_word = make_signed_t<word>;
  using signed_d_word = make_signed_double_word_t<word>;

  extern __shared__ char __smem[];
  int log_degree = cm_log_degree();

  // For mod_prime q, we prepare bconv_table in range
  // [- (q - 1) / 2, (q - 1) / 2] -- normalized
  // The src values are in range [0, 2^31), but we can normalize
  // them to (-2^30, 2^30) range
  // Therefore the multiplication result is in range
  // (- (q - 1) * 2^29, (q - 1) * 2^29)
  // We can do lazy reduction here.
  // After accumulating 8 results,
  // which will be in range (- (q - 1) * 2^32, (q - 1) * 2^32) < (-2^63, 2^63)
  // We then reduce this range to (-q * 2^31, q * 2^31) by
  // adding or subtracting q * 2^32
  // Then, we can again accumulate 4 results, and repeat the process

  signed_word *bconv_vector = reinterpret_cast<signed_word *>(__smem);
  int bconv_vector_size = src_len * (kLimbBatching * kNumThreadsY);

  signed_word *poly_frag = bconv_vector + bconv_vector_size;

  // Loading bconv table into shared memory
  int thread_idx_flattened = threadIdx.x + threadIdx.y * kNumThreadsX;
  int block_offset = blockIdx.y * bconv_vector_size;
  int bconv_table_size = src_len * dst_len;
  for (int i = thread_idx_flattened; i < bconv_vector_size;
       i += (kNumThreadsX * kNumThreadsY)) {
    int bconv_table_index = i + block_offset;
    if (bconv_table_index < bconv_table_size) {
      bconv_vector[i] = bconv_table[bconv_table_index];
    }
  }
  // __syncthreads();

  // Precomputation of offset for src & bconv_vector
  int degree_index =
      blockIdx.x * (kNumThreadsX * kUnrollNumber) + threadIdx.x * kUnrollNumber;
  src += degree_index;
  bconv_vector += src_len * threadIdx.y * kLimbBatching;
  poly_frag += threadIdx.x * kUnrollNumber;

  // Actual accumulation
  // Each thread is in charge of computing kLimbBatching x kUnrollNumber
  // submatrix of dst
  signed_d_word accum[kLimbBatching][kUnrollNumber] = {0};
  signed_word reg_bconv[kLimbBatching];
  signed_word reg_poly[kUnrollNumber];
  int dst_y_position =
      blockIdx.y * (kLimbBatching * kNumThreadsY) + threadIdx.y * kLimbBatching;

  word reg_primes[kLimbBatching];
  signed_word reg_inv_primes[kLimbBatching];
  for (int i = 0; i < kLimbBatching; i++) {
    int prime_index = dst_y_position + i;
    if (prime_index >= dst_len) break;

    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    reg_primes[i] = basic::StreamingLoadConst(primes + prime_index);
    reg_inv_primes[i] = basic::StreamingLoadConst(inv_primes + prime_index);
  }

  int num_accumulated = 0;
  for (int i = 0; i < src_len; i += kMaxNumAccum) {
    // 1. Load (kNumThreadsX * kUnrollNumber) x kMaxNumAccum chunk from memory
    // and store it in shared memory (poly_frag)
    if (i > 0)
      __syncthreads();  // works like a lock_acquire (we don't need to lock
                        // until reading __smem)
#pragma unroll
    for (int j = 0; j < (kMaxNumAccum / kNumThreadsY); j++) {
      int y_pos = j * kNumThreadsY + threadIdx.y;
      if (i + y_pos >= src_len) break;
      basic::VectorizedMove<signed_word, kUnrollNumber>(
          poly_frag + y_pos * (kNumThreadsX * kUnrollNumber),
          src + (y_pos << log_degree));
    }
    __syncthreads();  // works like a lock_release

    // 2. Perform actual matrix multiplication
    for (int j = 0; j < kMaxNumAccum; j++) {
      if (i + j >= src_len) break;

      basic::VectorizedMove<signed_word, kUnrollNumber>(
          reg_poly, poly_frag + j * (kUnrollNumber * kNumThreadsX));
      for (int k = 0; k < kLimbBatching; k++) {
        if (dst_y_position + k > dst_len) break;  // out of bounds
        signed_word bconv_const = bconv_vector[src_len * k];
#pragma unroll
        for (int l = 0; l < kUnrollNumber; l++) {
          accum[k][l] += basic::detail::__mult_wide(reg_poly[l], bconv_const);
        }
      }
      // BConv vector offset: i * kMaxNumAccum + j
      bconv_vector += 1;
      num_accumulated += 1;
    }

    // 3. Normalize temporary result if necessary
    if constexpr (std::is_same_v<word, uint32_t>) {
      if (num_accumulated >= kMaxNumAccum) {
        for (int k = 0; k < kLimbBatching; k++) {
          if (dst_y_position + k >= dst_len) break;  // out of bound
          signed_d_word prime_th = reg_primes[k];
          prime_th <<= (sizeof(word) * 8);
          signed_d_word prime_th_half = prime_th >> 1;
#pragma unroll
          for (int l = 0; l < kUnrollNumber; l++) {
            if (accum[k][l] < 0) accum[k][l] += prime_th;
            if (accum[k][l] >= prime_th_half) accum[k][l] -= prime_th;
          }
        }
        num_accumulated -= kMaxNumAccum;
      }
    }

    // src offset: i * kMaxNumAccum * degree
    src += (kMaxNumAccum << log_degree);
  }

  for (int k = 0; k < kLimbBatching; k++) {
    int prime_index = dst_y_position + k;
    if (dst_y_position + k >= dst_len) break;  // out of bound
    if (prime_index >= skip_start) {
      prime_index += (skip_end - skip_start);
    }
    word prime = reg_primes[k];
    signed_word inv_prime = reg_inv_primes[k];
    word res_tmp[kUnrollNumber];
#pragma unroll
    for (int l = 0; l < kUnrollNumber; l++) {
      res_tmp[l] = basic::ReduceMontgomery(accum[k][l], prime, inv_prime);
    }
    basic::VectorizedMove<word, kUnrollNumber>(
        dst + (prime_index << log_degree) + degree_index, res_tmp);
  }
}

}  // namespace kernel

template <typename word>
ModSwitchHandler<word>::ModSwitchHandler(
    const Parameter<word> &param, int level,
    const ElementWiseHandler<word> &elem_handler,
    const NTTHandler<word> &ntt_handler)
    : level_{level},
      num_aux_{level == -1 ? param.GetSSENumAux() : param.alpha_},
      beta_{level == -1 ? 1
                        : DivCeil(param.LevelToNP(level).num_main_ +
                                      param.GetMaxNumTer(),
                                  num_aux_)},
      param_{param},
      elem_handler_{elem_handler},
      ntt_handler_{ntt_handler} {
  static_assert(kMaxNumAccum % kNumThreadsY == 0,
                "kMaxNumAccum must be divisible by kNumThreadsY");
  static_assert(
      kNumThreadsPerBlock == kNumThreadsX * kNumThreadsY,
      "kNumThreadsPerBlock must be equal to kNumThreadsX * kNumThreadsY");

  if (!cm_populated_) {
    PopulateConstantMemory(param_);
    cm_populated_ = true;
  }

  NPInfo np = param_.LevelToNP(level_, num_aux_);
  int num_q_primes = np.GetNumQ();
  std::vector<word> level_primes = param_.GetPrimeVector(np);

  int padded_num_q_primes = np.num_main_ + param_.GetMaxNumTer();
  // Add pad to the front of the primes
  if (level_ == -1) padded_num_q_primes = num_q_primes;
  int num_pad = padded_num_q_primes - num_q_primes;

  // mod up constants
  mod_up1_.resize(num_q_primes);
  mod_up2_.resize(beta_);
  for (int i = 0; i < beta_; i++) {
    // absolute index
    int src_start = i * num_aux_;
    int src_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (src_end <= num_pad) continue;

    // relative index
    src_start = Max(0, src_start - num_pad);
    src_end = src_end - num_pad;

    std::vector<word> src_primes(level_primes.begin() + src_start,
                                 level_primes.begin() + src_end);
    std::vector<word> dst_primes(level_primes.begin(),
                                 level_primes.begin() + src_start);
    dst_primes.insert(dst_primes.end(), level_primes.begin() + src_end,
                      level_primes.end());

    DeviceVector<word> mod_up1_tmp;
    PopulateModSwitchConstants(mod_up1_tmp, mod_up2_[i], src_primes, dst_primes,
                               0, 0);
    cudaMemcpyAsync(mod_up1_.data() + src_start, mod_up1_tmp.data(),
                    mod_up1_tmp.size() * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);
  }

  std::vector<word> q_primes(level_primes.begin(),
                             level_primes.begin() + num_q_primes);
  std::vector<word> p_primes(level_primes.begin() + num_q_primes,
                             level_primes.end());

  // ModDown constants
  PopulateModSwitchConstants(mod_down1_, mod_down2_, p_primes, q_primes, 0, 0);
  DeviceVector<word> dummy;
  PopulateModDownEpilogueConstants(inv_prime_prod_, dummy, p_primes, q_primes,
                                   0, 0);

  // Rescale not performed at level -1 or 0
  if (level_ == -1 || level_ == 0) return;

  // Rescale constants
  NPInfo next_np = param_.LevelToNP(level_ - 1, num_aux_);
  std::vector<word> next_level_primes = param_.GetPrimeVector(next_np);
  int next_num_q_primes = next_np.GetNumQ();
  int main_diff = np.num_main_ - next_np.num_main_;
  int ter_diff = np.num_ter_ - next_np.num_ter_;

  std::vector<word> rescale_dst_primes(
      next_level_primes.begin(), next_level_primes.begin() + next_num_q_primes);

  std::vector<word> rescale_src_primes;
  std::vector<word> mod_down_rescale_src_primes;
  if (ter_diff > 0) {  // number of ter decreases
    // May restore main primes
    AssertTrue(main_diff <= 0, "Invalid rescale amount");
    rescale_src_primes.insert(rescale_src_primes.end(), level_primes.begin(),
                              level_primes.begin() + ter_diff);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    rescale_pad_start_ = 0;
    rescale_pad_end_ = num_q_primes - ter_diff;
    rescale_restore_start_ = next_num_q_primes + main_diff;
    rescale_restore_end_ = next_num_q_primes;
  } else if (main_diff > 0) {  // number of main decreases
    // May restore terminal primes
    AssertTrue(ter_diff <= 0, "Invalid rescale amount");
    rescale_src_primes.insert(rescale_src_primes.end(),
                              level_primes.begin() + num_q_primes - main_diff,
                              level_primes.begin() + num_q_primes);
    mod_down_rescale_src_primes = rescale_src_primes;
    mod_down_rescale_src_primes.insert(mod_down_rescale_src_primes.end(),
                                       p_primes.begin(), p_primes.end());
    rescale_pad_start_ = -ter_diff;
    rescale_pad_end_ = num_q_primes - main_diff - ter_diff;
    rescale_restore_start_ = 0;
    rescale_restore_end_ = -ter_diff;
  } else {
    Fail("Invalid rescale amount");
  }
  PopulateModSwitchConstants(rescale1_, rescale2_, rescale_src_primes,
                             rescale_dst_primes, rescale_restore_start_,
                             rescale_restore_end_);
  PopulateModDownEpilogueConstants(
      rescale_inv_prime_prod_, rescale_padding_, rescale_src_primes,
      rescale_dst_primes, rescale_restore_start_, rescale_restore_end_);
  PopulateModSwitchConstants(mod_down_rescale1_, mod_down_rescale2_,
                             mod_down_rescale_src_primes, rescale_dst_primes,
                             rescale_restore_start_, rescale_restore_end_);
  PopulateModDownEpilogueConstants(
      mod_down_rescale_inv_prime_prod_, mod_down_rescale_padding_,
      mod_down_rescale_src_primes, rescale_dst_primes, rescale_restore_start_,
      rescale_restore_end_);

  if (!kFuseModDownEpilogue) {
    HostVector<word> host_entire_padding(num_q_primes + num_aux_, 1);
    for (int i = 0; i < num_q_primes + num_aux_; i++) {
      word mod_prime = level_primes[i];
      for (int j = rescale_restore_start_; j < rescale_restore_end_; j++) {
        host_entire_padding[i] = primeutil::MultMod<word>(
            host_entire_padding[i], rescale_dst_primes[j], mod_prime);
      }
      host_entire_padding[i] =
          primeutil::ToMontgomery<word>(host_entire_padding[i], mod_prime);
    }
    CopyHostToDevice(entire_padding_, host_entire_padding);
  }
}

template <typename word>
void ModSwitchHandler<word>::PopulateModSwitchConstants(
    DeviceVector<word> &const1, DeviceVector<make_signed_t<word>> &const2,
    const std::vector<word> &src_primes, const std::vector<word> &dst_primes,
    int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();

  HostVector<word> mod_switch1(src_len, 1);
  HostVector<make_signed_t<word>> mod_switch2(src_len * dst_len, 1);

  for (int i = 0; i < src_len; i++) {
    word modulus = src_primes[i];

    for (int j = 0; j < src_len; j++) {
      if (j == i) continue;
      mod_switch1[i] =
          primeutil::MultMod<word>(mod_switch1[i], src_primes[j], modulus);
    }
    mod_switch1[i] = primeutil::MultMod<word>(
        mod_switch1[i], static_cast<word>(param_.degree_),
        modulus);  // for inverse N of INTT
    mod_switch1[i] = primeutil::InvMod<word>(mod_switch1[i], modulus);

    if (kFuseModDownEpilogue) {
      // BitPacker extension
      for (int j = restore_start; j < restore_end; j++) {
        mod_switch1[i] =
            primeutil::MultMod<word>(mod_switch1[i], dst_primes[j], modulus);
      }
    }
    // Deliberately not using the Montgomery form here
  }
  for (int m = 0; m < dst_len; m++) {
    word mod_prime = dst_primes[m];
    for (int i = 0; i < src_len; i++) {
      word accum = 1;
      for (int j = 0; j < src_len; j++) {
        if (i == j) continue;
        accum = primeutil::MultMod<word>(accum, src_primes[j], mod_prime);
      }
      // Double Montgomery form
      accum = primeutil::ToMontgomery<word>(accum, mod_prime);
      accum = primeutil::ToMontgomery<word>(accum, mod_prime);
      mod_switch2[m * src_len + i] = basic::Normalize(accum, mod_prime);
    }
  }

  CopyHostToDevice(const1, mod_switch1);
  CopyHostToDevice(const2, mod_switch2);
}

template <typename word>
void ModSwitchHandler<word>::PopulateModDownEpilogueConstants(
    DeviceVector<word> &inv_p_prod, DeviceVector<word> &padding,
    const std::vector<word> &src_primes, const std::vector<word> &dst_primes,
    int restore_start, int restore_end) {
  int src_len = src_primes.size();
  int dst_len = dst_primes.size();
  int restore_len = restore_end - restore_start;

  HostVector<word> host_inv_p_prod(dst_len, 1);
  for (int i = 0; i < dst_len; i++) {
    word mod_prime = dst_primes[i];
    for (int j = 0; j < src_len; j++) {
      host_inv_p_prod[i] = primeutil::MultMod<word>(host_inv_p_prod[i],
                                                    src_primes[j], mod_prime);
    }
    host_inv_p_prod[i] = primeutil::InvMod<word>(host_inv_p_prod[i], mod_prime);
    host_inv_p_prod[i] =
        primeutil::ToMontgomery<word>(host_inv_p_prod[i], mod_prime);
  }
  CopyHostToDevice(inv_p_prod, host_inv_p_prod);

  if (!kFuseModDownEpilogue) return;
  // We treat padding differently

  HostVector<word> host_padding(dst_len - restore_len, 1);
  for (int i = 0; i < dst_len - restore_len; i++) {
    int mod_prime_index = i;
    if (i >= restore_start) {
      mod_prime_index += restore_len;
    }
    word mod_prime = dst_primes[mod_prime_index];
    for (int j = restore_start; j < restore_end; j++) {
      host_padding[i] =
          primeutil::MultMod<word>(host_padding[i], dst_primes[j], mod_prime);
    }
    host_padding[i] = primeutil::ToMontgomery<word>(host_padding[i], mod_prime);
  }
  CopyHostToDevice(padding, host_padding);
}

template <typename word>
void ModSwitchHandler<word>::PseudoModUp(
    DvView<word> &dst, const DvConstView<word> &src,
    const DvConstView<word> &p_prod) const {
  NPInfo np = param_.LevelToNP(level_, 0);
  int num_q_primes = np.GetNumQ();
  AssertTrue(src.TotalSize() == num_q_primes * param_.degree_,
             "PseudoModUp src size mismatch");
  AssertTrue(dst.TotalSize() == num_q_primes * param_.degree_,
             "PseudoModUp dst size mismatch");
  AssertTrue(p_prod.TotalSize() == num_q_primes,
             "PseudoModUp p_prod size mismatch");
  std::vector<DvView<word>> dst_view{dst};
  std::vector<DvConstView<word>> src_view{src};
  elem_handler_.MultConst(dst_view, np, src_view, p_prod);
}
template <typename word>
void ModSwitchHandler<word>::ModUp(std::vector<DvView<word>> &dst,
                                   const DvConstView<word> &src) const {
  using signed_word = make_signed_t<word>;
  NPInfo np = param_.LevelToNP(level_, 0);
  int num_q_primes = np.GetNumQ();
  int prime_offset = (level_ == -1 ? 0 : param_.GetMaxNumTer() - np.num_ter_);
  int degree = param_.degree_;
  AssertTrue(src.QSize() == num_q_primes * degree, "ModUp src q size mismatch");
  AssertTrue(src.AuxSize() == 0, "ModUp src aux size mismatch");
  AssertTrue(static_cast<int>(dst.size()) == beta_, "ModUp dst size mismatch");

  // Do NTT
  DeviceVector<word> src_intt(num_q_primes * degree);
  DvView<word> src_intt_view = src_intt.View(0, 0);
  ntt_handler_.INTTAndMultConst(src_intt_view, np, src,
                                mod_up1_.ConstView(0, 0), true);

  int padded_num_q_primes = num_q_primes + prime_offset;
  // Do some checks and copy some values from src to dst
  for (int i = 0; i < beta_; i++) {
    // absolute index
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) continue;

    // relative index
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    int src_len = prime_index_end - prime_index_start;
    DvView<word> &dst_i = dst.at(i);

    AssertTrue(dst_i.AuxSize() == num_aux_ * degree,
               "ModUp dst aux size mismatch");
    AssertTrue(dst_i.QSize() == num_q_primes * degree,
               "ModUp dst q size mismatch");

    // Copy values from src to dst (asynchronously)
    cudaMemcpyAsync(dst_i.data() + prime_index_start * degree,
                    src.data() + prime_index_start * degree,
                    src_len * degree * sizeof(word), cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy);
  }

  np.num_aux_ = num_aux_;
  const word *primes = param_.GetPrimesPtr(np);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np);

  for (int i = 0; i < beta_; i++) {
    // absolute index
    int prime_index_start = i * num_aux_;
    int prime_index_end = Min((i + 1) * num_aux_, padded_num_q_primes);
    if (prime_index_end <= prime_offset) continue;

    // relative index
    prime_index_start = Max(prime_index_start - prime_offset, 0);
    prime_index_end = prime_index_end - prime_offset;
    int src_len = prime_index_end - prime_index_start;
    int dst_len = num_q_primes - src_len + num_aux_;
    DvView<word> &dst_i = dst.at(i);

    dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                  DivCeil(dst_len, kLimbBatching * kNumThreadsY));
    dim3 block_dim(kNumThreadsX, kNumThreadsY);
    // smem_size for bconv_table
    int smem_size =
        src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
    // extra for src
    smem_size +=
        kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);

    const signed_word *src_ptr = reinterpret_cast<const signed_word *>(
        src_intt.data() + prime_index_start * degree);

    kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
        dst_i.data(), primes, inv_primes, src_len, dst_len, prime_index_start,
        prime_index_end, src_ptr, mod_up2_.at(i).data());
    ntt_handler_.NTTForModUp(dst_i, np, prime_index_start, prime_index_end,
                             dst_i);
  }
}

template <typename word>
void ModSwitchHandler<word>::ModDown(DvView<word> &dst,
                                     const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == np.num_aux_ * degree,
             "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == num_q_primes * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::ModDown);
}

template <typename word>
void ModSwitchHandler<word>::Rescale(DvView<word> &dst,
                                     const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  NPInfo next_np = param_.LevelToNP(level_ - 1);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == 0, "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == next_np.GetNumQ() * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::Rescale);
}

template <typename word>
void ModSwitchHandler<word>::ModDownAndRescale(
    DvView<word> &dst, const DvConstView<word> &src) const {
  int degree = param_.degree_;
  NPInfo np = param_.LevelToNP(level_, num_aux_);
  NPInfo next_np = param_.LevelToNP(level_ - 1);
  int num_q_primes = np.GetNumQ();

  AssertTrue(src.AuxSize() == np.num_aux_ * degree,
             "ModDown: src aux size mismatch");
  AssertTrue(src.QSize() == num_q_primes * degree,
             "ModDown: src q size mismatch");
  AssertTrue(dst.AuxSize() == 0, "ModDown: dst aux size mismatch");
  AssertTrue(dst.QSize() == next_np.GetNumQ() * degree,
             "ModDown: dst q size mismatch");

  ModDownWorker(dst, src, ModDownType::ModDownAndRescale);
}

template <typename word>
void ModSwitchHandler<word>::ModDownWorker(DvView<word> &dst,
                                           const DvConstView<word> &src,
                                           ModDownType type) const {
  using signed_word = make_signed_t<word>;
  int degree = param_.degree_;
  int num_src_aux = (type == ModDownType::Rescale ? 0 : num_aux_);
  NPInfo np_src = param_.LevelToNP(level_, num_src_aux);
  NPInfo np_dst =
      param_.LevelToNP((type == ModDownType::ModDown ? level_ : level_ - 1), 0);
  NPInfo np_non_intt(Min(np_src.num_main_, np_dst.num_main_),
                     Min(np_src.num_ter_, np_dst.num_ter_), 0);

  int src_len = np_src.GetNumTotal() - np_non_intt.GetNumTotal();
  int dst_len = np_dst.GetNumTotal();

  DeviceVector<word> src_extend_temp;
  if (!kFuseModDownEpilogue && type != ModDownType::ModDown) {
    src_extend_temp.resize(src.TotalSize());
    std::vector<DvView<word>> src_extend_temp_view{
        src_extend_temp.View(src.AuxSize(), 0)};
    DvConstView<word> padding_view(entire_padding_.data(), np_src.GetNumTotal(),
                                   num_src_aux);
    elem_handler_.MultConst(src_extend_temp_view, np_src, {src}, padding_view);
  }
  DvConstView<word> src_view =
      (!kFuseModDownEpilogue && type != ModDownType::ModDown)
          ? src_extend_temp.ConstView(src.AuxSize())
          : src;

  // Performing INTTForModDown
  DeviceVector<word> src_intt(src_len * degree);

  DvView<word> src_intt_view = src_intt.View(0, 0);
  DvConstView<word> const1 =
      (type == ModDownType::ModDown
           ? mod_down1_.ConstView(num_src_aux)
           : (type == ModDownType::Rescale
                  ? rescale1_.ConstView()
                  : mod_down_rescale1_.ConstView(num_src_aux)));
  ntt_handler_.INTTForModDown(src_intt_view, np_src, np_non_intt, src_view,
                              const1);

  const word *primes = param_.GetPrimesPtr(np_dst);
  const signed_word *inv_primes = param_.GetInvPrimesPtr(np_dst);

  const signed_word *src_ptr =
      reinterpret_cast<const signed_word *>(src_intt.data());
  const signed_word *bconv_table =
      (type == ModDownType::ModDown
           ? mod_down2_.data()
           : (type == ModDownType::Rescale ? rescale2_.data()
                                           : mod_down_rescale2_.data()));

  // Do matrix multiplication
  dim3 grid_dim(degree / kUnrollNumber / kNumThreadsX,
                DivCeil(dst_len, kLimbBatching * kNumThreadsY));
  dim3 block_dim(kNumThreadsX, kNumThreadsY);
  // smem_size for bconv_table
  int smem_size =
      src_len * (kLimbBatching * kNumThreadsY) * sizeof(signed_word);
  // extra for src
  smem_size +=
      kMaxNumAccum * (kUnrollNumber * kNumThreadsX) * sizeof(signed_word);

  kernel::ModSwitchMatrixMult<word><<<grid_dim, block_dim, smem_size>>>(
      dst.data(), primes, inv_primes, src_len, dst_len, 0, 0, src_ptr,
      bconv_table);

  // Prepare the constants for ModDownEpilogue
  int pad_start = 0;
  int pad_end = dst_len;
  // rescale case
  if (type != ModDownType::ModDown) {
    pad_start = rescale_pad_start_;
    pad_end = rescale_pad_end_;
  }
  int pad_len = pad_end - pad_start;

  DvConstView<word> inv_prime_prod =
      (type == ModDownType::ModDown
           ? inv_prime_prod_.ConstView()
           : (type == ModDownType::Rescale
                  ? rescale_inv_prime_prod_.ConstView()
                  : mod_down_rescale_inv_prime_prod_.ConstView()));
  DvConstView<word> src2_padding(
      (type == ModDownType::ModDown
           ? DvConstView<word>(nullptr, 0, 0)
           : (type == ModDownType::Rescale
                  ? rescale_padding_.ConstView()
                  : mod_down_rescale_padding_.ConstView())));

  // Prepared offseted src2
  int src2_offset = Max(0, np_src.num_ter_ - np_dst.num_ter_);
  DvConstView<word> src2(src_view.data() + src2_offset * degree,
                         pad_len * degree, 0);

  if (kFuseModDownEpilogue) {
    ntt_handler_.NTTForModDown(dst, np_dst, np_non_intt, DvConstView<word>(dst),
                               src2, inv_prime_prod, src2_padding);

  } else {
    ntt_handler_.NTT(dst, np_dst, DvConstView<word>(dst), false);

    // (src - dst) * inv_prime_prod;
    std::vector<DvView<word>> dst_offset_view{
        DvView<word>(dst.data() + pad_start * degree, pad_len * degree)};
    std::vector<DvConstView<word>> dst_offset_const_view{
        DvConstView<word>(dst.data() + pad_start * degree, pad_len * degree)};
    std::vector<DvView<word>> dst_view{dst};

    elem_handler_.Neg(dst_view, np_dst, {DvConstView<word>(dst)});
    elem_handler_.Add(dst_offset_view, np_non_intt, {src2},
                      dst_offset_const_view);

    elem_handler_.MultConst(dst_view, np_dst, {DvConstView<word>(dst)},
                            inv_prime_prod);
  }
}

// explicit instantiation
template class ModSwitchHandler<uint32_t>;
template class ModSwitchHandler<uint64_t>;

}  // namespace cheddar