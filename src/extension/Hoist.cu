#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/ConstantMemory.cuh"
#include "extension/Hoist.h"

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

// Fused kernel for KeyMult, MAC, and Aut in the baby step.
template <typename word, int num_accum_padded>
__global__ void BSFusedKernel(
    word **dst_bx, word **dst_ax, const word **mod_up, const word **key_bx,
    const word **key_ax, int num_accum, int num_rotations, const word *primes,
    const make_signed_t<word> *inv_primes, int num_q_primes, word *key_extra,
    const word *input_bx_pseudo_modup, word *galois_factors) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int log_degree = cm_log_degree();
  int prime_index = (i >> log_degree);
  int x_idx = i & ((1 << log_degree) - 1);
  int mod_up_index = i;
  const word prime = primes[prime_index];
  const make_signed_t<word> montgomery = inv_primes[prime_index];
  // do not need to synchronize here
  word mod_up_a[num_accum_padded];
  for (int i = 0; i < num_accum; i++) {
    mod_up_a[i] = basic::StreamingLoad(mod_up[i] + mod_up_index);
  }

  word input_bx_pseudo_modup_value = 0;
  if (prime_index < num_q_primes) {
    input_bx_pseudo_modup_value =
        basic::StreamingLoad(input_bx_pseudo_modup + i);
  }
  for (int k = 0; k < num_rotations; k++) {
    word galois_factor = galois_factors[k];
    auto dst_index = basic::BitReverse(x_idx, log_degree + 1) + 1;
    dst_index = dst_index * galois_factor - 1;
    dst_index = basic::BitReverse(dst_index, log_degree + 1);

    word res_bx_value = 0;
    word res_ax_value = 0;
    for (int j = 0; j < num_accum; j++) {
      int key_index = i;
      if (prime_index >= num_q_primes) {
        key_index += key_extra[k];
      }
      word mod_up_value = mod_up_a[j];
      word key_ax_value =
          basic::StreamingLoad(key_ax[j + k * num_accum] + key_index);
      word key_bx_value =
          basic::StreamingLoad(key_bx[j + k * num_accum] + key_index);
      word mult =
          basic::MultMontgomery(mod_up_value, key_bx_value, prime, montgomery);
      res_bx_value = basic::Add(res_bx_value, mult, prime);

      mult =
          basic::MultMontgomery(mod_up_value, key_ax_value, prime, montgomery);
      res_ax_value = basic::Add(res_ax_value, mult, prime);
    }
    if (prime_index < num_q_primes) {
      res_bx_value =
          basic::Add(res_bx_value, input_bx_pseudo_modup_value, prime);
    }
    dst_bx[k][dst_index + (prime_index << log_degree)] = res_bx_value;
    dst_ax[k][dst_index + (prime_index << log_degree)] = res_ax_value;
  }
}

// Fused kernel for plaintext multiplication and accumulation in the giant step.
template <typename word, int num_bs_padded>
__global__ void GSFusedKernel(word **dst_bx, word **dst_ax, const word **bx,
                              const word **ax, const word **mx, int num_bs,
                              int num_gs, const word *primes,
                              const make_signed_t<word> *inv_primes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int log_degree = cm_log_degree();
  int prime_index = (i >> log_degree);
  int bxax_index = i;
  int cx_index = i;

  const word prime = primes[prime_index];
  const make_signed_t<word> montgomery = inv_primes[prime_index];

  word bx_[num_bs_padded];
  word ax_[num_bs_padded];
  for (int i = 0; i < num_bs; i++) {
    bx_[i] = basic::StreamingLoad(bx[i] + bxax_index);
    ax_[i] = basic::StreamingLoad(ax[i] + bxax_index);
  }

  for (int k = 0; k < num_gs; k++) {
    word res_bx = 0;
    word res_ax = 0;
    for (int j = 0; j < num_bs; j++) {
      if (mx[j + k * num_bs] == nullptr) continue;
      word mx_value = basic::StreamingLoad(mx[j + k * num_bs] + cx_index);
      word bx_value = bx_[j];
      word ax_value = ax_[j];
      word mult = basic::MultMontgomery(bx_value, mx_value, prime, montgomery);
      res_bx = basic::Add(res_bx, mult, prime);
      mult = basic::MultMontgomery(ax_value, mx_value, prime, montgomery);
      res_ax = basic::Add(res_ax, mult, prime);
    }
    dst_bx[k][i] = res_bx;
    dst_ax[k][i] = res_ax;
  }
}
}  // namespace kernel

template <typename word>
void HoistHandler<word>::ExtractBSIndices(const PlainHoistMap &hoist_map) {
  for (const auto &[_, bs_map] : hoist_map) {
    for (const auto &[bs_idx, _] : bs_map) {
      bs_indices_.insert(bs_idx);
    }
  }
}

template <typename word>
void HoistHandler<word>::CompilePlaintexts(ConstContextPtr<word> context,
                                           const PlainHoistMap &hoist_map) {
  for (const auto &[gs_idx, bs_map] : hoist_map) {
    hoist_pt_map_.try_emplace(gs_idx, std::map<int, Pt>{});
    gs_indices_.push_back(gs_idx);
    for (const auto &[bs_idx, message] : bs_map) {
      hoist_pt_map_.at(gs_idx).try_emplace(bs_idx, NPInfo(0, 0, 0));
      int num_p_primes = context->param_.alpha_;

      context->encoder_.Encode(hoist_pt_map_.at(gs_idx).at(bs_idx), pt_level_,
                               pt_scale_, message, num_p_primes);
    }
  }
  std::sort(gs_indices_.begin(), gs_indices_.end());
}

template <typename word>
HoistHandler<word>::HoistHandler(ConstContextPtr<word> context,
                                 const PlainHoistMap &hoist_map, int pt_level,
                                 double pt_scale, bool suppress_bs_swap)
    : pt_level_(pt_level), pt_scale_(pt_scale) {
  AssertTrue(hoist_map.size() > 0, "Hoist: hoist_map should not be empty");
  if (kOptimizeAutomorphism && hoist_map.size() == 1 &&
      hoist_map.begin()->first == 0 && !suppress_bs_swap) {
    // when gs = 1, swap bs and gs.
    PlainHoistMap optimized;

    for (const auto &[bs_idx, msg] : hoist_map.begin()->second) {
      int msg_size = msg.size();
      optimized.try_emplace(bs_idx, std::map<int, Message>{});

      optimized[bs_idx].try_emplace(0, msg_size);
      for (int i = 0; i < msg_size; i++) {
        optimized[bs_idx][0][(i + bs_idx) % msg_size] = msg.at(i);
      }
    }
    ExtractBSIndices(optimized);
    CompilePlaintexts(context, optimized);
  } else {
    ExtractBSIndices(hoist_map);
    CompilePlaintexts(context, hoist_map);
  }
  if (!cm_populated_) {
    PopulateConstantMemory(context->param_);
    cm_populated_ = true;
  }
}

template <typename word>
std::pair<int, int> HoistHandler<word>::CheckStrideMinKS() const {
  int non_zero_bs = 0;
  int bs_gcd = 0;
  int max_bs = 0;
  for (const auto &bs_idx : bs_indices_) {
    if (bs_idx == 0) continue;
    bs_gcd = GCD(bs_gcd, bs_idx);  // it's okay to put 0 in GCD
    non_zero_bs += 1;
    max_bs = Max(max_bs, bs_idx);
  }
  AssertTrue(non_zero_bs * bs_gcd == max_bs,
             "MinKS does not apply to the given hoist map baby-step sequence");

  int non_zero_gs = 0;
  int gs_gcd = 0;
  int max_gs = 0;
  for (const auto &gs_idx : gs_indices_) {
    if (gs_idx == 0) continue;
    gs_gcd = GCD(gs_gcd, gs_idx);  // it's okay to put 0 in GCD
    non_zero_gs += 1;
    max_gs = Max(max_gs, gs_idx);
  }
  AssertTrue(non_zero_gs * gs_gcd == max_gs,
             "MinKS does not apply to the given hoist map giant-step sequence");
  return {bs_gcd, gs_gcd};
}

template <typename word>
void HoistHandler<word>::BSFusedKeyMult(
    ConstContextPtr<word> context, std::map<int, Ct> &res,
    std::vector<Dv> &a_modup, const Ct &a_orig, const EvkMap<word> &keys,
    std::vector<int> &rotations, const Dv &input_bx_pseudo_modup) const {
  NPInfo a_orig_np = a_orig.GetNP();
  int level = context->param_.NPToLevel(a_orig_np);
  int num_main = a_orig_np.num_main_;
  int num_ter = a_orig_np.num_ter_;
  int num_aux = keys.GetRotationKey(rotations[0]).GetNP().num_aux_;
  int num_q = num_main + num_ter;
  int prime_offset = context->param_.GetMaxNumTer() - num_ter;

  for (auto &[_, accum] : res) {
    AssertTrue(&accum != &a_orig,
               "In-place operation is not supported for MultKeyNoModDown");
  }

  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);

  NPInfo np(num_main, num_ter, num_aux);

  for (auto &[_, accum] : res) {
    accum.RemoveRx();
    accum.ModifyNP(np);
    accum.SetScale(a_orig.GetScale());
    accum.SetNumSlots(a_orig.GetNumSlots());
  }

  int num_accum = 0;
  for (int i = 0; i < beta; i++) {
    int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) continue;
    num_accum++;
  }
  int num_accum_offset = beta - num_accum;
  int num_q_primes = a_orig.GetNP().GetNumQ();
  int num_rotations = rotations.size();

  // We can further optimize this part. We can call copy only once if we pack
  // the pointers for all accumulators together.
  // We can also consider passing the pointers as kernel arguments.

  // ready ptrs to copy to device.
  HostVector<const word *> modup_ptrs(num_accum, nullptr);
  HostVector<const word *> key_a_ptrs(num_accum * num_rotations, nullptr);
  HostVector<const word *> key_b_ptrs(num_accum * num_rotations, nullptr);
  HostVector<word *> dst_b_ptrs(num_rotations, nullptr);
  HostVector<word *> dst_a_ptrs(num_rotations, nullptr);
  HostVector<word> key_extra(num_rotations, 0);
  HostVector<word> galois_factors_h(num_rotations, 0);

  for (int i = 0; i < num_rotations; i++) {
    // keys
    for (int j = 0; j < num_accum; j++) {
      key_b_ptrs[i * num_accum + j] =
          keys.GetRotationKey(rotations[i]).bx_[j + num_accum_offset].data() +
          prime_offset * context->param_.degree_;
      key_a_ptrs[i * num_accum + j] =
          keys.GetRotationKey(rotations[i]).ax_[j + num_accum_offset].data() +
          prime_offset * context->param_.degree_;
    }
    const Evk &key = keys.GetRotationKey(rotations[i]);
    DvConstView<word> key_view = key.AxConstView(0, prime_offset);
    key_extra[i] = key_view.QSize() - num_q_primes * context->param_.degree_;

    // dst
    dst_b_ptrs[i] = res.at(rotations[i]).bx_.data();
    dst_a_ptrs[i] = res.at(rotations[i]).ax_.data();

    // galois factor
    int permute_amount = rotations[i];
    if (permute_amount == -1) {
      AssertTrue(false, "Conjugate case should not be handled.");
    }
    AssertTrue(
        permute_amount >= 0 && permute_amount < context->param_.degree_ / 2,
        "Permute: Invalid permute amount");
    galois_factors_h[i] = context->param_.GetGaloisFactor(
        context->param_.degree_ / 2 - permute_amount);
  }
  for (int i = 0; i < num_accum; i++) {
    modup_ptrs[i] = a_modup[i + num_accum_offset].data();
  }
  int num_primes = np.GetNumTotal();

  DeviceVector<const word *> modup_d_ptrs(num_accum);
  DeviceVector<const word *> key_a_d_ptrs(num_accum * num_rotations);
  DeviceVector<const word *> key_b_d_ptrs(num_accum * num_rotations);
  DeviceVector<word *> dst_b_d_ptrs(num_rotations);
  DeviceVector<word *> dst_a_d_ptrs(num_rotations);
  DeviceVector<word> key_extra_d(num_rotations);
  DeviceVector<word> galois_factors(num_rotations);
  CopyHostToDevice(modup_d_ptrs, modup_ptrs);
  CopyHostToDevice(key_a_d_ptrs, key_a_ptrs);
  CopyHostToDevice(key_b_d_ptrs, key_b_ptrs);
  CopyHostToDevice(dst_b_d_ptrs, dst_b_ptrs);
  CopyHostToDevice(dst_a_d_ptrs, dst_a_ptrs);
  CopyHostToDevice(key_extra_d, key_extra);
  CopyHostToDevice(galois_factors, galois_factors_h);

  const word *primes = context->param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = context->param_.GetInvPrimesPtr(np);
  dim3 block_dim(kernel_block_dim_);
  dim3 grid_dim(num_primes * context->param_.degree_ / kernel_block_dim_);

  AssertTrue(num_accum <= (1 << max_log_beta_),
             "num_accum should not be greater than " +
                 std::to_string(1 << max_log_beta_));
  constexpr_for<1, max_log_beta_ + 1>([&](auto i) {
    constexpr int num_accum_padded = 1 << i;
    if (num_accum > num_accum_padded) return;
    if (num_accum <= (1 << (i - 1))) return;
    kernel::BSFusedKernel<word, num_accum_padded><<<grid_dim, block_dim>>>(
        dst_b_d_ptrs.data(), dst_a_d_ptrs.data(), modup_d_ptrs.data(),
        key_b_d_ptrs.data(), key_a_d_ptrs.data(), num_accum, num_rotations,
        primes, inv_primes, num_q_primes, key_extra_d.data(),
        input_bx_pseudo_modup.data(), galois_factors.data());
  });
}

template <typename word>
void HoistHandler<word>::GSFusedPAccum(ConstContextPtr<word> context,
                                       std::map<int, Ct> &results,
                                       const std::vector<int> &gs_indices,
                                       const std::map<int, Ct> &bs) const {
  constexpr int kernel_block_dim_ = 256;

  // Check if all bs and pt have the same scale and number of primes
  const Ct &first_ct = bs.begin()->second;
  const Pt &first_pt = hoist_pt_map_.at(gs_indices.front()).begin()->second;
  double scale = first_ct.GetScale() * first_pt.GetScale();
  int num_slots = Max(first_ct.GetNumSlots(), first_pt.GetNumSlots());
  int num_q_primes = first_ct.GetNP().GetNumQ();
  int num_p_primes = first_ct.GetNP().num_aux_;
  NPInfo np = first_ct.GetNP();
  // bool bs_first = true;
  for (auto giant_step : gs_indices) {
    const std::map<int, Pt> &pt_map = hoist_pt_map_.at(giant_step);
    for (const auto &[bs_idx, pt] : pt_map) {
      const auto &ct = bs.at(bs_idx);
      AssertTrue(num_q_primes == ct.GetNP().GetNumQ(),
                 "Hoist: number of q primes mismatch");
      AssertTrue(num_p_primes == ct.GetNP().num_aux_,
                 "Hoist: number of p primes mismatch");
      // we do not check for pt, but num primes must also match between pts
      // but not necessarily between ct <-> pt
      context->AssertSameScale(scale, ct.GetScale() * pt.GetScale());
      num_slots = Max(num_slots, pt.GetNumSlots(), ct.GetNumSlots());
    }
  }
  // Set up results
  for (auto &res : results) {
    res.second.RemoveRx();
    res.second.ModifyNP(np);
    res.second.SetScale(scale);
    res.second.SetNumSlots(num_slots);
  }

  // We can further optimize this part. We can call copy only once if we pack
  // the pointers for all accumulators together.
  // We can also consider passing the pointers as kernel arguments.

  // Ready ptrs to copy to device
  int num_bs = bs_indices_.size();
  int num_gs = gs_indices.size();
  HostVector<const word *> bx_ptrs(num_bs);
  HostVector<const word *> ax_ptrs(num_bs);
  HostVector<const word *> mx_ptrs(num_gs * num_bs);
  HostVector<word *> dst_b_ptrs(num_gs);
  HostVector<word *> dst_a_ptrs(num_gs);
  int num_primes = np.GetNumTotal();

  // ptrs for (b,a) of each bs
  int idx = 0;
  for (auto bs_idx : bs_indices_) {
    const auto &ct = bs.at(bs_idx);
    bx_ptrs[idx] = ct.bx_.data();
    ax_ptrs[idx] = ct.ax_.data();
    idx++;
  }

  // ptrs for plaintexts of each (gs, bs)
  idx = 0;
  for (auto gs_idx : gs_indices) {
    const auto &pt_map = hoist_pt_map_.at(gs_idx);
    for (auto bs_idx : bs_indices_) {
      if (pt_map.find(bs_idx) == pt_map.end()) {
        mx_ptrs[idx] = nullptr;
      } else {
        mx_ptrs[idx] = pt_map.at(bs_idx).mx_.data();
      }
      idx++;
    }
  }

  idx = 0;
  for (const auto gs_idx : gs_indices) {
    auto &res = results.at(gs_idx);
    dst_b_ptrs[idx] = res.bx_.data();
    dst_a_ptrs[idx] = res.ax_.data();
    idx++;
  }

  // Copy to device
  DeviceVector<const word *> bx_d_ptrs(num_bs);
  DeviceVector<const word *> ax_d_ptrs(num_bs);
  DeviceVector<const word *> mx_d_ptrs(num_gs * num_bs);
  DeviceVector<word *> dst_b_d_ptrs(num_gs);
  DeviceVector<word *> dst_a_d_ptrs(num_gs);
  CopyHostToDevice(bx_d_ptrs, bx_ptrs);
  CopyHostToDevice(ax_d_ptrs, ax_ptrs);
  CopyHostToDevice(mx_d_ptrs, mx_ptrs);
  CopyHostToDevice(dst_b_d_ptrs, dst_b_ptrs);
  CopyHostToDevice(dst_a_d_ptrs, dst_a_ptrs);

  const word *primes = context->param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = context->param_.GetInvPrimesPtr(np);
  dim3 block_dim(kernel_block_dim_);
  dim3 grid_dim(num_primes * context->param_.degree_ / kernel_block_dim_);

  constexpr_for<1, max_log_bs_ + 1>([&](auto i) {
    constexpr int num_bs_padded = 1 << i;
    if (num_bs > num_bs_padded) return;
    if (num_bs <= (1 << (i - 1))) return;
    kernel::GSFusedKernel<word, num_bs_padded><<<grid_dim, block_dim>>>(
        dst_b_d_ptrs.data(), dst_a_d_ptrs.data(), bx_d_ptrs.data(),
        ax_d_ptrs.data(), mx_d_ptrs.data(), num_bs, num_gs, primes, inv_primes);
  });
}

template <typename word>
void HoistHandler<word>::EvaluateSingleAccum(ConstContextPtr<word> context,
                                             Ct &res,
                                             const std::map<int, Ct> &bs,
                                             const std::map<int, Pt> &pt_map,
                                             bool inplace) const {
  std::vector<std::vector<DvConstView<word>>> bs_bx_ax_view;
  std::vector<DvConstView<word>> pt_mx_view;

  double scale = 0;
  int num_slots = 0;

  NPInfo np;
  bool first = true;
  for (const auto &[bs_idx, pt] : pt_map) {
    const auto &ct = bs.at(bs_idx);
    if (first) {
      scale = ct.GetScale() * pt.GetScale();
      np = ct.GetNP();
      first = false;
    } else {
      context->AssertSameScale(scale, ct.GetScale() * pt.GetScale());
      AssertTrue(np.GetNumQ() == ct.GetNP().GetNumQ(),
                 "Hoist: number of q primes mismatch");
      AssertTrue(np.num_aux_ == ct.GetNP().num_aux_,
                 "Hoist: number of p primes mismatch");
      // we do not check for pt, but num primes must also match between
      // pts but not necessarily between ct <-> pt
    }
    num_slots = Max(num_slots, pt.GetNumSlots(), ct.GetNumSlots());

    bs_bx_ax_view.push_back(ct.ConstViewVector());
    pt_mx_view.push_back(pt.ConstView());
  }

  res.RemoveRx();
  res.ModifyNP(np);
  res.SetScale(scale);
  res.SetNumSlots(num_slots);
  std::vector<DvView<word>> res_bx_ax_view = res.ViewVector();
  // Add res to last if inplace
  if (inplace) {
    bs_bx_ax_view.push_back(res.ConstViewVector());
  }

  context->elem_handler_.PAccum(res_bx_ax_view, np, bs_bx_ax_view, pt_mx_view);
}

template <typename word>
void HoistHandler<word>::EvaluateFinalModDown(ConstContextPtr<word> context,
                                              Ct &res, Ct &final_accum,
                                              int input_num_slots,
                                              double input_scale) const {
  NPInfo next_np = context->param_.LevelToNP(pt_level_ - 1);
  auto &mod_switcher = context->mod_switch_handlers_.at(pt_level_);

  NPInfo np(next_np.num_main_, next_np.num_ter_, 0);

  // this may not be handled properly by intermediate operations
  res.ModifyNP(np);

  int num_slots = input_num_slots;
  for (auto &[_, pt_map] : hoist_pt_map_) {
    for (auto &[_, pt] : pt_map) {
      num_slots = Max(num_slots, pt.GetNumSlots());
    }
  }
  res.SetNumSlots(num_slots);

  DvView<word> res_bx = res.BxView();
  DvView<word> res_ax = res.AxView();
  mod_switcher.ModDownAndRescale(res_bx, final_accum.BxConstView());
  mod_switcher.ModDownAndRescale(res_ax, final_accum.AxConstView());

  res.SetScale(pt_scale_ * input_scale /
               context->param_.GetRescalePrimeProd(pt_level_));
}

template <typename word>
void HoistHandler<word>::EvaluateMinKSBabyStep(
    ConstContextPtr<word> context, std::map<int, Ct> &bs, const Ct &input,
    const EvkMap<word> &evk_map) const {
  auto [bs_stride, _] = CheckStrideMinKS();
  AssertTrue(bs.empty(), "Hoist: bs should be empty");
  Ct tmp;
  for (const auto &bs_idx : bs_indices_) {
    bs.try_emplace(bs_idx, NPInfo(0, 0, 0));
    if (bs_idx == 0) {
      context->Copy(bs[0], input);
    } else {
      int prev_bs_idx = bs_idx - bs_stride;
      AssertTrue(bs.find(prev_bs_idx) != bs.end(),
                 "Hoist: MinKS baby-step sequence is not complete");
      context->HRot(bs[bs_idx], bs[prev_bs_idx],
                    evk_map.GetRotationKey(bs_stride), bs_stride);
    }
  }
}

template <typename word>
void HoistHandler<word>::EvaluateMinKSGiantStep(
    ConstContextPtr<word> context, Ct &res, const std::map<int, Ct> &bs,
    const EvkMap<word> &evk_map) const {
  // Reversed traversal
  AssertFalse(bs.empty(), "Hoist: bs should not be empty");

  auto [_, gs_stride] = CheckStrideMinKS();
  Ct accum;
  int prev_gs_idx = 0;
  bool first = true;
  for (auto it = hoist_pt_map_.rbegin(); it != hoist_pt_map_.rend(); it++) {
    int gs_idx = it->first;
    const auto &pt_map = it->second;
    EvaluateSingleAccum(context, accum, bs, pt_map, !first);
    if (!first) {
      AssertTrue(prev_gs_idx == gs_idx + gs_stride,
                 "Hoist: MinKS giant-step sequence is not complete");
    }
    if (gs_idx != 0) {
      context->HRot(accum, accum, evk_map.GetRotationKey(gs_stride), gs_stride);
    } else {
      AssertTrue(first || prev_gs_idx == gs_stride,
                 "Hoist: MinKS giant-step sequence is not complete");
    }
    first = false;
    prev_gs_idx = gs_idx;
  }
  context->Rescale(res, accum);
}

template <typename word>
void HoistHandler<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                                  const Ct &input, const EvkMap<word> &evk_map,
                                  bool min_ks) const {
  std::map<int, Ct> bs;
  EvaluateBabyStep(context, bs, input, evk_map, min_ks);
  EvaluateGiantStep(context, res, bs, evk_map, min_ks);
}

template <typename word>
void HoistHandler<word>::EvaluateBabyStep(ConstContextPtr<word> context,
                                          std::map<int, Ct> &bs,
                                          const Ct &input,
                                          const EvkMap<word> &evk_map,
                                          bool min_ks) const {
  NPInfo input_np = input.GetNP();
  int num_main_primes = input_np.num_main_;
  int num_ter_primes = input_np.num_ter_;
  int num_q_primes = input_np.GetNumQ();
  int num_p_primes = context->param_.alpha_;
  int prime_offset = context->param_.GetMaxNumTer() - input_np.num_ter_;
  int beta = DivCeil(num_q_primes + prime_offset, num_p_primes);
  int degree = context->param_.degree_;
  AssertTrue(num_q_primes == context->param_.LevelToNP(pt_level_).GetNumQ(),
             "Hoist: input level mismatch");
  AssertTrue(input.GetNP().num_aux_ == 0, "Hoist: input should be mod-down");
  AssertFalse(input.HasRx(), "Hoist: input should be relinearized");

  if (bs_indices_.size() == 1 && *bs_indices_.begin() == 0) {
    bs.try_emplace(0, NPInfo(num_main_primes, num_ter_primes, 0));
    context->Copy(bs[0], input);
    return;
  }

  if (min_ks) {
    EvaluateMinKSBabyStep(context, bs, input, evk_map);
    return;
  }

  auto &mod_switcher = context->mod_switch_handlers_.at(pt_level_);

  // hoisted evaluation
  AssertTrue(bs.empty(), "Hoist: bs should be empty");

  // 1. ModUp
  std::vector<Dv> tmp_modup;
  std::vector<DvView<word>> tmp_modup_view;
  for (int i = 0; i < beta; i++) {
    tmp_modup.emplace_back((num_q_primes + num_p_primes) * degree);
    tmp_modup_view.push_back(tmp_modup[i].View(num_p_primes * degree));
  }
  mod_switcher.ModUp(tmp_modup_view, input.AxConstView());

  // 2. Baby-step rotations
  Dv *input_bx_pseudo_modup;

  DvConstView<word> p_prod_view(context->p_prod_.data() + prime_offset,
                                num_q_primes);
  Dv pseudo_modup_tmp;

  NPInfo modup_np(num_main_primes, num_ter_primes, num_p_primes);

  // Special handling for bs_idx = 0 case
  if (bs_indices_.find(0) == bs_indices_.end()) {
    pseudo_modup_tmp.resize(num_q_primes * degree);
    DvView<word> pseudo_modup_tmp_view = pseudo_modup_tmp.View();
    mod_switcher.PseudoModUp(pseudo_modup_tmp_view, input.BxConstView(),
                             p_prod_view);
    pseudo_modup_tmp.ZeroExtend(num_p_primes * degree);
    input_bx_pseudo_modup = &pseudo_modup_tmp;
  } else {
    bs.try_emplace(0, input_np);
    bs[0].SetScale(input.GetScale());
    bs[0].SetNumSlots(input.GetNumSlots());
    DvView<word> bs_0_bx_view = bs[0].BxView();
    DvView<word> bs_0_ax_view = bs[0].AxView();
    mod_switcher.PseudoModUp(bs_0_bx_view, input.BxConstView(), p_prod_view);
    mod_switcher.PseudoModUp(bs_0_ax_view, input.AxConstView(), p_prod_view);
    bs[0].bx_.ZeroExtend(num_p_primes * degree);
    bs[0].ax_.ZeroExtend(num_p_primes * degree);
    bs[0].ModifyNP(modup_np);
    input_bx_pseudo_modup = &(bs[0].bx_);
  }

  // We can fuse KeyMult, MAC, and Automorphism together for better performance,
  // it reduces the number of global memory reads/writes for the intermediate
  // results and ModUp(a).
  bool can_fuse_bs = beta <= (1 << max_log_beta_);
  if (kFuseBSKeyMult && can_fuse_bs) {
    std::vector<int> rotations;
    for (const auto &bs_idx : bs_indices_) {
      if (bs_idx != 0) {
        bs.try_emplace(bs_idx, modup_np);
        rotations.push_back(bs_idx);
      }
    }
    BSFusedKeyMult(context, bs, tmp_modup, input, evk_map, rotations,
                   *input_bx_pseudo_modup);
  } else {
    Ct tmp(modup_np);

    for (const auto &bs_idx : bs_indices_) {
      if (bs_idx != 0) {
        const auto &key = evk_map.GetRotationKey(bs_idx);
        bs.try_emplace(bs_idx, modup_np);

        // KeyMult
        context->MultKeyNoModDown(tmp, tmp_modup, input, key);
        DvView<word> tmp_bx_q_view(tmp.bx_.data(), num_q_primes * degree, 0);
        std::vector<DvView<word>> tmp_bx_view = {tmp_bx_q_view};
        std::vector<DvConstView<word>> tmp_bx_const_view = {tmp_bx_q_view};
        std::vector<DvConstView<word>> input_bx_pseudo_modup_view = {
            DvConstView<word>((*input_bx_pseudo_modup).data(),
                              num_q_primes * degree, 0)};
        // MAC
        context->elem_handler_.Add(tmp_bx_view, input_np, tmp_bx_const_view,
                                   {(*input_bx_pseudo_modup).ConstView()});
        // Automorphism
        context->Permute(bs[bs_idx], tmp, bs_idx);
      }
    }
  }
}

// 3. Giant-step accumulation and rotations

template <typename word>
void HoistHandler<word>::EvaluateGiantStep(ConstContextPtr<word> context,
                                           Ct &res, const std::map<int, Ct> &bs,
                                           const EvkMap<word> &evk_map,
                                           bool min_ks) const {
  AssertFalse(bs.empty(), "Hoist: bs should not be empty");
  const Ct &ref_ct = bs.begin()->second;
  NPInfo ref_np = ref_ct.GetNP();
  int num_aux_primes = context->param_.alpha_;
  NPInfo q_prime_np(ref_np.num_main_, ref_np.num_ter_, 0);
  NPInfo q_p_prime_np(ref_np.num_main_, ref_np.num_ter_, num_aux_primes);
  int prime_offset = context->param_.GetMaxNumTer() - ref_np.num_ter_;
  int num_q_primes = ref_np.GetNumQ();
  int padded_num_q_primes = num_q_primes + prime_offset;
  int beta = DivCeil(padded_num_q_primes, num_aux_primes);
  int input_num_slots = ref_ct.GetNumSlots();
  double input_scale = ref_ct.GetScale();
  int degree = context->param_.degree_;
  auto &mod_switcher = context->mod_switch_handlers_.at(pt_level_);

  // when only gs indices exits
  if (bs_indices_.size() == 1 && *bs_indices_.begin() == 0) {
    AssertFalse(min_ks, "Hoist: min_ks should be false for bs == 1 case");
    const Ct &ct = bs.begin()->second;
    std::vector<Dv> ax_modup;
    std::vector<DvView<word>> ax_modup_view;
    for (int i = 0; i < beta; i++) {
      ax_modup.emplace_back((num_q_primes + num_aux_primes) * degree);
      ax_modup_view.push_back(ax_modup[i].View(num_aux_primes * degree));
    }
    mod_switcher.ModUp(ax_modup_view, ct.AxConstView());
    std::map<int, Ct> pt_mult;

    Dv *bx_pseudo = nullptr;
    Dv bx_pseudo_tmp;
    DvConstView<word> p_prod_view(context->p_prod_.data() + prime_offset,
                                  num_q_primes);
    if (hoist_pt_map_.find(0) == hoist_pt_map_.end()) {
      bx_pseudo_tmp.resize(num_q_primes * degree);
      DvConstView<word> ct_bx_view(ct.bx_.data(), num_q_primes * degree, 0);
      DvView<word> bx_pseudo_tmp_view = bx_pseudo_tmp.View();
      mod_switcher.PseudoModUp(bx_pseudo_tmp_view, ct_bx_view, p_prod_view);
      bx_pseudo = &bx_pseudo_tmp;
    }

    std::vector<std::vector<DvConstView<word>>> ct_bx_ax_view;
    std::vector<int> rot_indices;

    Ct final_accum;
    bool inplace = false;
    for (const auto &[gs_idx, pt_map] : hoist_pt_map_) {
      const auto &pt = pt_map.begin()->second;

      if (gs_idx == 0) {
        AssertFalse(inplace, "Hoist: inplace should be false for gs_idx == 0");
        pt_mult.try_emplace(0, q_prime_np);
        DvConstView<word> ct_bx_view(ct.bx_.data(), num_q_primes * degree, 0);
        DvConstView<word> ct_ax_view(ct.ax_.data(), num_q_primes * degree, 0);
        DvView<word> final_accum_bx_view = pt_mult.at(0).BxView();
        DvView<word> final_accum_ax_view = pt_mult.at(0).AxView();
        mod_switcher.PseudoModUp(final_accum_bx_view, ct_bx_view, p_prod_view);
        mod_switcher.PseudoModUp(final_accum_ax_view, ct_ax_view, p_prod_view);
        pt_mult.at(0).bx_.ZeroExtend(num_aux_primes * degree);
        pt_mult.at(0).ax_.ZeroExtend(num_aux_primes * degree);
        pt_mult.at(0).ModifyNP(q_p_prime_np);
        bx_pseudo = &pt_mult.at(0).bx_;
        context->Mult(final_accum, pt_mult.at(0), pt);

        inplace = true;
        continue;
      }

      const auto &key = evk_map.GetRotationKey(gs_idx);

      pt_mult.try_emplace(gs_idx, q_p_prime_np);

      context->MultKeyNoModDown(pt_mult.at(gs_idx), ax_modup, ct, key);
      DvView<word> pt_mult_bx_view(pt_mult.at(gs_idx).bx_.data(),
                                   num_q_primes * degree, 0);
      std::vector<DvView<word>> pt_mult_bx_view_vec = {pt_mult_bx_view};
      std::vector<DvConstView<word>> bx_pseudo_view = {
          DvView<word>((*bx_pseudo).data(), num_q_primes * degree, 0)};
      context->elem_handler_.Add(pt_mult_bx_view_vec, q_prime_np,
                                 {pt_mult_bx_view}, bx_pseudo_view);
      context->Mult(pt_mult.at(gs_idx), pt_mult.at(gs_idx), pt);

      ct_bx_ax_view.push_back(
          {pt_mult.at(gs_idx).BxConstView(), pt_mult.at(gs_idx).AxConstView()});
      rot_indices.push_back(gs_idx);
    }

    if (!inplace) {
      final_accum.ModifyNP(q_p_prime_np);
      final_accum.SetScale(input_scale);
      final_accum.SetNumSlots(input_num_slots);
    }
    if (rot_indices.size() != 0) {
      auto final_accum_view = final_accum.ViewVector();
      auto final_accum_const_view = final_accum.ConstViewVector();
      if (inplace) {
        ct_bx_ax_view.push_back(final_accum_const_view);
      }
      context->elem_handler_.PermuteAccum(final_accum_view, q_p_prime_np,
                                          rot_indices, ct_bx_ax_view);
    }
    EvaluateFinalModDown(context, res, final_accum, input_num_slots,
                         input_scale);
    return;
  }

  if (min_ks) {
    EvaluateMinKSGiantStep(context, res, bs, evk_map);
    return;
  }

  if (kOptimizeAutomorphism) {
    EvaluateGiantStepOptimized(context, res, bs, evk_map);
    return;
  }

  Ct tmp, accum;
  // 3-1. simplified sequence for non-BSGS accumulation.
  if (gs_indices_.size() == 1 && gs_indices_.at(0) == 0) {
    EvaluateSingleAccum(context, accum, bs, hoist_pt_map_.begin()->second);
    EvaluateFinalModDown(context, res, accum, input_num_slots, input_scale);
    return;
  }

  // 3-2. regular BSGS accumulation sequence
  Ct final_accum;
  bool final_accum_init = false;
  std::vector<Dv> tmp_modup;
  std::vector<DvView<word>> tmp_modup_view;
  for (int i = 0; i < beta; i++) {
    tmp_modup.emplace_back((num_q_primes + num_aux_primes) * degree);
    tmp_modup_view.push_back(tmp_modup[i].View(num_aux_primes * degree));
  }
  Dv tmp_moddown(num_q_primes * degree);
  for (const auto &[gs_idx, pt_map] : hoist_pt_map_) {
    if (gs_idx == 0 && final_accum_init == false) {
      EvaluateSingleAccum(context, final_accum, bs, pt_map);
      final_accum_init = true;
      continue;
    } else {
      EvaluateSingleAccum(context, accum, bs, pt_map);
    }

    // giant-step rotation
    if (gs_idx != 0) {
      // Perform mod_down and re mod_up --> mult key(gs_idx) --> permute
      DvView<word> tmp_moddown_view = tmp_moddown.View(0);
      mod_switcher.ModDown(tmp_moddown_view, accum.AxConstView());
      mod_switcher.ModUp(tmp_modup_view, tmp_moddown.ConstView());
      const auto &key = evk_map.GetRotationKey(gs_idx);
      context->MultKeyNoModDown(tmp, tmp_modup, accum, key);
      std::vector<DvView<word>> tmp_bx_view = {tmp.BxView()};

      context->elem_handler_.Add(tmp_bx_view, q_p_prime_np, {tmp.BxConstView()},
                                 {accum.BxConstView()});
      if (!final_accum_init) {
        context->Permute(final_accum, tmp, gs_idx);
        final_accum_init = true;
        continue;
      } else {
        context->Permute(accum, tmp, gs_idx);
      }
    }

    // we cannot use context->Add which checks for the use of the same
    // primes
    context->Add(final_accum, final_accum, accum);
  }
  EvaluateFinalModDown(context, res, final_accum, input_num_slots, input_scale);
}

template <typename word>
void HoistHandler<word>::EvaluateGiantStepOptimized(
    ConstContextPtr<word> context, Ct &res, const std::map<int, Ct> &bs,
    const EvkMap<word> &evk_map) const {
  AssertFalse(bs.empty(), "Hoist: bs should not be empty");
  const Ct &ref_ct = bs.begin()->second;

  NPInfo ref_ct_np = ref_ct.GetNP();
  int num_q_primes = ref_ct_np.GetNumQ();
  int num_p_primes = ref_ct_np.num_aux_;
  int prime_offset = context->param_.GetMaxNumTer() - ref_ct_np.num_ter_;
  int beta = DivCeil(num_q_primes + prime_offset, num_p_primes);
  int input_num_slots = ref_ct.GetNumSlots();
  double input_scale = ref_ct.GetScale();
  int degree = context->param_.degree_;
  auto &mod_switcher = context->mod_switch_handlers_.at(pt_level_);

  Ct tmp;

  // 3-1. simplified sequence for non-BSGS accumulation.
  // But this should not occur in optimized cases
  if (gs_indices_.size() == 1 && gs_indices_.at(0) == 0) {
    EvaluateSingleAccum(context, tmp, bs, hoist_pt_map_.begin()->second);
    EvaluateFinalModDown(context, res, tmp, input_num_slots, input_scale);
    return;
  }

  // 3-2. regular BSGS accumulation sequence
  std::map<int, Ct> accum;
  bool gs_idx_0_exists = false;
  Ct *final_accum;
  accum.try_emplace(0, ref_ct_np);
  final_accum = &accum[0];

  if (hoist_pt_map_.begin() != hoist_pt_map_.end()) gs_idx_0_exists = true;
  for (const auto &[gs_idx, pt_map] : hoist_pt_map_) {
    accum.try_emplace(gs_idx, ref_ct_np);
  }

  // Plaintext multiplication for all baby-step results and accumulation.
  // We can fuse the plaintext multiplication and accumulation.
  bool can_fuse_gs = bs_indices_.size() <= (1 << max_log_bs_);
  if (kFuseGSPAccum && can_fuse_gs) {
    GSFusedPAccum(context, accum, gs_indices_, bs);
  } else {
    for (const auto &[gs_idx, pt_map] : hoist_pt_map_) {
      EvaluateSingleAccum(context, accum.at(gs_idx), bs, pt_map);
    }
  }

  // giant-step rotation and accumulation
  std::vector<Dv> tmp_modup;
  std::vector<DvView<word>> tmp_modup_view;
  for (int i = 0; i < beta; i++) {
    tmp_modup.emplace_back((num_q_primes + num_p_primes) * degree);
    tmp_modup_view.push_back(tmp_modup[i].View(num_p_primes * degree));
  }
  Dv tmp_moddown(num_q_primes * degree);

  bool first = true;
  for (const auto &[gs_idx, ct] : accum) {
    if (gs_idx == 0) continue;

    DvView<word> tmp_moddown_view = tmp_moddown.View(0);
    mod_switcher.ModDown(tmp_moddown_view, ct.AxConstView());
    std::vector<DvView<word>> tmp_modup_view_vector;
    for (int i = 0; i < beta; i++) {
      tmp_modup_view_vector.push_back(tmp_modup[i].View(num_p_primes * degree));
    }

    mod_switcher.ModUp(tmp_modup_view, tmp_moddown_view);
    const auto &key = evk_map.GetRotationKey(gs_idx);

    if (first & !gs_idx_0_exists) {
      context->MultKeyNoModDown(*final_accum, tmp_modup, ct, key);
      context->Permute(*final_accum, *final_accum, gs_idx);
    } else {
      context->MultKeyNoModDown(tmp, tmp_modup, ct, key);
      context->Permute(tmp, tmp, gs_idx);
      context->Add(*final_accum, *final_accum, tmp);
    }
    first = false;
  }

  std::vector<const Dv *> ct_bx;
  std::vector<std::vector<DvConstView<word>>> ct_bx_view;
  std::vector<int> rot_indices;
  for (const auto &[gs_idx, ct] : accum) {
    if (gs_idx == 0) continue;
    ct_bx.push_back(&ct.bx_);
    ct_bx_view.push_back({ct.BxConstView()});
    rot_indices.push_back(gs_idx);
  }
  Dv tmp_add(ref_ct_np.GetNumTotal() * degree);
  DvView<word> tmp_add_view = tmp_add.View(ref_ct_np.num_aux_ * degree);
  std::vector<DvView<word>> tmp_add_view_vector = {tmp_add_view};
  std::vector<DvView<word>> accum_view_vector = {(*final_accum).BxView()};

  // inplace
  ct_bx_view.push_back({(*final_accum).BxConstView()});
  context->elem_handler_.PermuteAccum(accum_view_vector, ref_ct_np, rot_indices,
                                      ct_bx_view);

  EvaluateFinalModDown(context, res, *final_accum, input_num_slots,
                       input_scale);
}

template <typename word>
void HoistHandler<word>::AddRequiredRotations(EvkRequest &req,
                                              bool min_ks) const {
  if (min_ks) {
    auto [bs_stride, gs_stride] = CheckStrideMinKS();
    req.AddRequest(bs_stride, pt_level_);
    req.AddRequest(gs_stride, pt_level_);
    return;
  }

  // else
  for (const auto &bs_idx : bs_indices_) {
    if (bs_idx == 0) continue;
    req.AddRequest(bs_idx, pt_level_);
  }

  if (bs_indices_.size() == 1 && *bs_indices_.begin() == 0) {
    for (const auto &gs_idx : gs_indices_) {
      if (gs_idx == 0) continue;
      req.AddRequest(gs_idx, pt_level_);
    }
    return;
  }

  for (const auto &gs_idx : gs_indices_) {
    if (gs_idx == 0) continue;
    req.AddRequest(gs_idx, pt_level_);
  }
}

template class HoistHandler<uint32_t>;
template class HoistHandler<uint64_t>;

}  // namespace cheddar