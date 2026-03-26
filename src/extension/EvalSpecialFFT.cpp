#include "extension/EvalSpecialFFT.h"

#include <cmath>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
EvalSpecialFFT<word>::EvalSpecialFFT(ConstContextPtr<word> context,
                                     const BootParameter &boot_param,
                                     int num_slots, double cts_const,
                                     double stc_const)
    : num_slots_{num_slots},
      boot_param_{boot_param},
      cts_const_{cts_const},
      stc_const_{stc_const},
      full_slot_{num_slots == context->param_.degree_ / 2} {
  AssertTrue(num_slots >= 256,
             "Currently only high number of slots are supported");
  AssertTrue(IsPowOfTwo(num_slots), "Number of slots must be a power of 2");
  AssertTrue(num_slots <= context->param_.degree_ / 2,
             "Number of slots exceeds the maximum possible");
  PopulatePlainMatrices(context);
  PreparePlaintexts(context);
}

template <typename word>
std::pair<int, int> EvalSpecialFFT<word>::BSGSSplit(int num_diag) const {
  AssertTrue(IsPowOfTwo(num_diag) || IsPowOfTwo(num_diag + 1),
             "Invalid number of diagonals for EvalSpecialFFT");
  // this is somewhat heuristic
  int bs, gs;
  if (num_diag <= 4) {
    return {num_diag, 1};
  }

  switch (num_diag) {
    case 7:
    case 8:
    case 15:
      // consider using bs = 5;
    case 16:
      bs = 4;
      break;
    case 31:
    case 32:
    case 63:
      // consider using bs = 9, 11;
    case 64:
      // consider using bs = 11;
      bs = 8;
      break;
    default:  // over 127, don't care actually
      bs = 1 << DivCeil(Log2Ceil(num_diag), 2);
      break;
  }
  gs = DivCeil(num_diag, bs);

  return {bs, gs};
}

template <typename word>
void EvalSpecialFFT<word>::PopulatePlainMatrices(
    ConstContextPtr<word> context) {
  int M = context->param_.degree_ * 2;
  const auto &encoder = context->encoder_;

  int num_stages = Log2Ceil(num_slots_);
  plain_fft_stages_.resize(num_stages);
  plain_ifft_stages_.resize(num_stages);

  for (int i = 0; i < num_stages; i++) {
    int stride = 1 << i;
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M / st8;

    // Multiplication order left (0) --> right (num_stages - 1)
    auto &fft_target = plain_fft_stages_[i];
    auto &ifft_target = plain_ifft_stages_[num_stages - i - 1];

    fft_target = StripedMatrix(num_slots_, num_slots_);
    ifft_target = StripedMatrix(num_slots_, num_slots_);

    fft_target.try_emplace(0, num_slots_, Complex(0));
    fft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      fft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &fft_diag_0 = fft_target[0];
    auto &fft_diag_plus = fft_target[stride];
    auto &fft_diag_minus = fft_target[num_slots_ - stride];

    ifft_target.try_emplace(0, num_slots_, Complex(0));
    ifft_target.try_emplace(stride, num_slots_, Complex(0));
    if (i != num_stages - 1) {
      ifft_target.try_emplace(num_slots_ - stride, num_slots_, Complex(0));
    }
    auto &ifft_diag_0 = ifft_target[0];
    auto &ifft_diag_plus = ifft_target[stride];
    auto &ifft_diag_minus = ifft_target[num_slots_ - stride];

    for (int j = 0; j < stride; j++) {
      int fft_twiddle_index = (context->param_.GetGaloisFactor(j) % st8) * gap;
      int ifft_twiddle_index =
          (st8 - (context->param_.GetGaloisFactor(j) % st8)) * gap;
      Complex fft_twiddle = encoder.GetTwiddleFactor(fft_twiddle_index);
      Complex ifft_twiddle = encoder.GetTwiddleFactor(ifft_twiddle_index);

      // FFT
      // (x, y) = (x + y * twiddle, x - y * twiddle)
      fft_diag_0[j] = 1;
      fft_diag_plus[j] = fft_twiddle;
      fft_diag_minus[j + stride] = 1;
      fft_diag_0[j + stride] = -fft_twiddle;
      /* Matrix form
      fft_target[j][j] = 1;
      fft_target[j][j + stride] = fft_twiddle;
      fft_target[j + stride][j] = 1;
      fft_target[j + stride][j + stride] = -fft_twiddle;
      */

      // IFFT
      // (x, y) = (x + y, (x - y) * twiddle)
      ifft_diag_0[j] = 1;
      ifft_diag_plus[j] = 1;
      ifft_diag_minus[j + stride] = ifft_twiddle;
      ifft_diag_0[j + stride] = -ifft_twiddle;

      /* Matrix form
      ifft_target[j][j] = 1;
      ifft_target[j][j + stride] = 1;
      ifft_target[j + stride][j] = ifft_twiddle;
      ifft_target[j + stride][j + stride] = -ifft_twiddle;
      */
    }

    // For the rest, we can simply copy the values
    int num_double = Log2Ceil(num_slots_ / stride_group_size);

    for (int r = 0; r < num_double; r++) {
      std::copy(fft_diag_0.begin(),
                fft_diag_0.begin() + stride_group_size * (1 << r),
                fft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(fft_diag_plus.begin(),
                fft_diag_plus.begin() + stride_group_size * (1 << r),
                fft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(fft_diag_minus.begin(),
                  fft_diag_minus.begin() + stride_group_size * (1 << r),
                  fft_diag_minus.begin() + stride_group_size * (1 << r));
      }
      std::copy(ifft_diag_0.begin(),
                ifft_diag_0.begin() + stride_group_size * (1 << r),
                ifft_diag_0.begin() + stride_group_size * (1 << r));
      std::copy(ifft_diag_plus.begin(),
                ifft_diag_plus.begin() + stride_group_size * (1 << r),
                ifft_diag_plus.begin() + stride_group_size * (1 << r));
      if (i != num_stages - 1) {
        std::copy(ifft_diag_minus.begin(),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r),
                  ifft_diag_minus.begin() + stride_group_size * (1 << r));
      }
    }
  }
}

template <typename word>
void EvalSpecialFFT<word>::PreparePlaintexts(ConstContextPtr<word> context) {
  int num_cts_phases = boot_param_.num_cts_levels_;
  int num_stc_phases = boot_param_.num_stc_levels_;
  int log_num_slots = Log2Ceil(num_slots_);

  int cts_level = boot_param_.GetCtSStartLevel();
  int stc_level = boot_param_.GetStCStartLevel();
  AssertTrue(num_cts_phases >= 2, "Use at least 2 levels for CtS");
  AssertTrue(num_stc_phases >= 2, "Use at least 2 levels for StC");

  int cts_stages_left = log_num_slots;
  int cts_stages_cumul = 0;
  double cts_const_div = std::pow(cts_const_, 1.0 / num_cts_phases);
  // std::cout << "cts_const_div: " << cts_const_div << std::endl;
  double stc_const_div = std::pow(stc_const_, 1.0 / num_stc_phases);
  // std::cout << "stc_const_div: " << stc_const_div << std::endl;

  // We will use different scaling methodology for CtS and StC
  double cts_scale = 1.0;
  for (int i = 0; i < cts_level; i++) {
    cts_scale *= context->param_.GetRescalePrimeProd(cts_level - i);
  }
  cts_scale = std::pow(cts_scale, 1.0 / num_cts_phases);

  for (int i = 0; i < num_cts_phases; i++) {
    std::cout << "CtS preparation phase " << i << std::endl;
    // CtS: high strides (num_slots / 2) --> low strides (1)
    int num_stages;
    if (i == 0) {
      num_stages = DivCeil(cts_stages_left, num_cts_phases);
    } else {
      num_stages = cts_stages_left / (num_cts_phases - i);
    }
    cts_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_ifft_stages_[cts_stages_cumul];
    for (int j = cts_stages_cumul + 1; j < cts_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_ifft_stages_[j], phase_matrix);
    }

    // Decomposing into Wx and -iWx part for later decomposition of real and
    // imag part for non-full-slot cases
    if (i == num_cts_phases - 1 && !full_slot_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, -1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, cts_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == num_cts_phases - 1) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "CtS phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation;
    int additional_pt_rot = -(1 << cts_stages_left);
    if (i == 0) {
      pre_rotation = (1 << cts_stages_left);
    } else if (i == num_cts_phases - 1) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << cts_stages_left);
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    cts_phases_.emplace_back(context, phase_matrix, cts_level - i,
                             context->param_.GetRescalePrimeProd(cts_level - i),
                             bs, gs, pre_rotation, additional_pt_rot);
    cts_stages_cumul += num_stages;
  }

  // 2. StC initialization
  int stc_stages_left = log_num_slots;
  int stc_stages_cumul = 0;
  for (int i = 0; i < num_stc_phases; i++) {
    std::cout << "StC preparation phase " << i << std::endl;
    // StC: low strides (1) --> high strides (num_slots / 2)
    int num_stages = stc_stages_left / (num_stc_phases - i);
    stc_stages_left -= num_stages;

    StripedMatrix phase_matrix = plain_fft_stages_[stc_stages_cumul];
    for (int j = stc_stages_cumul + 1; j < stc_stages_cumul + num_stages; j++) {
      phase_matrix = StripedMatrix::Mult(plain_fft_stages_[j], phase_matrix);
    }

    if (i == 0 && !full_slot_) {
      StripedMatrix extended(num_slots_ * 2, num_slots_ * 2);
      for (auto &[i, diag] : phase_matrix) {
        int dst_idx = i;
        if (i >= num_slots_ / 2) dst_idx += num_slots_;
        extended.try_emplace(dst_idx, num_slots_ * 2, Complex(0));
        for (int j = 0; j < num_slots_; j++) {
          extended.at(dst_idx)[j] = diag[j];
          extended.at(dst_idx)[j + num_slots_] = diag[j] * Complex(0, 1);
        }
      }
      phase_matrix = extended;
    }
    phase_matrix = StripedMatrix::Mult(phase_matrix, stc_const_div);

    int num_eff_diag = phase_matrix.GetNumDiag();
    if (i == 0) num_eff_diag += 1;
    auto [bs, gs] = BSGSSplit(num_eff_diag);

    // std::cout << "StC phase " << i << ": bs = " << bs << ", gs = " << gs
    //          << std::endl;

    // Min-KS adjustment (can be used also for hoisting)
    int pre_rotation, additional_pt_rot;
    if (i == 0) {
      pre_rotation = -(1 << num_stages);
      additional_pt_rot = (1 << num_stages);
    } else if (i == num_stc_phases - 1) {
      pre_rotation = (1 << stc_stages_cumul);
      additional_pt_rot = 0;
    } else {
      pre_rotation = -((1 << num_stages) - 1) * (1 << stc_stages_cumul);
      additional_pt_rot = (1 << (num_stages + stc_stages_cumul));
    }
    // std::cout << "Pre rotation: " << pre_rotation << std::endl;
    // std::cout << "Additional pt rot: " << additional_pt_rot << std::endl;

    // double stc_scale = context->param_.GetScale(stc_level - i);
    double stc_scale = context->param_.GetRescalePrimeProd(stc_level - i);
    stc_phases_.emplace_back(context, phase_matrix, stc_level - i, stc_scale,
                             bs, gs, pre_rotation, additional_pt_rot);
    stc_stages_cumul += num_stages;
  }
}

template <typename word>
void EvalSpecialFFT<word>::AddRequiredRotations(EvkRequest &req,
                                                bool min_ks) const {
  for (const auto &cts_phase : cts_phases_) {
    cts_phase.AddRequiredRotations(req, min_ks);
  }
  for (const auto &stc_phase : stc_phases_) {
    stc_phase.AddRequiredRotations(req, min_ks);
  }
  if (!full_slot_) {
    req.AddRequest(num_slots_, boot_param_.GetEndLevel());
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateCtS(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  int num_cts_phases = cts_phases_.size();
  cts_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_cts_phases; i++) {
    cts_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
  }
}

template <typename word>
void EvalSpecialFFT<word>::EvaluateStC(ConstContextPtr<word> context, Ct &res,
                                       const Ct &input,
                                       const EvkMap<word> &evk_map,
                                       bool min_ks) const {
  int num_stc_phases = stc_phases_.size();
  stc_phases_.at(0).Evaluate(context, res, input, evk_map, min_ks);
  for (int i = 1; i < num_stc_phases; i++) {
    stc_phases_.at(i).Evaluate(context, res, res, evk_map, min_ks);
  }

  Ct tmp;
  if (!full_slot_) {
    res.SetNumSlots(num_slots_ * 2);
    // res += HRot(res, num_slots_)
    context->HRotAdd(res, res, res, evk_map.GetRotationKey(num_slots_),
                     num_slots_);
  }
  res.SetNumSlots(num_slots_);
}

template class EvalSpecialFFT<uint32_t>;
template class EvalSpecialFFT<uint64_t>;

}  // namespace cheddar
