#include "extension/EvalMod.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
EvalMod<word>::EvalMod(ConstContextPtr<word> context,
                       const BootParameter &boot_param) {
  // Do not check here, checking handled by the BootContext
  int start_level = boot_param.GetEvalModStartLevel();
  int num_double_angle = boot_param.num_double_angle_;
  int double_angle_ratio = 1 << num_double_angle;
  int first_log_scale =
      std::log2(context->param_.GetRescalePrimeProd(start_level)) + 0.5;
  AssertTrue(first_log_scale <= 62, "Invalid eval_mod_scale");
  start_scale_ = (UINT64_C(1) << first_log_scale);

  // See the development notes for details
  int actual_K = (1 << boot_param.num_double_angle_) * boot_param.initial_K_;
  context->encoder_.EncodeConstant(initial_const_, start_level, start_scale_,
                                   -0.25 / actual_K);

  const auto &mod_coefficients = boot_param.mod_coefficients_;
  int mod_levels = Log2Ceil(mod_coefficients.size());

  double target_scale = start_scale_;
  for (int i = 0; i < mod_levels; i++) {
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(start_level - i);
  }

  mod_functions_.emplace_back(boot_param.mod_coefficients_, start_level,
                              start_scale_, target_scale, true);
  mod_functions_[0].Compile(context);

  // TODO(jongmin.kim): add support for other evalmod functions.

  double sqrt2pi = std::pow(0.5 / M_PI, 1.0 / double_angle_ratio);
  for (int i = 0; i < num_double_angle; i++) {
    // 2 x^2 - 1
    sqrt2pi *= sqrt2pi;
    int double_angle_level = start_level - mod_levels - i;
    double_angle_.emplace_back(context, 2, -sqrt2pi, double_angle_level,
                               target_scale, double_angle_level, target_scale);
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(double_angle_level);
  }
  end_scale_ = target_scale;
}

template <typename word>
void EvalMod<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                             const Ct &input, const Evk &mult_key) {
  context->Add(res, input, initial_const_);
  mod_functions_[0].Evaluate(context, res, res, mult_key);
  for (const auto &da : double_angle_) {
    da.Evaluate(context, res, res, res, mult_key);
  }
}

template <typename word>
int EvalMod<word>::GetEvalModPolyDegree(int poly_index /*= 0*/) const {
  return mod_functions_.at(poly_index).GetPolyDegree();
}

template <typename word>
int EvalMod<word>::GetNumDoubleAngle() const {
  return double_angle_.size();
}

template class EvalMod<uint32_t>;
template class EvalMod<uint64_t>;

}  // namespace cheddar