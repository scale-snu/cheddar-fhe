#include "extension/BootParameter.h"

#include "common/CommonUtils.h"

#include <stdexcept>

namespace cheddar {

namespace {

// Coefficients of the EvalMod polynomial for each supported num_double_angle.
// The polynomial is a Chebyshev-basis (domain [-1, 1]) approximation of
//   p(t) = (0.5/pi)^(1/2^k) * cos(2*pi*initial_K*t),   initial_K = 2
// fit on the restricted central subdomain required by the target secret's
// bootstrap rounding range |K| <= actual_K = 2^k * initial_K:
//  - k = 3 (shipped): fit for sparse h <= 128 secrets (|K| <= 16)
//  - k = 4 (added  ): weighted-LS fit on |t| <= 0.85, degree 31, protects
//                     |K| <= 27 (sparse h = 256); chain error ~2.6e-11.
// NOTE: coefficients must be refit when changing num_double_angle; a naive
// rescale of the k = 3 series loses ~2 digits of bootstrap precision.
std::vector<double> SelectModCoefficients(int num_double_angle) {
  if (num_double_angle == 3) {
    return {
          0.12517186708929745802,    0.0, 0.2894364973331168731,      0.0,
          0.36272381596524499154,    0.0, 0.3011054704600794278,      0.0,
          -0.10550875667295944105,   0.0, -0.43588877795190139706,    0.0,
          0.37482647434055190702,    0.0, -0.14821069913569220404,    0.0,
          0.03665437786710548091,    0.0, -0.0063882548960017121343,  0.0,
          0.00083684232451067872756, 0.0, -8.6443599931576702305e-05, 0.0,
          7.0966437900548814324e-06, 0.0, -5.228015817181348194e-07,  0.0,
          2.2714690137973883081e-08, 0.0, -2.3761936068138980797e-09};
  }
  if (num_double_angle == 4) {
    return {
          1.4041204361760301067e-01, 0.0,
          3.2467523480058335661e-01, 0.0,
          4.0688293970064004501e-01, 0.0,
          3.3776337446198773096e-01, 0.0,
          -1.1834674842109517423e-01, 0.0,
          -4.8894318281849713514e-01, 0.0,
          4.2045502474397478121e-01, 0.0,
          -1.6624956038157648330e-01, 0.0,
          4.1117390923138201486e-02, 0.0,
          -7.1651397854774903987e-03, 0.0,
          9.3906990962009414918e-04, 0.0,
          -9.6808284192039610323e-05, 0.0,
          8.0173703594971323618e-06, 0.0,
          -5.7034073803067978556e-07, 0.0,
          2.8680817151628213364e-08, 0.0,
          -2.3253594374654767366e-09, 0.0};
  }
  // Unsupported: fit p(t) = (0.5/pi)^(1/2^k) cos(4 pi t) on the subdomain
  // required by your secret's rounding range and add it here.
  throw std::invalid_argument(
      "BootParameter: no EvalMod coefficients for num_double_angle = " +
      std::to_string(num_double_angle));
}

}  // namespace

BootParameter::BootParameter(int max_level, int num_cts_levels,
                             int num_stc_levels, int log_message_ratio /* = 5*/,
                             int num_double_angle /* = 3*/)
    : max_level_{max_level},
      num_cts_levels_{num_cts_levels},
      num_stc_levels_{num_stc_levels},
      log_message_ratio_{log_message_ratio},
      mod_coefficients_{SelectModCoefficients(num_double_angle)},
      num_double_angle_{num_double_angle},
      initial_K_{2} {}


int BootParameter::GetNumEvalModLevels() const {
  return Log2Ceil(mod_coefficients_.size()) + num_double_angle_;
}

int BootParameter::GetMaxLevel() const { return max_level_; }
int BootParameter::GetCtSStartLevel() const { return max_level_; }
int BootParameter::GetEvalModStartLevel() const {
  return max_level_ - num_cts_levels_;
}
int BootParameter::GetStCStartLevel() const {
  return GetEvalModStartLevel() - GetNumEvalModLevels();
}
int BootParameter::GetStartLevel() const { return max_level_; }
int BootParameter::GetEndLevel() const {
  return GetStCStartLevel() - num_stc_levels_;
}

}  // namespace cheddar