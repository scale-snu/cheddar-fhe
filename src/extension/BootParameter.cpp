#include "extension/BootParameter.h"

#include "common/CommonUtils.h"

namespace cheddar {

BootParameter::BootParameter(int max_level, int num_cts_levels,
                             int num_stc_levels, int log_message_ratio /* = 5*/)
    : max_level_{max_level},
      num_cts_levels_{num_cts_levels},
      num_stc_levels_{num_stc_levels},
      log_message_ratio_{log_message_ratio},
      mod_coefficients_{
          0.12517186708929745802,    0.0, 0.2894364973331168731,      0.0,
          0.36272381596524499154,    0.0, 0.3011054704600794278,      0.0,
          -0.10550875667295944105,   0.0, -0.43588877795190139706,    0.0,
          0.37482647434055190702,    0.0, -0.14821069913569220404,    0.0,
          0.03665437786710548091,    0.0, -0.0063882548960017121343,  0.0,
          0.00083684232451067872756, 0.0, -8.6443599931576702305e-05, 0.0,
          7.0966437900548814324e-06, 0.0, -5.228015817181348194e-07,  0.0,
          2.2714690137973883081e-08, 0.0, -2.3761936068138980797e-09},
      num_double_angle_{3},
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