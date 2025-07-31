#pragma once

#include <vector>

#include "Export.h"

namespace cheddar {

/**
 * @brief A class for storing bootstrapping parameters.
 *
 */
struct API_EXPORT BootParameter {
  /**
   * @brief Construct a new BootParameter object
   *
   * @param max_level the maximum level
   * @param num_cts_levels the number of levels for CoeffToSlot (CtS)
   * @param num_stc_levels the number of levels for SlotToCoeff (StC)
   * @param log_message_ratio approximately log2(q0 / scale to target at the
   * lowest level). Do not modify if you do not know what you are doing.
   */
  BootParameter(int max_level, int num_cts_levels, int num_stc_levels,
                int log_message_ratio = 5);

  const int max_level_;
  const int num_cts_levels_;
  const int num_stc_levels_;

  const int log_message_ratio_;

  // The following three parameters are inter-related, so changing
  // one of them requires changing the others.
  // TODO (jongmin.kim): Allow changing these parameters
  const std::vector<double> mod_coefficients_;
  const int num_double_angle_;
  const int initial_K_;

  /**
   * @brief Get the level consumption for EvalMod
   *
   * @return int the number of levels consumed by EvalMod
   */
  int GetNumEvalModLevels() const;

  /**
   * @brief Get the maximum level
   *
   * @return int the maximum level
   */
  int GetMaxLevel() const;

  /**
   * @brief Get the starting level for CoeffToSlot (CtS)
   *
   * @return int the starting level for CoeffToSlot (CtS)
   */
  int GetCtSStartLevel() const;

  /**
   * @brief Get the starting level for EvalMod
   *
   * @return int the starting level for EvalMod
   */
  int GetEvalModStartLevel() const;

  /**
   * @brief Get the starting level for SlotToCoeff (StC)
   *
   * @return int the starting level for SlotToCoeff (StC)
   */
  int GetStCStartLevel() const;

  /**
   * @brief Get the starting level for bootstrapping. Should be the same as
   * GetMaxLevel() and GetCtSStartLevel().
   *
   * @return int the starting level for bootstrapping
   */
  int GetStartLevel() const;

  /**
   * @brief Get the ending level for bootstrapping. Should be the same as
   * default_encryption_level_.
   *
   * @return int the ending level for bootstrapping
   */
  int GetEndLevel() const;
};

}  // namespace cheddar
