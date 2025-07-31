#pragma once

#include <cmath>
#include <vector>

#include "Export.h"
#include "core/Context.h"
#include "extension/BootParameter.h"
#include "extension/EvalPoly.h"

namespace cheddar {

template <typename word>
class API_EXPORT BootContext;

/**
 * @brief EvalMod class for evaluating the mod function during bootstrapping.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT EvalMod {
 private:
  friend class BootContext<word>;

  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  std::vector<EvalPoly<word>> mod_functions_;
  std::vector<AXYPBZ<word>> double_angle_;

  Constant<word> initial_const_;

  double start_scale_;
  double end_scale_;

 public:
  /**
   * @brief Construct a new EvalMod object
   *
   * @param context CKKS context (should actually point to a BootContext)
   * @param boot_param bootstrapping parameters
   */
  EvalMod(ConstContextPtr<word> context, const BootParameter &boot_param);

  EvalMod(const EvalMod &) = delete;
  EvalMod &operator=(const EvalMod &) = delete;
  EvalMod(EvalMod &&) = default;

  /**
   * @brief Evaluate the mod function on the input ciphertext.
   *
   * @param context CKKS context
   * @param res result ciphertext
   * @param input input ciphertext
   * @param mult_key Multiplication key
   */
  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const Evk &mult_key);

  /**
   * @brief Get the polynomial degree of the mod function.
   *
   * @param poly_index index of the polynomial (default: 0)
   * @return int the degree of the polynomial
   */
  int GetEvalModPolyDegree(int poly_index = 0) const;

  /**
   * @brief Get the number of double angle function evaluations.
   *
   * @return int the number of double angle function evaluations
   */
  int GetNumDoubleAngle() const;
};

}  // namespace cheddar
