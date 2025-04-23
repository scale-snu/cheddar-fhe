#pragma once

#include <memory>
#include <vector>

#include "Export.h"
#include "core/Container.h"
#include "core/ElementWise.h"
#include "core/Encode.h"
#include "core/MemoryPool.h"
#include "core/ModSwitch.h"
#include "core/MultiLevelCiphertext.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

template <typename word>
class API_EXPORT Context {
 protected:
  // short-hand notations
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Const = Constant<word>;

  Context(const Parameter<word> &param);

  void MatchResultWith(Ct &res, const Ct &a) const;
  void MatchResultWith(Ct &res, const Ct &a, const Ct &b) const;
  void AdjustLevelForMultKey(int &level, const int num_q,
                             const int num_aux) const;

  DvConstView<word> GetPProd(NPInfo &np) const;
  const ModSwitchHandler<word> &GetDtSModSwitchHandler() const;
  const ModSwitchHandler<word> &GetStDModSwitchHandler() const;

 public:
  void AssertSameScale(const double &scale1, const double &scale2) const;

  template <typename Container1>
  void AssertSameScale(const Container1 &a, const double &scale) const {
    AssertSameScale(scale, a.GetScale());
  }

  template <typename Container1>
  void AssertSameScale(const double &scale, const Container1 &a) const {
    AssertSameScale(scale, a.GetScale());
  }

  template <typename Container1, typename Container2>
  void AssertSameScale(const Container1 &a, const Container2 &b) const {
    AssertSameScale(a.GetScale(), b.GetScale());
  }

  template <typename Container1, typename Container2, typename... Args>
  void AssertSameScale(const Container1 &a, const Container2 &b,
                       Args... args) const {
    AssertSameScale(a, b);
    AssertSameScale(b, args...);
  }

  /**
   * @brief Create a new Context object. This is the only way to create a new
   * Context and should be used instead of the constructor.
   *
   * @param param CKKS parameter
   * @return std::shared_ptr<Context<word>> a shared pointer to the new Context
   */
  static std::shared_ptr<Context<word>> Create(const Parameter<word> &param);

  // disable copying (or moving also)
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  // For forwarding purposes
  Context(Context &&) = default;

  // Make it polymorphic
  virtual ~Context();

  // The order matters here.
  const Parameter<word> &param_;
  MemoryPool memory_pool_;
  ElementWiseHandler<word> elem_handler_;
  NTTHandler<word> ntt_handler_;
  std::vector<ModSwitchHandler<word>> mod_switch_handlers_;
  Encoder<word> encoder_;

  DeviceVector<word> p_prod_;
  DeviceVector<word> p_prod_dts_;
  std::vector<Const> level_down_consts_;

  /**
   * @brief Copy a ciphertext to another ciphertext. Falls back to nop if the
   * two ciphertexts are the same.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Copy(Ct &res, const Ct &a) const;

  // Basic functions
  // The functions in the Context are meant to be only used for
  // operands using the same prime set. (with same scale)

  /**
   * @brief Add two ciphertexts. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Add(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Add a ciphertext with a plaintext. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   */
  void Add(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Add a ciphertext with a constant. res = a + b
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void Add(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief Subtract two ciphertexts. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Subtract a plaintext from a ciphertext. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input plaintext (right)
   */
  void Sub(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Subtract a constant from a ciphertext. res = a - b
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input constant (right)
   */
  void Sub(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief Subtract a ciphertext from a plaintext. res = a - b
   *
   * @param res result ciphertext
   * @param a input plaintext (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Pt &a, const Ct &b) const;

  /**
   * @brief Subtract a ciphertext from a constant. res = a - b
   *
   * @param res result ciphertext
   * @param a input constant (left)
   * @param b input ciphertext (right)
   */
  void Sub(Ct &res, const Const &a, const Ct &b) const;

  /**
   * @brief Negate a ciphertext. res = -a
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Neg(Ct &res, const Ct &a) const;

  /**
   * @brief Multiply two ciphertexts (only perform tensor). res = a * b.
   * Relinearization or rescaling will not be performed
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   */
  void Mult(Ct &res, const Ct &a, const Ct &b) const;

  /**
   * @brief Multiply a ciphertext with a plaintext. res = a * b. Rescaling will
   * not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   */
  void Mult(Ct &res, const Ct &a, const Pt &b) const;

  /**
   * @brief Multiply a ciphertext with a constant. res = a * b. Rescaling will
   * not be performed
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void Mult(Ct &res, const Ct &a, const Const &b) const;

  // Unsafe functions are used for performance reasons and they only work
  // for specific cases. Do not use them unless you know what you are doing.

  // For Ct x Ct/Pt/Const mult, we should be able to perform operations
  // between operands at different levels. However, it will make scale
  // management very difficult.

  /**
   * @brief Check if two levels are compatible for MultUnsafe.
   *
   * @param level1 level of the first operand
   * @param level2 level of the second operand
   * @return true if the two levels are compatible
   * @return false if the two levels are not compatible
   */
  bool IsMultUnsafeCompatible(int level1, int level2) const;

  /**
   * @brief Multiply two ciphertexts at a designated level. res = a * b.
   * Relinearization or rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Ct &b, int level = -1) const;

  /**
   * @brief Multiply a ciphertext with a plaintext at a designated level. res =
   * a * b. Rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input plaintext
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Pt &b, int level = -1) const;

  /**
   * @brief Multiply a ciphertext with a constant at a designated level. res = a
   * * b. Rescaling will not be performed.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   * @param level (default: -1 --> min(a's level, b's level))
   */
  void MultUnsafe(Ct &res, const Ct &a, const Const &b, int level = -1) const;

  // Should be used after MultKey

  /**
   * @brief Permute the polynomials in the ciphertext according to the rotation
   * distance.
   *
   * @param res result_ciphertext
   * @param a input ciphertext
   * @param rot_dist rotation distance
   */
  void Permute(Ct &res, const Ct &a, int rot_dist) const;

  /**
   * @brief Permute the polynomials in the ciphertext for conjugation.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void PermuteConjugate(Ct &res, const Ct &a) const;

  /**
   * @brief Multiply sqrt(-1) to the ciphertext. This does not require
   * rescaling.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void MultImaginaryUnit(Ct &res, const Ct &a) const;

  /**
   * @brief Perform rescaling on the ciphertext. Level will be reduced by 1.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   */
  void Rescale(Ct &res, const Ct &a) const;

  // Key-related operations
  // Performs ModUp -> key mult -> ModDown (+ Rescale)

  /**
   * @brief Perform relinearization on a ciphertext with three polynomials.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key multiplication key
   */
  void Relinearize(Ct &res, const Ct &a, const Evk &key) const;

  /**
   * @brief Perform relinearization and rescaling on a ciphertext with three
   * polynomials. This function is faster than performing relinearization and
   * rescaling separately and its cost is similar to just a single
   * relinearization.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key multiplication key
   */
  void RelinearizeRescale(Ct &res, const Ct &a, const Evk &key) const;

  /**
   * @brief Multiply a ciphertext with a evaluation key.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param key any evaluation key
   */
  void MultKey(Ct &res, const Ct &a, const Evk &key) const;

  // Short-hand functions

  /**
   * @brief MultKey + Permute
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param rot_key rotation key
   * @param rot_dist rotation distance
   */
  void HRot(Ct &res, const Ct &a, const Evk &rot_key, int rot_dist) const;

  /**
   * @brief MultKey + PermuteConjugate
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param conj_key conjugation key
   */
  void HConj(Ct &res, const Ct &a, const Evk &conj_key) const;

  /**
   * @brief Mult + (Relinearize or RelinearizeRescale)
   *
   * @param res result ciphertext
   * @param a input ciphertext (left)
   * @param b input ciphertext (right)
   * @param mult_key multiplication key
   * @param rescale whether to rescale the result
   */
  void HMult(Ct &res, const Ct &a, const Ct &b, const Evk &mult_key,
             bool rescale = true) const;

  /**
   * @brief res = res + a * b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param b input constant
   */
  void MadUnsafe(Ct &res, const Ct &a, const Const &b) const;

  /**
   * @brief res = (a << rot_dist) + b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext to perform HRot
   * @param b input ciphertext to add
   * @param rot_key rotation key
   * @param rot_dist rotation distance
   */
  void HRotAdd(Ct &res, const Ct &a, const Ct &b, const Evk &rot_key,
               int rot_dist) const;

  /**
   * @brief res = conj(a) + b, but faster.
   *
   * @param res result ciphertext
   * @param a input ciphertext to perform HConj
   * @param b input ciphertext to add
   * @param conj_key conjugation key
   */
  void HConjAdd(Ct &res, const Ct &a, const Ct &b, const Evk &conj_key) const;

  /**
   * @brief Reduce the level of the input ciphertext to a target level.
   * This is not an optimized implementation of LevelDown.
   *
   * @param res result ciphertext
   * @param a input ciphertext
   * @param target_level target level (<= a's level)
   */
  void LevelDown(Ct &res, const Ct &a, int target_level) const;

  /**
   * @brief Add lower-level versions in the MultiLevelCiphertext.
   *
   * @param ml_ct the MultiLevelCiphertext
   * @param min_level the minimum level to support
   */
  void AddLowerLevelsUntil(MultiLevelCiphertext<word> &ml_ct,
                           int min_level) const;

  // Special-purpose functions for bootstrapping/hoisting
  void MultKeyNoModDown(Ct &accum, const std::vector<Dv> &a_modup,
                        const Ct &a_orig, const Evk &key) const;
  void MultKeyNoModDown(Ct &accum, const Ct &a, const Evk &key) const;
};

template <typename word>
using ContextPtr = std::shared_ptr<Context<word>>;

template <typename word>
using ConstContextPtr = std::shared_ptr<const Context<word>>;

}  // namespace cheddar
