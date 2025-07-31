#pragma once

#include <unordered_map>

#include "Export.h"
#include "core/Container.h"

namespace cheddar {

/**
 * @brief Class for storing client-prepared evaluation keys.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT EvkMap : public std::unordered_map<int, EvaluationKey<word>> {
 private:
  using Base = std::unordered_map<int, EvaluationKey<word>>;
  using Evk = EvaluationKey<word>;

  const Evk &GetEvk(int key_idx) const;

 public:
  static inline constexpr int kConjugationKeyIndex = 11111111;
  static inline constexpr int kMultiplicationKeyIndex = -22222222;
  static inline constexpr int kDenseToSparseKeyIndex = -33333333;
  static inline constexpr int kSparseToDenseKeyIndex = -44444444;

  using Base::Base;
  EvkMap(const EvkMap &) = delete;
  EvkMap &operator=(const EvkMap &) = delete;
  EvkMap(EvkMap &&) = default;

  const Evk &GetRotationKey(int rot_idx) const;
  const Evk &GetMultiplicationKey() const;
  const Evk &GetConjugationKey() const;
  const Evk &GetDenseToSparseKey() const;
  const Evk &GetSparseToDenseKey() const;
};

}  // namespace cheddar
