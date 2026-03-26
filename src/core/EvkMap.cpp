#include "core/EvkMap.h"

#include "common/Assert.h"

namespace cheddar {

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetEvk(int key_idx) const {
  auto it = this->find(key_idx);
  AssertTrue(it != this->end(),
             "GetEvk: Key not found for index " + std::to_string(key_idx));
  return it->second;
}

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetRotationKey(int rot_idx) const {
  AssertTrue(rot_idx > 0, "GetRotationKey: Invalid rotation index");
  return GetEvk(rot_idx);
}

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetMultiplicationKey() const {
  return GetEvk(kMultiplicationKeyIndex);
}

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetConjugationKey() const {
  return GetEvk(kConjugationKeyIndex);
}

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetDenseToSparseKey() const {
  return GetEvk(kDenseToSparseKeyIndex);
}

template <typename word>
const EvaluationKey<word> &EvkMap<word>::GetSparseToDenseKey() const {
  return GetEvk(kSparseToDenseKeyIndex);
}

template class EvkMap<uint32_t>;
template class EvkMap<uint64_t>;

}  // namespace cheddar
