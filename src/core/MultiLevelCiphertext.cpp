#include "core/MultiLevelCiphertext.h"

#include "common/Assert.h"

namespace cheddar {

template <typename word>
void MultiLevelCiphertext<word>::StaticInit(const Parameter<word> &param,
                                            const Encoder<word> &encoder) {
  param_ = &param;
  int max_level = param_->max_level_;
  level_down_consts_.clear();
  level_down_consts_.resize(max_level + 1);
  for (int i = 1; i < max_level; i++) {
    double scale = param_->GetRescalePrimeProd(i);
    encoder.EncodeConstant(level_down_consts_.at(i), i, scale, 1.0);
  }
}

template <typename word>
void MultiLevelCiphertext<word>::StaticDestroy() {
  param_ = nullptr;
  level_down_consts_.clear();
}

template <typename word>
MultiLevelCiphertext<word>::MultiLevelCiphertext(Ct &&ct) {
  NPInfo np = ct.GetNP();
  AssertTrue(!ct.HasRx(), "MultiLevelCiphertext: Rx is not allowed.");
  AssertTrue(np.num_aux_ == 0,
             "MultiLevelCiphertext: Aux primes are not allowed.");
  int level = param_->NPToLevel(np);
  level_map_.try_emplace(level, std::move(ct));
}

template <typename word>
void MultiLevelCiphertext<word>::AllocateLevel(int level) {
  NPInfo np = param_->LevelToNP(level, 0);
  AssertTrue(!Exists(level), "AddCiphertextAtLevel: Level " +
                                 std::to_string(level) + " already exists");
  level_map_.try_emplace(level, np);
}

template <typename word>
const Constant<word> &MultiLevelCiphertext<word>::GetLevelDownConst(int level) {
  return level_down_consts_.at(level);
}

template <typename word>
int MultiLevelCiphertext<word>::GetMaxLevel() const {
  AssertTrue(!level_map_.empty(), "MultiLevelCiphertext: level_map_ is empty.");
  return level_map_.rbegin()->first;
}

template <typename word>
int MultiLevelCiphertext<word>::GetMinLevel() const {
  AssertTrue(!level_map_.empty(), "MultiLevelCiphertext: level_map_ is empty.");
  return level_map_.begin()->first;
}

template <typename word>
Ciphertext<word> &MultiLevelCiphertext<word>::AtLevel(int level) {
  AssertTrue(Exists(level), "Level does not exist");
  return level_map_.at(level);
}

template <typename word>
const Ciphertext<word> &MultiLevelCiphertext<word>::AtLevel(int level) const {
  AssertTrue(Exists(level), "Level does not exist");
  return level_map_.at(level);
}

template <typename word>
bool MultiLevelCiphertext<word>::Exists(int level) const {
  return level_map_.find(level) != level_map_.end();
}

template <typename word>
void MultiLevelCiphertext<word>::Clear() {
  level_map_.clear();
}

template class MultiLevelCiphertext<uint32_t>;
template class MultiLevelCiphertext<uint64_t>;

}  // namespace cheddar
