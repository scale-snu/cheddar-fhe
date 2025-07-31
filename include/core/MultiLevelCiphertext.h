#pragma once

#include <map>

#include "Export.h"
#include "core/Container.h"
#include "core/Encode.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief We sometimes need to keep multiple ciphertexts at different levels but
 * with the same scale. This class is used to keep track of those ciphertexts.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT MultiLevelCiphertext {
 private:
  using Ct = Ciphertext<word>;

  std::map<int, Ct> level_map_;

  static inline const Parameter<word> *param_ = nullptr;

  // different from the one in Context
  static inline std::vector<Constant<word>> level_down_consts_{};

 public:
  static void StaticInit(const Parameter<word> &param,
                         const Encoder<word> &encoder);
  static void StaticDestroy();

  MultiLevelCiphertext(Ct &&ct);

  // movable, but not copyable
  MultiLevelCiphertext(MultiLevelCiphertext &&) = default;
  MultiLevelCiphertext &operator=(MultiLevelCiphertext &&) = default;

  int GetMaxLevel() const;
  int GetMinLevel() const;

  Ct &AtLevel(int level);
  const Ct &AtLevel(int level) const;
  bool Exists(int level) const;
  void Clear();

  // For the use in Context::AddLowerLevelsUntil

  void AllocateLevel(int level);
  static const Constant<word> &GetLevelDownConst(int level);
};

}  // namespace cheddar
