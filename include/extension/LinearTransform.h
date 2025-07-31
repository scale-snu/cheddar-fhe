#pragma once

#include <iostream>
#include <set>
#include <unordered_map>

#include "core/Context.h"
#include "core/EvkMap.h"
#include "extension/Hoist.h"
#include "extension/StripedMatrix.h"
#include "Export.h"

namespace cheddar {

/**
 * @brief A factory class for EvalHoist.
 *
 * @tparam word
 */
template <typename word>
class API_EXPORT LinearTransform {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Dv = DeviceVector<word>;

  int pt_level_;
  double pt_scale_;

  int bs_;
  int gs_;
  int pre_rotation_;
  int additional_pt_rot_;

  // shoule be the last members
  int stride_;
  HoistHandler<word> hoist_;

  int DetermineStride(const StripedMatrix &matrix);
  PlainHoistMap ConstructPlainHoistMap(const StripedMatrix &matrix);

 public:
  LinearTransform(ConstContextPtr<word> context, const StripedMatrix &matrix,
                  int pt_level, double pt_scale, int bs, int gs = 1,
                  int pre_rotation = 0, int additional_pt_rot = 0);

  bool IsUsingBSGS() const;
  int GetBS() const;
  int GetGS() const;
  int GetPreRotationAmount() const;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const EvkMap<word> &evk_map, bool min_ks = false) const;
};

}  // namespace cheddar