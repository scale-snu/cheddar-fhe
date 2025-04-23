#pragma once

#include <complex>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "Export.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"

namespace cheddar {

using Message = std::vector<std::complex<double>>;
using PlainHoistMap = std::map<int, std::map<int, Message>>;

template <typename word>
struct API_EXPORT GSPlainAccumPtrs {
  const word **bx_;
  const word **ax_;
  const word **mx_;
  word **dst_bx_;
  word **dst_ax_;
};

template <typename word>
class API_EXPORT HoistHandler {
 private:
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Complex = std::complex<double>;

  int pt_level_;
  double pt_scale_;

  std::set<int> bs_indices_;
  std::vector<int> gs_indices_;

  static constexpr int kernel_block_dim_ = 256;
  static inline bool cm_populated_ = false;

  std::map<int, std::map<int, Pt>> hoist_pt_map_;

  // initialization-related methods
  void ExtractBSIndices(const PlainHoistMap &hoist_map);
  void CompilePlaintexts(ConstContextPtr<word> context,
                         const PlainHoistMap &hoist_map);
  std::pair<int, int> CheckStrideMinKS() const;

  // evaluation-related methods
  void EvaluateMultipleAccum(ConstContextPtr<word> context,
                             std::map<int, Ct> &results,
                             const std::vector<int> &gs_indices,
                             const std::map<int, Ct> &bs) const;
  void EvaluateSingleAccum(ConstContextPtr<word> context, Ct &res,
                           const std::map<int, Ct> &bs,
                           const std::map<int, Pt> &pt_map,
                           bool inplace = false) const;
  void EvaluateFinalModDown(ConstContextPtr<word> context, Ct &res,
                            Ct &final_accum, int input_num_slots,
                            double input_scale) const;
  void EvaluateFinalModDownNoRescale(ConstContextPtr<word> context, Ct &res,
                                     Ct &final_accum, int input_num_slots,
                                     double input_scale) const;
  void EvaluateMinKS(ConstContextPtr<word> context, Ct &res, const Ct &input,
                     const EvkMap<word> &evk_map) const;

  void EvaluateMinKSBabyStep(ConstContextPtr<word> context,
                             std::map<int, Ct> &bs, const Ct &input,
                             const EvkMap<word> &evk_map) const;
  void EvaluateMinKSGiantStep(ConstContextPtr<word> context, Ct &res,
                              const std::map<int, Ct> &bs,
                              const EvkMap<word> &evk_map) const;

  void EvaluateGiantStepOptimized(ConstContextPtr<word> context, Ct &res,
                                  const std::map<int, Ct> &bs,
                                  const EvkMap<word> &evk_map) const;
  void BSMultKeyBatched(ConstContextPtr<word> context, std::vector<Ct *> &res,
                        std::vector<Dv> &a_modup, const Ct &a_orig,
                        const EvkMap<word> &keys, std::vector<int> &rotations,
                        const Dv &input_bx_pseudo_modup) const;

 public:
  HoistHandler(ConstContextPtr<word> context, const PlainHoistMap &hoist_map,
               int pt_level, double pt_scale, bool suppress_bs_swap = false,
               bool swap_bs_gs = false);

  HoistHandler(const HoistHandler &) = delete;
  HoistHandler &operator=(const HoistHandler &) = delete;
  HoistHandler(HoistHandler &&) = default;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateBabyStep(ConstContextPtr<word> context, std::map<int, Ct> &bs,
                        const Ct &input, const EvkMap<word> &evk_map,
                        bool min_ks = false) const;
  void EvaluateGiantStep(ConstContextPtr<word> context, Ct &res,
                         const std::map<int, Ct> &bs,
                         const EvkMap<word> &evk_map,
                         bool min_ks = false) const;
  void EvaluateHoistRotAccum(ConstContextPtr<word> context, Ct &res,
                             const Ct &input,
                             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar
