#pragma once

#include "Export.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootParameter.h"
#include "extension/LinearTransform.h"

namespace cheddar {

/**
 * @brief A class for the homomorphic evaluation of special FFT
 * (SlotToCoeff/StC) and IFFT (CoeffToSlot/CtS)
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT EvalSpecialFFT {
 private:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;
  using Complex = std::complex<double>;

  const int num_slots_;
  BootParameter boot_param_;

  const double cts_const_;
  const double stc_const_;
  const bool full_slot_;

  std::vector<LinearTransform<word>> cts_phases_;
  std::vector<LinearTransform<word>> stc_phases_;

  std::vector<StripedMatrix> plain_fft_stages_;
  std::vector<StripedMatrix> plain_ifft_stages_;

  std::pair<int, int> BSGSSplit(int num_diag) const;
  void PopulatePlainMatrices(ConstContextPtr<word> context);
  void PreparePlaintexts(ConstContextPtr<word> context);

 public:
  EvalSpecialFFT(ConstContextPtr<word> context, const BootParameter &boot_param,
                 int num_slots, double cts_const, double stc_const);

  EvalSpecialFFT(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT &operator=(const EvalSpecialFFT &) = delete;
  EvalSpecialFFT(EvalSpecialFFT &&) = default;

  void AddRequiredRotations(EvkRequest &req, bool min_ks = false) const;

  void EvaluateCtS(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateStC(ConstContextPtr<word> context, Ct &res, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
};

}  // namespace cheddar