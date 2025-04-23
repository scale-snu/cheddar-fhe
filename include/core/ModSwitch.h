#pragma once

#include <vector>

#include "Export.h"
#include "core/DeviceVector.h"
#include "core/ElementWise.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

template <typename word>
class API_EXPORT ModSwitchHandler {
 private:
  using Dv = DeviceVector<word>;

  // The order mattters here
  const int level_;
  const int num_aux_;
  const int beta_;

  const Parameter<word> &param_;
  const ElementWiseHandler<word> &elem_handler_;
  const NTTHandler<word> &ntt_handler_;

 public:
  ModSwitchHandler(const Parameter<word> &param, int level,
                   const ElementWiseHandler<word> &elem_handler,
                   const NTTHandler<word> &ntt_handler);

  // diable copying (or moving also)
  ModSwitchHandler(const ModSwitchHandler &) = delete;
  ModSwitchHandler &operator=(const ModSwitchHandler &) = delete;

  // for forwarding purposes
  ModSwitchHandler(ModSwitchHandler &&) = default;

  void PseudoModUp(DvView<word> &dst, const DvConstView<word> &src,
                   const DvConstView<word> &p_prod) const;
  void ModUp(std::vector<DvView<word>> &dst,
             const DvConstView<word> &src) const;
  void ModDown(DvView<word> &dst, const DvConstView<word> &src) const;
  void Rescale(DvView<word> &dst, const DvConstView<word> &src) const;
  void ModDownAndRescale(DvView<word> &dst, const DvConstView<word> &src) const;

 private:
  // ModUp constants
  Dv mod_up1_;
  std::vector<DeviceVector<make_signed_t<word>>> mod_up2_;

  // ModDown constants
  Dv mod_down1_;
  DeviceVector<make_signed_t<word>> mod_down2_;
  Dv inv_prime_prod_;

  // Rescale constants
  int rescale_pad_start_;
  int rescale_pad_end_;
  int rescale_restore_start_;
  int rescale_restore_end_;

  Dv rescale1_;
  DeviceVector<make_signed_t<word>> rescale2_;
  Dv rescale_inv_prime_prod_;
  Dv rescale_padding_;

  // ModDownAndRescale constants
  Dv mod_down_rescale1_;
  DeviceVector<make_signed_t<word>> mod_down_rescale2_;
  Dv mod_down_rescale_inv_prime_prod_;
  Dv mod_down_rescale_padding_;

  Dv entire_padding_;

  // heuristic CUDA kernel block number;
  static constexpr int block_dim_ = 256;
  static inline bool cm_populated_ = false;

  void PopulateModSwitchConstants(Dv &const1,
                                  DeviceVector<make_signed_t<word>> &const2,
                                  const std::vector<word> &src_primes,
                                  const std::vector<word> &dst_primes,
                                  int restore_start, int restore_end);
  void PopulateModDownEpilogueConstants(Dv &inv_p_prod, Dv &padding,
                                        const std::vector<word> &src_primes,
                                        const std::vector<word> &dst_primes,
                                        int restore_start, int restore_end);

  enum class ModDownType { ModDown, Rescale, ModDownAndRescale };
  void ModDownWorker(DvView<word> &dst, const DvConstView<word> &src,
                     ModDownType type) const;
};

}  // namespace cheddar
