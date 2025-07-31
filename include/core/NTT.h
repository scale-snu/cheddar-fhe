#pragma once

#include "Export.h"
#include "core/DeviceVector.h"
#include "core/Parameter.h"

namespace cheddar {

namespace kernel {

// dst = f(dst, src, const_src, prime, montgormey)
template <typename word>
using elem_func_t = void (*)(make_signed_t<word> &, const make_signed_t<word>,
                             const word, const word, const make_signed_t<word>);

}  // namespace kernel

enum class API_EXPORT NTTType { NTT, INTT };
enum class API_EXPORT Phase { Phase1, Phase2 };

template <typename word>
class API_EXPORT NTTHandler {
 private:
  using Dv = DeviceVector<word>;
  using Hv = HostVector<word>;

  static inline bool cm_populated_ = false;

  const Parameter<word> &param_;

  Dv twiddle_factors_;
  Dv twiddle_factors_msb_;
  Dv inv_twiddle_factors_;
  Dv inv_twiddle_factors_msb_;
  Dv inv_degree_;
  Dv inv_degree_mont_;
  Dv montgomery_converter_;

  int GetLsbSize() const;
  int GetMsbSize() const;
  int GetLogWarpBatching() const;
  int GetStageMerging(NTTType type, Phase phase) const;
  int GetBlockDim(NTTType type, Phase phase) const;

 public:
  // TODO: allow for different log_degree
  static constexpr int min_log_degree_ = 16;
  static constexpr int max_log_degree_ = 16;

  explicit NTTHandler(const Parameter<word> &param);

  // disable copying (or moving also)
  NTTHandler(const NTTHandler &) = delete;
  NTTHandler &operator=(const NTTHandler &) = delete;

  // dst = NTT(src), montgomery_conversion is false by default
  void NTT(DvView<word> &dst, const NPInfo &np, const DvConstView<word> &src,
           bool montgomery_conversion = false) const;

  // dst = INTT(src), montgomery_conversion is true by default
  void INTT(DvView<word> &dst, const NPInfo &np, const DvConstView<word> &src,
            bool montgomery_conversion = true) const;

  // dst = INTT(src) * src_const
  void INTTAndMultConst(DvView<word> &dst, const NPInfo &np,
                        const DvConstView<word> &src,
                        const DvConstView<word> &src_const,
                        bool normalize = false) const;

  // special variants for ModUp and ModDown/Rescale/ModDownAndRescale

  void NTTForModUp(DvView<word> &dst, const NPInfo &np, int skip_start,
                   int skip_end, const DvConstView<word> &src) const;
  void NTTForModDown(DvView<word> &dst, const NPInfo &np_src1,
                     const NPInfo &np_src2, const DvConstView<word> &src1,
                     const DvConstView<word> &src2,
                     const DvConstView<word> &inv_p_prod,
                     const DvConstView<word> &src2_padding =
                         DvConstView<word>(nullptr, 0)) const;
  void INTTForModDown(DvView<word> &dst, const NPInfo &np_src,
                      const NPInfo &np_non_intt, const DvConstView<word> &src,
                      const DvConstView<word> &src_const) const;

  DvConstView<word> ImaginaryUnitConstView(const NPInfo &np) const;

 private:
  void PopulateTwiddleFactors();
};

}  // namespace cheddar