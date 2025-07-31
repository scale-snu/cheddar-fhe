#pragma once

#include <vector>

#include "Export.h"
#include "core/DeviceVector.h"
#include "core/Parameter.h"

// ElementWise functions all have the same structure, which we

namespace cheddar {

/**
 * @brief This class is just a collection of element-wise functions
 *
 * @tparam word either uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT ElementWiseHandler {
 private:
  const Parameter<word> &param_;

  static constexpr int kernel_block_dim_ = 256;
  static constexpr int max_num_poly_ = 3;
  static constexpr int max_num_accum_ = 8;
  static inline bool cm_populated_ = false;

  uint32_t PermuteAmountToGaloisFactor(int permute_amount) const;
  void AssertNPMatch(std::vector<DvView<word>> &dst, const NPInfo &np) const;

  template <bool const_accum>
  void CPAccumWorker(std::vector<DvView<word>> &dst, const NPInfo &np,
                     const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
                     const std::vector<DvConstView<word>> &common_srcs) const;

  void PermuteAccumWorker(
      std::vector<DvView<word>> &dst, const NPInfo &np,
      const std::vector<int> &permute_amounts,
      const std::vector<std::vector<DvConstView<word>>> &srcs) const;

 public:
  ElementWiseHandler(const Parameter<word> &param);

  // disable copying (or moving also)
  ElementWiseHandler(const ElementWiseHandler &) = delete;
  ElementWiseHandler &operator=(const ElementWiseHandler &) = delete;

  // for forwarding purposes
  ElementWiseHandler(ElementWiseHandler &&) = default;

  // ----- Basic functions ----- //
  // dst = src1 + src2
  void Add(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1,
           const std::vector<DvConstView<word>> &src2) const;
  // dst = src1 - src2
  void Sub(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1,
           const std::vector<DvConstView<word>> &src2) const;
  // dst = -src1
  void Neg(std::vector<DvView<word>> &dst, const NPInfo &np,
           const std::vector<DvConstView<word>> &src1) const;
  // dst = src1 * src2
  void Mult(std::vector<DvView<word>> &dst, const NPInfo &np,
            const std::vector<DvConstView<word>> &src1,
            const std::vector<DvConstView<word>> &src2) const;
  // dst = src1 * src2 (pt)
  void PMult(std::vector<DvView<word>> &dst, const NPInfo &np,
             const std::vector<DvConstView<word>> &src1,
             const DvConstView<word> &src2) const;

  // dst = src1 + const_src;
  void AddConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                const std::vector<DvConstView<word>> &src1,
                const DvConstView<word> &src_const) const;
  // dst = src1 - const_src;
  void SubConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                const std::vector<DvConstView<word>> &src1,
                const DvConstView<word> &src_const) const;
  // dst = const_src - src1;
  void SubOppositeConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                        const std::vector<DvConstView<word>> &src1,
                        const DvConstView<word> &src_const) const;
  // dst = src1 * const_src;
  void MultConst(std::vector<DvView<word>> &dst, const NPInfo &np,
                 const std::vector<DvConstView<word>> &src1,
                 const DvConstView<word> &src_const) const;

  void Tensor(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<DvConstView<word>> &src1,
              const std::vector<DvConstView<word>> &src2) const;

  void Permute(std::vector<DvView<word>> &dst, const NPInfo &np,
               int permute_amount,
               const std::vector<DvConstView<word>> &src1) const;

  // ----- Accumulation functions ----- //

  void PermuteAccum(
      std::vector<DvView<word>> &dst, const NPInfo &np,
      const std::vector<int> &permute_amounts,
      const std::vector<std::vector<DvConstView<word>>> &srcs) const;
  void Accum(std::vector<DvView<word>> &dst, const NPInfo &np,
             const std::vector<std::vector<DvConstView<word>>> &srcs) const;
  void PAccum(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
              const std::vector<DvConstView<word>> &pt_srcs) const;
  void CAccum(std::vector<DvView<word>> &dst, const NPInfo &np,
              const std::vector<std::vector<DvConstView<word>>> &ct_srcs,
              const std::vector<DvConstView<word>> &const_srcs) const;

  // Special functions, only use it when you know what you are doing
  void ModUpToMax(DvView<word> &dst, const DvConstView<word> &src1) const;

  void MultImaginaryUnit(std::vector<DvView<word>> &dst, const NPInfo &np,
                         const std::vector<DvConstView<word>> &src1,
                         const DvConstView<word> &src_i_unit) const;
};

}  // namespace cheddar
