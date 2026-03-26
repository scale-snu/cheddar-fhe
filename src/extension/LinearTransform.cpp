#include "extension/LinearTransform.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
int LinearTransform<word>::DetermineStride(const StripedMatrix &matrix) {
  int gcd_rot = 0;
  int max_rot = 0;
  int num_pt = 0;

  int width = matrix.GetWidth();

  for (const auto &[i, _] : matrix) {
    num_pt += 1;
    int rot = (i - pre_rotation_) % width;
    if (rot < 0) {
      rot += width;
    }
    if (gcd_rot == 0 && rot == 0) {
      gcd_rot = rot;
    } else {
      gcd_rot = GCD(gcd_rot, rot);
    }
    max_rot = Max(max_rot, rot);
  }
  AssertTrue(num_pt > 1, "LinearTransform requires at least 2 plaintexts");
  AssertTrue(gcd_rot > 0, "Something went wrong during LinearTransform setup");
  int max_pt_dist = (bs_ * gs_ - 1) * gcd_rot;

  AssertTrue(max_rot <= max_pt_dist,
             "Incompatible matrix and LinearTransform parameters: bs (" +
                 std::to_string(bs_) + "), gs (" + std::to_string(gs_) +
                 "), pre_rotation (" + std::to_string(pre_rotation_) + ")");
  return gcd_rot;
}

template <typename word>
PlainHoistMap LinearTransform<word>::ConstructPlainHoistMap(
    const StripedMatrix &matrix) {
  int height = matrix.GetHeight();
  int width = matrix.GetWidth();

  AssertTrue(IsPowOfTwo(height),
             "LinearTransform requires power-of-two height");
  AssertTrue(IsPowOfTwo(width), "LinearTransform requires power-of-two width");
  AssertTrue(height == width, "LinearTransform requires square matrix");

  int gs_stride = stride_ * bs_;

  PlainHoistMap hoist_map;
  for (const auto &[i, diag] : matrix) {
    int rot = (i - pre_rotation_) % width;
    if (rot < 0) {
      rot += width;
    }
    int bs_rot = rot % gs_stride;
    int gs_rot = rot - bs_rot;
    if (hoist_map.find(gs_rot) == hoist_map.end()) {
      hoist_map.try_emplace(gs_rot, std::map<int, Message>());
    }
    hoist_map[gs_rot].try_emplace(bs_rot, height, 0);
    int offset = (gs_rot - additional_pt_rot_) % height;
    if (offset < 0) {
      offset += height;
    }
    for (int j = 0; j < height; j++) {
      // reverse gs rot
      hoist_map[gs_rot][bs_rot][(j + offset) % height] = diag[j];
    }
  }
  return hoist_map;
}

template <typename word>
LinearTransform<word>::LinearTransform(ConstContextPtr<word> context,
                                       const StripedMatrix &matrix,
                                       int pt_level, double pt_scale, int bs,
                                       int gs /*= 1*/, int pre_rotation /*= 0*/,
                                       int additional_pt_rot /*= 0*/)
    : pt_level_{pt_level},
      pt_scale_{pt_scale},
      bs_{bs},
      gs_{gs},
      pre_rotation_{pre_rotation},
      additional_pt_rot_{additional_pt_rot},
      stride_{DetermineStride(matrix)},
      hoist_{context, ConstructPlainHoistMap(matrix), pt_level, pt_scale} {}

template <typename word>
bool LinearTransform<word>::IsUsingBSGS() const {
  return bs_ > 1 && gs_ > 1;
}

template <typename word>
int LinearTransform<word>::GetBS() const {
  return bs_;
}

template <typename word>
int LinearTransform<word>::GetGS() const {
  return gs_;
}

template <typename word>
int LinearTransform<word>::GetPreRotationAmount() const {
  return pre_rotation_;
}

template <typename word>
void LinearTransform<word>::AddRequiredRotations(
    EvkRequest &req, bool min_ks /*= false*/) const {
  return hoist_.AddRequiredRotations(req, min_ks);
}

template <typename word>
void LinearTransform<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                                     const Ct &input,
                                     const EvkMap<word> &evk_map,
                                     bool min_ks /*= false*/) const {
  hoist_.Evaluate(context, res, input, evk_map, min_ks);
}

template class LinearTransform<uint32_t>;
template class LinearTransform<uint64_t>;

}  // namespace cheddar