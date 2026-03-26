#pragma once

#include <vector>

#include "core/DeviceVector.h"

namespace cheddar {

// Max length is likely to be 3
template <typename word, int num_poly>
struct OutputPtrList {
  word *ptrs_[num_poly];

  OutputPtrList() {
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = nullptr;
    }
  };

  OutputPtrList(std::vector<DvView<word>> &vec) {
    AssertTrue(static_cast<int>(vec.size()) == num_poly,
               "OutputPtrList size mismatch");
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = vec.at(i).data();
    }
  }

  OutputPtrList(DvView<word> &v0) {
    if constexpr (num_poly == 1) {
      ptrs_[0] = v0.data();
    } else {
      Fail("OutputPtrList size mismatch");
    }
  }
};

template <typename word, int num_poly>
struct InputPtrList {
  const word *ptrs_[num_poly];
  int extra_ = 0;

  InputPtrList() {
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = nullptr;
    }
  };

  InputPtrList(const std::vector<DvConstView<word>> &vec) {
    AssertTrue(static_cast<int>(vec.size()) == num_poly,
               "InputPtrList size mismatch");
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = vec[i].data();
    }
  }

  InputPtrList(const DvConstView<word> &v0) {
    if constexpr (num_poly == 1) {
      ptrs_[0] = v0.data();
    } else {
      Fail("OutputPtrList size mismatch");
    }
  }
};

template <typename word, int num_poly>
struct PermuteInputPtrList {
  const word *ptrs_[num_poly];
  int extra_ = 0;
  uint32_t galois_factor_ = 1;

  PermuteInputPtrList() {
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = nullptr;
    }
  };

  PermuteInputPtrList(const std::vector<DvConstView<word>> &vec) {
    AssertTrue(static_cast<int>(vec.size()) == num_poly,
               "PermuteInputPtrList size mismatch");
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = vec[i].data();
    }
  }

  PermuteInputPtrList(const DvConstView<word> &v0) {
    if constexpr (num_poly == 1) {
      ptrs_[0] = v0.data();
    } else {
      Fail("OutputPtrList size mismatch");
    }
  }
};

template <typename word, int num_poly>
struct CPAccumInputPtrList {
  const word *ptrs_[num_poly];
  int extra_ = 0;
  const word *common_ptr_;
  int common_extra_ = 0;

  CPAccumInputPtrList() {
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = nullptr;
    }
    common_ptr_ = nullptr;
  };

  CPAccumInputPtrList(const std::vector<DvConstView<word>> &vec,
                      const DvConstView<word> &common_ptr) {
    AssertTrue(static_cast<int>(vec.size()) == num_poly,
               "PAccumInputPtrList size mismatch");
    for (int i = 0; i < num_poly; i++) {
      ptrs_[i] = vec[i].data();
    }
    common_ptr_ = common_ptr.data();
  }

  CPAccumInputPtrList(const DvConstView<word> &v0,
                      const DvConstView<word> &v1) {
    if constexpr (num_poly == 1) {
      ptrs_[0] = v0.data();
      common_ptr_ = v1.data();
    } else {
      Fail("OutputPtrList size mismatch");
    }
  }
};

}  // namespace cheddar
