#pragma once

#include <thrust/host_vector.h>

#include <iostream>
#include <rmm/device_uvector.hpp>

#include "Export.h"

namespace cheddar {

/**
 * @brief A thin wrapper around thrust::host_vector.
 *
 */
template <typename word>
class API_EXPORT HostVector : public thrust::host_vector<word> {
  using Base = thrust::host_vector<word>;

 public:
  using Base::Base;
};

/**
 * @brief Provides a view of a device vector.
 *
 * @tparam word uint32_t/uint64_t/int32_t/int64_t
 */
template <typename word>
class API_EXPORT DvView {
 public:
  DvView(word *data, int size, int aux_size = 0);

  word *data();
  const word *data() const;

  int TotalSize() const;
  int AuxSize() const;
  int QSize() const;

  DvView(const DvView &) = default;
  DvView &operator=(const DvView &) = default;

 private:
  word *data_;
  const int size_;
  const int aux_size_;
};

/**
 * @brief Provides a read-only view of a device vector.
 *
 * @tparam word uint32_t/uint64_t/int32_t/int64_t
 */
template <typename word>
class API_EXPORT DvConstView {
 public:
  DvConstView(const word *data, int size, int aux_size = 0);

  const word *data() const;
  int TotalSize() const;
  int AuxSize() const;
  int QSize() const;

  DvConstView(const DvConstView &) = default;
  DvConstView &operator=(const DvConstView &) = default;

  DvConstView(const DvView<word> &view);

 private:
  const word *data_;
  const int size_;
  const int aux_size_;
};

/**
 * @brief A wrapper around rmm::device_uvector used for various GPU memory
 * allocations.
 *
 * @tparam word uint32_t/uint64_t/int32_t/int64_t
 */
template <typename word>
class API_EXPORT DeviceVector : public rmm::device_uvector<word> {
 private:
  using Base = rmm::device_uvector<word>;
  using Base::resize;

 public:
  // A constructor without initilization.
  explicit DeviceVector(int size = 0, cudaStream_t stream = cudaStreamLegacy);

  // movable, but prevent unexplicit copying, instead use below copy functions
  DeviceVector(DeviceVector &&) = default;
  DeviceVector &operator=(DeviceVector &&) = default;

  /**
   * @brief Resize the DeviceVector, but without initialization.
   *
   * @param size The new size of the DeviceVector.
   */
  void resize(int size);

  /**
   * @brief Add zero padding to the back of the DeviceVector.
   *
   * @param size The size to be added. New size = old size + size.
   */
  void ZeroExtend(int size);

  /**
   * @brief Provides a view of the device vector.
   *
   * @param aux_size auxiliary part size (semantic information)
   * @param front_offset offset-part not accessible from the view
   * @return DvView<word> the view of the device vector
   */
  DvView<word> View(int aux_size = 0, int front_offset = 0);

  /**
   * @brief Provides a read-only view of the device vector.
   *
   * @param aux_size auxiliary part size (semantic information)
   * @param front_offset offset-part not accessible from the view
   * @return DvConstView<word> the read-only view of the device vector
   */
  DvConstView<word> ConstView(int aux_size = 0, int front_offset = 0) const;
};

template <typename word>
API_EXPORT void CopyHostToDevice(DeviceVector<word> &dst,
                                 const HostVector<word> &src);

template <typename word>
API_EXPORT void CopyDeviceToHost(HostVector<word> &dst,
                                 const DeviceVector<word> &src);

template <typename word>
API_EXPORT void CopyDeviceToDevice(DeviceVector<word> &dst,
                                   const DeviceVector<word> &src);

}  // namespace cheddar