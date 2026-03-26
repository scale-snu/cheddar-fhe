#include "core/DeviceVector.h"

#include <cstdint>

#include "common/Assert.h"

namespace cheddar {

template <typename word>
DvView<word>::DvView(word *data, int size, int aux_size /*= 0*/)
    : data_(data), size_(size), aux_size_(aux_size) {}

template <typename word>
word *DvView<word>::data() {
  return data_;
}

template <typename word>
const word *DvView<word>::data() const {
  return data_;
}

template <typename word>
int DvView<word>::TotalSize() const {
  return size_;
}

template <typename word>
int DvView<word>::AuxSize() const {
  return aux_size_;
}

template <typename word>
int DvView<word>::QSize() const {
  return size_ - aux_size_;
}

template <typename word>
DvConstView<word>::DvConstView(const word *data, int size, int aux_size /*= 0*/)
    : data_(data), size_(size), aux_size_(aux_size) {}

template <typename word>
const word *DvConstView<word>::data() const {
  return data_;
}

template <typename word>
int DvConstView<word>::TotalSize() const {
  return size_;
}

template <typename word>
int DvConstView<word>::AuxSize() const {
  return aux_size_;
}

template <typename word>
int DvConstView<word>::QSize() const {
  return size_ - aux_size_;
}

template <typename word>
DvConstView<word>::DvConstView(const DvView<word> &view)
    : data_(view.data()), size_(view.TotalSize()), aux_size_(view.AuxSize()) {}

template <typename word>
DeviceVector<word>::DeviceVector(int size /*= 0*/,
                                 cudaStream_t stream /*= cudaStreamLegacy*/)
    : Base(size, stream) {}

template <typename word>
void DeviceVector<word>::resize(int size) {
  this->resize(size, this->stream());
}

template <typename word>
void DeviceVector<word>::ZeroExtend(int size) {
  AssertTrue(size >= 0,
             "DeviceVector::ZeroExtend: size must be positive, but got " +
                 std::to_string(size));
  if (size == 0) return;
  auto old_size = this->size();
  resize(old_size + size);
  cudaMemsetAsync(this->data() + old_size, 0, size * sizeof(word),
                  this->stream());
}

template <typename word>
DvView<word> DeviceVector<word>::View(int aux_size /*= 0*/,
                                      int front_offset /*= 0*/) {
  return DvView<word>(this->data() + front_offset, this->size() - front_offset,
                      aux_size);
}

template <typename word>
DvConstView<word> DeviceVector<word>::ConstView(
    int aux_size /*= 0*/, int front_offset /*= 0*/) const {
  return DvConstView<word>(this->data() + front_offset,
                           this->size() - front_offset, aux_size);
}

template <typename word>
void CopyHostToDevice(DeviceVector<word> &dst, const HostVector<word> &src) {
  dst.resize(src.size());
  cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(word),
                  cudaMemcpyHostToDevice, dst.stream());
}

template <typename word>
void CopyDeviceToHost(HostVector<word> &dst, const DeviceVector<word> &src) {
  dst.resize(src.size());
  cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(word),
                  cudaMemcpyDeviceToHost, src.stream());
}

template <typename word>
void CopyDeviceToDevice(DeviceVector<word> &dst,
                        const DeviceVector<word> &src) {
  dst.resize(src.size());
  CheckTrue(dst.stream() == src.stream(),
            "CopyDeviceToDevice: Copying between different streams...");
  if (src.data() == dst.data()) return;
  cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(word),
                  cudaMemcpyDeviceToDevice, dst.stream());
}

// Explicit instantiation of the template classes
template class DvView<int32_t>;
template class DvView<int64_t>;
template class DvView<uint32_t>;
template class DvView<uint64_t>;
template class DvView<uint32_t *>;
template class DvView<uint64_t *>;
template class DvView<const uint32_t *>;
template class DvView<const uint64_t *>;
template class DvConstView<int32_t>;
template class DvConstView<int64_t>;
template class DvConstView<uint32_t>;
template class DvConstView<uint64_t>;
template class DvConstView<uint32_t *>;
template class DvConstView<uint64_t *>;
template class DvConstView<const uint32_t *>;
template class DvConstView<const uint64_t *>;
template class DeviceVector<int32_t>;
template class DeviceVector<int64_t>;
template class DeviceVector<uint32_t>;
template class DeviceVector<uint64_t>;
template class DeviceVector<uint32_t *>;
template class DeviceVector<uint64_t *>;
template class DeviceVector<const uint32_t *>;
template class DeviceVector<const uint64_t *>;

// Explicit instantiation of the template functions
template void CopyHostToDevice(DeviceVector<int32_t> &dst,
                               const HostVector<int32_t> &src);
template void CopyHostToDevice(DeviceVector<int64_t> &dst,
                               const HostVector<int64_t> &src);
template void CopyHostToDevice(DeviceVector<uint32_t> &dst,
                               const HostVector<uint32_t> &src);
template void CopyHostToDevice(DeviceVector<uint64_t> &dst,
                               const HostVector<uint64_t> &src);
template void CopyHostToDevice(DeviceVector<uint32_t *> &dst,
                               const HostVector<uint32_t *> &src);
template void CopyHostToDevice(DeviceVector<uint64_t *> &dst,
                               const HostVector<uint64_t *> &src);
template void CopyHostToDevice(DeviceVector<const uint32_t *> &dst,
                               const HostVector<const uint32_t *> &src);
template void CopyHostToDevice(DeviceVector<const uint64_t *> &dst,
                               const HostVector<const uint64_t *> &src);
template void CopyDeviceToHost(HostVector<int32_t> &dst,
                               const DeviceVector<int32_t> &src);
template void CopyDeviceToHost(HostVector<int64_t> &dst,
                               const DeviceVector<int64_t> &src);
template void CopyDeviceToHost(HostVector<uint32_t> &dst,
                               const DeviceVector<uint32_t> &src);
template void CopyDeviceToHost(HostVector<uint64_t> &dst,
                               const DeviceVector<uint64_t> &src);
template void CopyDeviceToHost(HostVector<uint32_t *> &dst,
                               const DeviceVector<uint32_t *> &src);
template void CopyDeviceToHost(HostVector<uint64_t *> &dst,
                               const DeviceVector<uint64_t *> &src);
template void CopyDeviceToHost(HostVector<const uint32_t *> &dst,
                               const DeviceVector<const uint32_t *> &src);
template void CopyDeviceToHost(HostVector<const uint64_t *> &dst,
                               const DeviceVector<const uint64_t *> &src);
template void CopyDeviceToDevice(DeviceVector<int32_t> &dst,
                                 const DeviceVector<int32_t> &src);
template void CopyDeviceToDevice(DeviceVector<int64_t> &dst,
                                 const DeviceVector<int64_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint32_t> &dst,
                                 const DeviceVector<uint32_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint64_t> &dst,
                                 const DeviceVector<uint64_t> &src);
template void CopyDeviceToDevice(DeviceVector<uint32_t *> &dst,
                                 const DeviceVector<uint32_t *> &src);
template void CopyDeviceToDevice(DeviceVector<uint64_t *> &dst,
                                 const DeviceVector<uint64_t *> &src);
template void CopyDeviceToDevice(DeviceVector<const uint32_t *> &dst,
                                 const DeviceVector<const uint32_t *> &src);
template void CopyDeviceToDevice(DeviceVector<const uint64_t *> &dst,
                                 const DeviceVector<const uint64_t *> &src);

}  // namespace cheddar