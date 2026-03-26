#include "core/MemoryPool.h"

namespace cheddar {

template <typename word>
MemoryPool::MemoryPool(const Parameter<word> &param)
    : base_(), memory_pool_(&base_) {
  // Hueristically add bins to save memory and speed-up bootstrapping.
  const int degree = param.degree_;
  const int word_size = param.word_size_;
  int limb_size = word_size * degree;

  // Some general sizes for small allocation sizes;
  int bin_size = 512;
  int next_threshold = limb_size;
  // Should be: 512, 2048, 8192, 32768, 131072
  for (; bin_size < next_threshold; bin_size *= 4) {
    memory_pool_.add_bin(bin_size);
  }
  bin_size = next_threshold;
  int chunk_size = param.alpha_ * limb_size;
  for (; bin_size < chunk_size; bin_size *= 2) {
    // Maybe one more bin will be added
    memory_pool_.add_bin(bin_size);
  }
  // Finally, about dnum additional bins;
  bin_size = chunk_size;
  int max_size = (param.L_ + param.alpha_) * limb_size;
  for (; bin_size < max_size; bin_size += chunk_size) {
    memory_pool_.add_bin(bin_size);
  }
  memory_pool_.add_bin(max_size);

  rmm::mr::set_current_device_resource(&memory_pool_);
}

MemoryPool::~MemoryPool() {
  // reset to cuda_device_resource
  rmm::mr::set_current_device_resource(nullptr);
}

template MemoryPool::MemoryPool(const Parameter<uint32_t> &param);
template MemoryPool::MemoryPool(const Parameter<uint64_t> &param);

}  // namespace cheddar