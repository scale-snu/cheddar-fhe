#pragma once

#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "Export.h"
#include "core/Parameter.h"

namespace cheddar {
// After the creation of an MemoryPool object, all memory allocations on the
// current device uses binning_memory_resouce.

class API_EXPORT MemoryPool {
  using DefaultUpstream = rmm::mr::cuda_async_memory_resource;
  using MemoryPoolBase = rmm::mr::binning_memory_resource<DefaultUpstream>;

 public:
  template <typename word>
  explicit MemoryPool(const Parameter<word>& param);
  ~MemoryPool();

 private:
  DefaultUpstream base_;
  MemoryPoolBase memory_pool_;
};

}  // namespace cheddar