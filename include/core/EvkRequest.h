#pragma once

#include <map>

namespace cheddar {

class EvkRequest : public std::map<int, int> {
  using Base = std::map<int, int>;

 public:
  using Base::Base;

  void AddRequest(int rot_idx, int level);
};

}  // namespace cheddar