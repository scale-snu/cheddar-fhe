#include "core/EvkRequest.h"

#include "common/Assert.h"

namespace cheddar {

void EvkRequest::AddRequest(int rot_idx, int level) {
  AssertTrue(rot_idx >= 0, "Invalid rotation index");
  AssertTrue(level >= 0, "Invalid level");

  if (this->find(rot_idx) == this->end()) {
    this->try_emplace(rot_idx, level);
  } else if (this->at(rot_idx) < level) {
    this->at(rot_idx) = level;
  }
}

}  // namespace cheddar