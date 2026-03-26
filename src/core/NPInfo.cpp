#include "core/NPInfo.h"

#include "common/Assert.h"

namespace cheddar {

int NPInfo::GetNumQ() const { return num_main_ + num_ter_; }
int NPInfo::GetNumTotal() const { return num_main_ + num_ter_ + num_aux_; }

NPInfo::NPInfo(int num_main /*= 0*/, int num_ter /*= 0*/, int num_aux /*= 0*/)
    : num_main_{num_main}, num_ter_{num_ter}, num_aux_{num_aux} {
  AssertTrue(num_main >= 0, "Negative num_main given");
  AssertTrue(num_ter >= 0, "Negative num_ter given");
  AssertTrue(num_aux >= 0, "Negative num_aux given");
}

NPInfo::NPInfo(const NPInfo &other) {
  AssertTrue(other.num_main_ >= 0, "Negative num_main given");
  AssertTrue(other.num_ter_ >= 0, "Negative num_ter given");
  AssertTrue(other.num_aux_ >= 0, "Negative num_aux given");
  num_main_ = other.num_main_;
  num_ter_ = other.num_ter_;
  num_aux_ = other.num_aux_;
}

NPInfo &NPInfo::operator=(const NPInfo &other) {
  AssertTrue(other.num_main_ >= 0, "Negative num_main given");
  AssertTrue(other.num_ter_ >= 0, "Negative num_ter given");
  AssertTrue(other.num_aux_ >= 0, "Negative num_aux given");
  num_main_ = other.num_main_;
  num_ter_ = other.num_ter_;
  num_aux_ = other.num_aux_;
  return *this;
}

bool NPInfo::operator==(const NPInfo &other) const {
  return num_main_ == other.num_main_ && num_ter_ == other.num_ter_ &&
         num_aux_ == other.num_aux_;
}

bool NPInfo::IsSubsetOf(const NPInfo &other) const {
  return num_main_ <= other.num_main_ && num_ter_ <= other.num_ter_ &&
         num_aux_ <= other.num_aux_;
}

bool NPInfo::IsSupersetOf(const NPInfo &other) const {
  return num_main_ >= other.num_main_ && num_ter_ >= other.num_ter_ &&
         num_aux_ >= other.num_aux_;
}
}  // namespace cheddar