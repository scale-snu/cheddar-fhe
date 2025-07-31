#pragma once

#include <map>
#include <set>
#include <vector>

#include "Export.h"
#include "core/Type.h"

namespace cheddar {

class API_EXPORT StripedMatrix : public std::map<int, std::vector<Complex>> {
 private:
  using Base = std::map<int, std::vector<Complex>>;
  int height_;
  int width_;

 public:
  explicit StripedMatrix(int height = 0, int width = 0);

  StripedMatrix(const StripedMatrix &) = default;
  StripedMatrix &operator=(const StripedMatrix &) = default;
  StripedMatrix(StripedMatrix &&) = default;
  StripedMatrix &operator=(StripedMatrix &&) = default;

  int GetHeight() const;
  int GetWidth() const;
  int GetNumDiag() const;

  static StripedMatrix Mult(const StripedMatrix &a, const StripedMatrix &b);
  static StripedMatrix Mult(const StripedMatrix &a, const Complex b);
};

}  // namespace cheddar
