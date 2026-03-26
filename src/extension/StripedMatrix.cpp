#include "extension/StripedMatrix.h"

#include "common/Assert.h"

namespace cheddar {

StripedMatrix::StripedMatrix(int height /*= 0*/, int width /*= 0*/)
    : Base(), height_(height), width_(width) {}

int StripedMatrix::GetHeight() const { return height_; }

int StripedMatrix::GetWidth() const { return width_; }

int StripedMatrix::GetNumDiag() const { return this->size(); }

StripedMatrix StripedMatrix::Mult(const StripedMatrix &a,
                                  const StripedMatrix &b) {
  AssertTrue(a.width_ == a.height_ && b.width_ == b.height_,
             "StripedMatrix must be square to perform multiplication");
  AssertTrue(a.GetHeight() == b.GetWidth(), "StripedMatrix dimension mismatch");
  int width = a.width_;

  StripedMatrix c(width, width);

  for (const auto &[i, diag_a] : a) {
    for (const auto &[j, diag_b] : b) {
      int dest_idx = (i + j) % width;
      if (c.find(dest_idx) == c.end())
        c.try_emplace(dest_idx, std::vector<Complex>(width));
      for (int k = 0; k < width; k++) {
        c[dest_idx][k] += (diag_a[k] * diag_b[(k + i) % width]);
      }
    }
  }
  return c;
}

StripedMatrix StripedMatrix::Mult(const StripedMatrix &a, const Complex b) {
  StripedMatrix c(a.GetHeight(), a.GetWidth());
  for (const auto &[i, diag] : a) {
    int diag_size = diag.size();
    c.try_emplace(i, diag_size);
    for (int j = 0; j < diag_size; j++) {
      c[i][j] = diag[j] * b;
    }
  }
  return c;
}

}  // namespace cheddar