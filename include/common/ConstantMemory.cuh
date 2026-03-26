#pragma once

#include <cstdint>
#include <vector>

#include "core/Parameter.h"

__constant__ int __cm_log_degree;
__constant__ int __cm_degree;
__constant__ int __cm_alpha;
__constant__ int __cm_L;

namespace cheddar {

template <typename word>
static void PopulateConstantMemory(const Parameter<word> &param) {
  /*
  std::vector<make_signed_t<word>> primes_inv(num_primes);
  for (int i = 0; i < num_primes; i++) {
    primes_inv[i] = primeutil::InvModBase<word>(param.primes_[i]);
  }
  */

  int log_degree = param.log_degree_;
  int degree = 1 << log_degree;
  int alpha = param.alpha_;
  int L = param.L_;

  cudaMemcpyToSymbol(__cm_log_degree, &log_degree, sizeof(int));
  cudaMemcpyToSymbol(__cm_degree, &degree, sizeof(int));
  cudaMemcpyToSymbol(__cm_alpha, &alpha, sizeof(int));
  cudaMemcpyToSymbol(__cm_L, &L, sizeof(int));
}

}  // namespace cheddar

// Macros for usage in template using <typename word>
#define cm_log_degree() (__cm_log_degree)
#define cm_degree() (__cm_degree)
#define cm_alpha() (__cm_alpha)
#define cm_L() (__cm_L)
