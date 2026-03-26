#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "common/DoubleWord.h"
#include "core/Type.h"

namespace cheddar {
// Functions in the "primeutil" namespace do not care much about performance
namespace primeutil {
namespace detail {

/**
 * @brief Performs extended Euclidean algorithm to find x, y satisfying
 * ax + by = gcd(a, b). x, y can be negative.
 *
 * @param a         any number
 * @param b         any number
 * @param x         result x (abs(x) < b/(2 * gcd) if gcd != min(a, b))
 * @param y         result y (abs(y) < a/(2 * gcd) if gcd != min(a, b))
 * @return int64_t  gcd of a and b (positive if a and b are positive)
 */
// https://www.geeksforgeeks.org/euclidean-algorithms-basic-and-extended/
template <typename T>
T __gcd_extended(T a, T b, T &x, T &y) {
  static_assert(std::is_integral_v<T>,
                "__gcd_extended requires integral types to work");
  static_assert(std::is_signed_v<T>,
                "__gcd_extended requires signed types to work");

  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  T x1, y1;
  T gcd = __gcd_extended<T>(b % a, a, x1, y1);
  x = y1 - (b / a) * x1;
  y = x1;

  return gcd;
}

}  // namespace detail

/**
 * @brief Computes (a * b) % q, simply using the operator %.
 *
 * @tparam word  either uint32_t or uint64_t
 * @param a      any number
 * @param b      any number
 * @param q      the prime modulus
 * @return word  (a * b) % q
 */
template <typename word>
word MultMod(const word a, const word b, const word q) {
  auto c = static_cast<make_double_word_t<word>>(a) * b;
  return static_cast<word>(c % q);
}

/**
 * @brief Compute (a^(-1)) mod q
 *
 * @tparam word  either uint32_t or uint64_t
 * @param a      any number
 * @param q      the prime modulus
 * @return word  inverse of a mod q
 */
template <typename word>
word InvMod(word a, const word q) {
  a = a % q;
  if (a == 0) {
    std::cerr << "Inverse of 0 is undefined in InvMod" << std::endl;
    exit(EXIT_FAILURE);
  }
  make_signed_t<word> x, y;
  detail::__gcd_extended<make_signed_t<word>>(a, q, x, y);
  // a * x + q * y = 1 --> a * x = 1 mod q --> x = a^(-1) mod q
  if (x < 0) x += q;
  return static_cast<word>(x);
}

/**
 * @brief Computes a^x % q
 *
 * @tparam word  either uint32_t or uint64_t
 * @param a      any number
 * @param x      the exponent
 * @param q      the prime modulus
 * @return word  a^x % q
 */
template <typename word>
word PowMod(word a, int64_t x, const word q) {
  // Fermat's little theorem a^(q-1) = 1 mod q always holds for a prime q.
  a = a % q;
  if (a == 0) {
    if (x >= 0) {
      return 0;
    }
    std::cerr << "Negative power of 0 is undefined in PowMod" << std::endl;
    exit(EXIT_FAILURE);
  }
  word exp_mod = q - 1;
  x = x % static_cast<int64_t>(exp_mod);
  uint64_t x_value = std::abs(x);
  bool sign = (x >= 0);
  if (x_value > (exp_mod >> 1)) {
    x_value = exp_mod - x_value;
    sign = !sign;
  }
  // negative sign
  if (sign == false) {
    a = InvMod<word>(a, q);
  }
  word res = 1;
  while (x_value > 0) {
    if (x_value & 1) {
      res = MultMod<word>(res, a, q);
    }
    x_value >>= 1;
    a = MultMod<word>(a, a, q);
  }
  return res;
}

/**
 * @brief Find a generator in the finite field of q
 *
 * @tparam word  either uint32_t or uint64_t
 * @param q      prime number
 * @return word  smallest generator in the finite field of q
 */
template <typename word>
word FindGenerator(word q) {
  word phi = q - 1;

  // find the prime factors of phi
  std::vector<word> prime_factors;
  word temp = phi;
  word sqrt_phi = static_cast<word>(std::sqrt(phi));
  for (word i = 2; i <= sqrt_phi; i++) {
    if (temp % i == 0) {
      prime_factors.push_back(i);
      while (temp % i == 0) {
        temp /= i;
      }
      if (temp == 1) break;
    }
  }
  if (temp > 1) {
    prime_factors.push_back(temp);
  }

  // It is guaranteed that for any r, r^phi = 1 mod q.
  // To verify whether r is a generator,
  // we only need to make sure
  // r^(phi/factor) != 1 mod q
  // for all prime factors.
  for (word r = 2; r <= phi; r++) {
    bool found = true;
    for (word factor : prime_factors) {
      if (PowMod(r, static_cast<int64_t>(phi / factor), q) == 1) {
        found = false;
        break;
      }
    }
    if (found) return r;
  }
  std::cerr << "Cannot find a generator" << std::endl;
  exit(EXIT_FAILURE);
  return 1;
}

/**
 * @brief Find a primitive M-th root of unity in the finite field of q
 *
 * @tparam word either uint32_t or uint64_t
 * @param M     order of the root of unity
 * @param q     prime number
 * @return word a primitive M-th root of unity
 */
template <typename word>
word FindPrimitiveMthRoot(int M, word q) {
  word g = FindGenerator(q);

  if ((q - 1) % M == 0) {
    word factor = (q - 1) / M;
    g = PowMod(g, factor, q);
    return g;
  }
  std::cerr << "Prime number is not NTT-friendly" << std::endl;
  exit(EXIT_FAILURE);
  return 1;
}

/**
 * @brief Find the inverse of q mod base (base = 2^word_size)
 *
 * @tparam word                either uint32_t or uint64_t
 * @param q                    prime number
 * @return make_signed_t<word> q^(-1) mod base
 */
template <typename word>
make_signed_t<word> InvModBase(const word q) {
  word base = 0;
  base = ~base;  // it's actually base - 1;

  // base = q * quotient + remainder
  word quotient = base / q;
  word remainder = (base % q) + 1;

  make_signed_t<word> x1, y1;
  // remainder * x1 + q * y1 = 1
  detail::__gcd_extended<make_signed_t<word>>(remainder, q, x1, y1);

  // We can get q^(-1) mod base based on the following equations:
  // base * x1 + q * y1 = (q * quotient + remainder) * x1 + q * y1
  //                    = q * quotient * x1 + 1
  // base * x1 + q * (y1 - quotient * x1) = 1
  // q * (y1 - quotient * x1) = 1 mod base
  // y1 - quotient * x1 = q^(-1) mod base
  make_signed_t<word> inv_q =
      y1 - static_cast<make_signed_t<word>>(quotient) * x1;
  // abs(x1) < (q / 2)
  // abs(y1) < remainder / 2
  // quotient * x1 = (base - remainder) / q * x1
  // abs(quotient * x1) < (base - remainder) / 2
  // abs(y1 - quotient * x1) < base / 2
  return inv_q;
}

/**
 * @brief Convert a number into the Montgomery form
 *
 * @tparam word either uint32_t or uint64_t
 * @param a     any number
 * @param q     prime number
 * @return word a * 2^word_size mod q
 */
template <typename word>
word ToMontgomery(const word a, const word q) {
  constexpr auto word_bit_width = sizeof(word) * 8;
  make_double_word_t<word> t = a;
  t <<= word_bit_width;
  return static_cast<word>(t % q);
}

}  // namespace primeutil
}  // namespace cheddar
