#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "common/Assert.h"

namespace cheddar {
namespace detail {

/**
 * @brief Count the number of leading zeros for any integer type.
 *
 * @tparam Int any integer type
 * @param a the integer to count leading zeros
 * @return size_t the number of leading zeros
 */
template <typename Int>
size_t CountLeadingZeros(Int a) {
  static_assert(std::is_integral<Int>::value,
                "CountLeadingZeros only accepts integers");
  if (a < 0) return 0;
  constexpr size_t HALF_BIT = sizeof(Int) * 4;
  std::remove_cv_t<Int> a_high = a >> HALF_BIT;
  std::remove_cv_t<Int> a_low = (a << HALF_BIT) >> HALF_BIT;

  // Count trailing zeros in the MSB.
  size_t zero = 0;
  size_t bit = HALF_BIT;

  bool all_zero = (a_high == 0);
  if (all_zero) zero += bit;

  while (bit >= 2) {
    bit = bit >> 1;
    if (all_zero) {
      a_high = a_low >> bit;
      a_low = a_low - (a_high << bit);
    } else {
      std::remove_cv_t<Int> tmp = a_high >> bit;
      a_low = a_high - (tmp << bit);
      a_high = tmp;
    }
    all_zero = (a_high == 0);
    if (all_zero) zero += bit;
  }
  if (all_zero && a_low == 0) zero += 1;

  return zero;
}

}  // namespace detail

/**
 * @brief Calculate Ceiling(log2(a)). e.g., Log2Ceil(5) = 3, Log2Ceil(8) = 3
 *
 * @tparam Int any integer type
 * @param a the integer to calculate Log2Ceil
 * @return int the result of Ceiling(log2(a))
 */
template <typename Int>
int Log2Ceil(Int a) {
  static_assert(std::is_integral<Int>::value, "Log2Ceil only accepts integers");
  AssertFalse(a <= 0, "Log2Ceil only accepts positive values");
  int ret = sizeof(Int) * 8 - detail::CountLeadingZeros<Int>(a - 1);
  return ret;
}

/**
 * @brief Calculate Floor(log2(a)). e.g., Log2Floor(5) = 2, Log2Floor(8) = 3
 *
 * @tparam Int any integer type
 * @param a the integer to calculate Log2Floor
 * @return int the result of Floor(log2(a))
 */
template <typename Int>
int Log2Floor(Int a) {
  static_assert(std::is_integral<Int>::value,
                "Log2Floor only accepts integers");
  AssertFalse(a <= 0, "Log2Floor only accepts positive values");
  int ret = sizeof(Int) * 8 - detail::CountLeadingZeros<Int>(a) - 1;
  return ret;
}

/**
 * @brief Calculate Ceiling(a / n). e.g., DivCeil(5, 3) = 2, DivCeil(8, 2) = 4
 *
 * @tparam Int any integer type
 * @param a the dividend
 * @param n the divisor
 * @return int the result of Ceiling(a / n)
 */
template <typename Int>
int DivCeil(Int a, Int n) {
  static_assert(std::is_integral<Int>::value, "DivCeil only accepts integers");
  return (a + n - 1) / n;
}

/**
 * @brief Bit reverse the integer a. e.g., BitReverseInt(0b0011, 3) = 0b0110
 *
 * @tparam Int any integer type
 * @param a the integer to be reversed
 * @param num_bits the number of bits to be reversed (<= sizeof(Int) * 8)
 * @return std::make_unsigned_t<Int> the reversed integer
 */
template <typename Int>
std::make_unsigned_t<Int> BitReverseInt(Int a, int num_bits) {
  static const unsigned char table[] = {
      0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0,
      0x30, 0xb0, 0x70, 0xf0, 0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
      0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4,
      0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
      0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc,
      0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
      0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca,
      0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
      0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6,
      0x36, 0xb6, 0x76, 0xf6, 0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
      0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1,
      0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
      0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9,
      0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
      0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd,
      0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
      0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3,
      0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
      0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 0x07, 0x87, 0x47, 0xc7,
      0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
      0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
      0x3f, 0xbf, 0x7f, 0xff,
  };
  static_assert(std::is_integral<Int>::value,
                "BitReverseInt only accepts integers");
  constexpr int max_num_bits = sizeof(Int) * 8;
  AssertTrue(num_bits <= max_num_bits,
             "BitReverseInt only accepts num_bits <= sizeof(Int) * 8");

  // Bit-reverse using a lookup table (byte by byte)
  using UInt = std::remove_cv_t<std::make_unsigned_t<Int>>;
  UInt input = static_cast<UInt>(a);
  UInt ret = 0;
  constexpr int num_bytes = sizeof(UInt);
  for (int i = 0; i < num_bytes; i++) {
    ret = (ret << 8) | static_cast<UInt>(table[(input >> (i * 8)) & 0xff]);
  }

  // Discard the bits that are not in the range of num_bits
  return ret >> (sizeof(UInt) * 8 - num_bits);
}

/**
 * @brief Bit reverse the vector data. e.g.,
 * BitReverseVector({0, 1, 2, 3}) = {0, 2, 1, 3}
 * The length of the result will be zero-padded to the next power of 2.
 * e.g., BitReverseVector({0, 1, 2}) = {0, 2, 1, 0}
 *
 * @tparam T any move constructible and move assignable type
 * @param data the vector to be reversed
 */
template <typename T>
void BitReverseVector(std::vector<T> &data) {
  static_assert(
      std::is_move_constructible<T>::value && std::is_move_assignable<T>::value,
      "BitReverseVector only accepts move constructible and move assignable "
      "types");
  int num_slots = data.size();
  int log_slots = Log2Ceil(num_slots);
  data.resize(1 << log_slots);
  for (int i = 0; i < num_slots; i++) {
    int j = BitReverseInt(i, log_slots);
    if (i < j) {
      std::swap(data[i], data[j]);
    }
  }
}

/**
 * @brief Check if the integer a is a power of 2.
 *
 * @tparam Int any integer type
 * @param a the integer to be checked
 * @return true if a is a power of 2
 * @return false if a is not a power of 2
 */
template <typename Int>
bool IsPowOfTwo(Int a) {
  static_assert(std::is_integral<Int>::value,
                "IsPowOfTwo only accepts integers");
  AssertTrue(a > 0, "IsPowOfTwo only accepts positive values");
  return (a & (a - 1)) == 0;
}

template <typename Any>
Any Min(Any a, Any b) {
  static_assert(std::is_arithmetic<Any>::value, "Min only accepts arithmetic");
  return a > b ? b : a;
}

template <typename Any, typename... Args>
Any Min(Any a, Any b, Args... args) {
  return Min(a, Min(b, args...));
}

template <typename Any>
Any Max(Any a, Any b) {
  static_assert(std::is_arithmetic<Any>::value, "Max only accepts arithmetic");
  return a > b ? a : b;
}

template <typename Any, typename... Args>
Any Max(Any a, Any b, Args... args) {
  return Max(a, Max(b, args...));
}

template <typename Any>
Any Abs(Any a) {
  static_assert(std::is_arithmetic<Any>::value, "Abs only accepts arithmetic");
  return a < 0 ? -a : a;
}

template <typename Int>
Int GCD(Int a, Int b) {
  static_assert(std::is_integral<Int>::value, "GCD only accepts integers");
  if (b == 0) {
    return a;
  } else {
    return GCD(b, a % b);
  }
}
template <typename Int, typename... Args>
Int GCD(Int a, Int b, Args... args) {
  Int result = GCD(a, b);
  result = GCD(result, args...);
  return result;
}

}  // namespace cheddar
