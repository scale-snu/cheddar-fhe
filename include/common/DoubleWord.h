#pragma once

#include <cstdint>
#include <type_traits>

#include "core/Type.h"

namespace cheddar {
namespace detail {

template <typename _Tp>
struct __make_double_word {
  typedef _Tp __type;
};

template <>
struct __make_double_word<int8_t> {
  typedef int16_t __type;
};

template <>
struct __make_double_word<int16_t> {
  typedef int32_t __type;
};

template <>
struct __make_double_word<int32_t> {
  typedef int64_t __type;
};

template <>
struct __make_double_word<int64_t> {
  typedef int128_t __type;
};

template <>
struct __make_double_word<uint8_t> {
  typedef uint16_t __type;
};

template <>
struct __make_double_word<uint16_t> {
  typedef uint32_t __type;
};

template <>
struct __make_double_word<uint32_t> {
  typedef uint64_t __type;
};

template <>
struct __make_double_word<uint64_t> {
  typedef uint128_t __type;
};

}  // namespace detail

template <typename _Tp>
class make_double_word {
  using __clean_type = typename std::remove_cv<_Tp>::type;

  static_assert(std::is_integral_v<__clean_type>,
                "make_double_word only accepts integral types");
  static_assert(!std::is_same_v<__clean_type, bool>,
                "make_double_word cannot handle boolean");
  // just don't care about cv qualifiers
 public:
  using type = typename detail::__make_double_word<__clean_type>::__type;
};

/**
 * @brief A type trait to get the double bit-width type of a given integral
 * type.
 *
 * @tparam word integral type (except bool)
 */
template <typename word>
using make_double_word_t = typename make_double_word<word>::type;

/**
 * @brief Shorthand for make_double_word_t<make_signed_t<word>>
 *
 * @tparam word integral type (except bool)
 */
template <typename word>
using make_signed_double_word_t = make_double_word_t<make_signed_t<word>>;

/**
 * @brief Shorthand for make_double_word_t<make_unsigned_t<word>>
 *
 * @tparam word integral type (except bool)
 */
template <typename word>
using make_unsigned_double_word_t = make_double_word_t<make_unsigned_t<word>>;

}  // namespace cheddar
