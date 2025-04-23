#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

namespace cheddar {

// nvcc version >= 11.6 should support __int128
using int128_t = __int128_t;
using uint128_t = __uint128_t;

using Complex = std::complex<double>;

using std::make_signed;
using std::make_signed_t;
using std::make_unsigned;
using std::make_unsigned_t;

}  // namespace cheddar
