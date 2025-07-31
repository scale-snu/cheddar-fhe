#pragma once

#include <tommath.h>

#include <cstdint>
#include <memory>

#include "Export.h"

namespace cheddar {

/**
 * @brief A thin wrapper around Tommath's mp_int type. This class is used to
 * represent large integers and perform arithmetic operations on them.
 */
class API_EXPORT BigInt {
 public:
  static_assert(sizeof(unsigned long int) == 8,
                "Unsigned long int should be a 64-bit integer");

  explicit BigInt(uint64_t value);
  explicit BigInt(double value);

  // Copyable
  BigInt(const BigInt& other);
  BigInt& operator=(const BigInt& other);

  ~BigInt();

  // --- Get the value ---

  uint64_t GetUnsigned() const;
  double GetDouble() const;

  // --- basic operations ---

  static void Neg(BigInt& result, const BigInt& op);

  // result = op1 + op2
  static void Add(BigInt& result, const BigInt& op1, const BigInt& op2);

  // result = op1 - op2
  static void Sub(BigInt& result, const BigInt& op1, const BigInt& op2);

  // result = op1 * op2
  static void Mult(BigInt& result, const BigInt& op1, const BigInt& op2);

  // --- more complex operations ---

  // result = op >> 1
  static void Div2(BigInt& result, const BigInt& op);

  // result = op % mod. The result is always positive.
  static void Mod(BigInt& result, const BigInt& op, const BigInt& mod);

  // result = op % mod. The result is in range [- (mod - 1) / 2, (mod - 1) / 2]
  static void NormalizeMod(BigInt& result, const BigInt& op, const BigInt& mod,
                           const BigInt& half_mod);

 private:
  mp_int* data_;
};

}  // namespace cheddar
