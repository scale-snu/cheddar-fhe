#include "common/Assert.h"
#include "core/BigInt.h"

#define CHECK_GMP_ERROR(name, expr) \
  do {                              \
    try {                           \
      expr;                         \
    } catch (...) {                 \
      std::string __err_msg = name; \
      __err_msg += " failed";       \
      Fail(__err_msg);              \
    }                               \
  } while (0)

namespace cheddar {

BigInt::BigInt(uint64_t value) {
  mpz_init(data_);
  mpz_set_ui(data_, value);
}

BigInt::BigInt(double value) {
  mpz_init(data_);
  mpz_set_d(data_, value);
}

BigInt::BigInt(const BigInt &other) {
  mpz_init(data_);
  mpz_set(data_, other.data_);
}

BigInt &BigInt::operator=(const BigInt &other) {
  if (this != &other) {
    mpz_set(data_, other.data_);
  }
  return *this;
}

BigInt::~BigInt() { mpz_clear(data_); }

uint64_t BigInt::GetUnsigned() const {
  AssertFalse(mpz_sgn(data_) < 0, "BigInt::GetUnsigned: negative value");
  return mpz_get_ui(data_);
}

double BigInt::GetDouble() const { return mpz_get_d(data_); }

// result = -op
void BigInt::Neg(BigInt &result, const BigInt &op) {
  mpz_neg(result.data_, op.data_);
}

// result = op1 + op2
void BigInt::Add(BigInt &result, const BigInt &op1, const BigInt &op2) {
  mpz_add(result.data_, op1.data_, op2.data_);
}

// result = op1 - op2
void BigInt::Sub(BigInt &result, const BigInt &op1, const BigInt &op2) {
  mpz_sub(result.data_, op1.data_, op2.data_);
}

// result = op1 * op2
void BigInt::Mult(BigInt &result, const BigInt &op1, const BigInt &op2) {
  mpz_mul(result.data_, op1.data_, op2.data_);
}

// --- more complex operations ---

// result = op >> 1
void BigInt::Div2(BigInt &result, const BigInt &op) {
  AssertFalse(mpz_sgn(op.data_) < 0, "BigInt::Div2: negative value");
  mpz_fdiv_q_2exp(result.data_, op.data_, 1);
}

// result = op % mod. The result is always positive.
void BigInt::Mod(BigInt &result, const BigInt &op, const BigInt &mod) {
  mpz_mod(result.data_, op.data_, mod.data_);
}

// result = op % mod. The result is in range [- (mod - 1) / 2, (mod - 1) / 2]
void BigInt::NormalizeMod(BigInt &result, const BigInt &op, const BigInt &mod,
                          const BigInt &half_mod) {
  mpz_mod(result.data_, op.data_, mod.data_);
  if (mpz_cmp(result.data_, half_mod.data_) > 0) {  // result > half_mod
    mpz_sub(result.data_, result.data_, mod.data_);
  }
}

}  // namespace cheddar
