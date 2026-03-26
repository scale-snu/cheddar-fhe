#include "common/Assert.h"
#include "core/BigInt.h"

#define CHECK_MP_ERROR(command, name)         \
  do {                                        \
    auto __err = (command);                   \
    if (__err) {                              \
      std::string __err_msg = name;           \
      __err_msg += " failed: ";               \
      __err_msg += mp_error_to_string(__err); \
      Fail(__err_msg);                        \
    }                                         \
  } while (0)

namespace cheddar {

BigInt::BigInt(uint64_t value) {
  data_ = new mp_int;
  CHECK_MP_ERROR(mp_init_u64(data_, value), "BigInt::BigInt(uint64_t)");
}

BigInt::BigInt(double value) {
  data_ = new mp_int;
  CHECK_MP_ERROR(mp_init(data_), "BigInt::BigInt(double)");
  CHECK_MP_ERROR(mp_set_double(data_, value), "BigInt::BigInt(double)");
}

BigInt::BigInt(const BigInt &other) {
  data_ = new mp_int;
  CHECK_MP_ERROR(mp_init_copy(data_, other.data_),
                 "BigInt::BigInt(const BigInt&)");
}

BigInt &BigInt::operator=(const BigInt &other) {
  if (this != &other) {
    CHECK_MP_ERROR(mp_copy(other.data_, data_), "BigInt::operator=");
  }
  return *this;
}

BigInt::~BigInt() {
  mp_clear(data_);
  delete data_;
}

uint64_t BigInt::GetUnsigned() const {
  AssertFalse(mp_isneg(data_), "BigInt::GetUnsigned: negative value");
  return mp_get_u64(data_);
}
double BigInt::GetDouble() const { return mp_get_double(data_); }

// result = -op
void BigInt::Neg(BigInt &result, const BigInt &op) {
  CHECK_MP_ERROR(mp_neg(op.data_, result.data_), "BigInt::Neg");
}

// result = op1 + op2
void BigInt::Add(BigInt &result, const BigInt &op1, const BigInt &op2) {
  CHECK_MP_ERROR(mp_add(op1.data_, op2.data_, result.data_), "BigInt::Add");
}

// result = op1 - op2
void BigInt::Sub(BigInt &result, const BigInt &op1, const BigInt &op2) {
  CHECK_MP_ERROR(mp_sub(op1.data_, op2.data_, result.data_), "BigInt::Sub");
}

// result = op1 * op2
void BigInt::Mult(BigInt &result, const BigInt &op1, const BigInt &op2) {
  CHECK_MP_ERROR(mp_mul(op1.data_, op2.data_, result.data_), "BigInt::Mult");
}
// --- more complex operations ---

// result = op >> 1
void BigInt::Div2(BigInt &result, const BigInt &op) {
  AssertFalse(mp_isneg(op.data_), "BigInt::Div2: negative value");
  CHECK_MP_ERROR(mp_div_2(op.data_, result.data_), "BigInt::Div2");
}

// result = op % mod. The result is always positive.
void BigInt::Mod(BigInt &result, const BigInt &op, const BigInt &mod) {
  // Do not check sign for performance
  CHECK_MP_ERROR(mp_mod(op.data_, mod.data_, result.data_), "BigInt::Mod");
}

// result = op % mod. The result is in range [- (mod - 1) / 2, (mod - 1) / 2]
void BigInt::NormalizeMod(BigInt &result, const BigInt &op, const BigInt &mod,
                          const BigInt &half_mod) {
  // Do not check sign for performance
  CHECK_MP_ERROR(mp_mod(op.data_, mod.data_, result.data_),
                 "BigInt::NormalizeMod");
  auto sign = mp_cmp(result.data_, half_mod.data_);
  if (sign == MP_GT) {  // result > half_mod
    CHECK_MP_ERROR(mp_sub(result.data_, mod.data_, result.data_),
                   "BigInt::NormalizeMod");
  }
}

}  // namespace cheddar