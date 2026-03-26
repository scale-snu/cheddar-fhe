#include "core/Encode.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"
#include "core/BigInt.h"

namespace cheddar {

template <typename word>
Encoder<word>::Encoder(const Parameter<word> &param,
                       const NTTHandler<word> &ntt_handler)
    : param_{param},
      ntt_handler_{ntt_handler},
      M_{param.degree_ * 2},
      twiddle_factors_(param.degree_ * 2) {
  for (int i = 0; i < param.degree_ * 2; i++) {
    // e^(2*pi*sqrt(-1)*i/M)
    twiddle_factors_[i] = std::polar(1.0, 2.0 * M_PI * i / M_);
  }
}

template <typename word>
void Encoder<word>::Encode(Plaintext<word> &ptxt, int level, double scale,
                           const std::vector<Complex> &message,
                           int num_aux /*= 0*/) const {
  int msg_length = message.size();
  int num_slots = 1 << Log2Ceil<int>(msg_length);
  std::vector<Complex> padded_msg(num_slots);
  std::copy(message.begin(), message.end(), padded_msg.begin());
  EncodeWorker(ptxt, level, scale, padded_msg, num_aux);
}

template <typename word>
void Encoder<word>::EncodeWorker(Plaintext<word> &ptxt, int level, double scale,
                                 std::vector<Complex> &message,
                                 int num_aux /*= 0*/) const {
  SpecialIFFT(message);
  ComplexVectorToPlaintext(ptxt, level, scale, message, num_aux);
  NPInfo np = param_.LevelToNP(level, num_aux);
  auto mx_temp = ptxt.View();
  ntt_handler_.NTT(mx_temp, np, ptxt.ConstView(), true);
}

template <typename word>
void Encoder<word>::Decode(std::vector<Complex> &message,
                           const Plaintext<word> &ptxt) const {
  // Message padding
  Plaintext<word> tmp;
  NPInfo np = ptxt.GetNP();
  AssertTrue(np.num_aux_ == 0, "Decode: Aux primes not supported");
  auto primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  word degree = param_.degree_;

  // Create iCRT constants
  HostVector<word> icrt1(num_total_primes);
  for (int i = 0; i < num_total_primes; i++) {
    word mod_prime = primes[i];
    word prod = 1;
    for (int j = 0; j < num_total_primes; j++) {
      if (i == j) continue;
      prod = primeutil::MultMod(prod, primes[j], mod_prime);
    }
    prod = primeutil::MultMod(prod, degree, mod_prime);
    // deliberately not using the Montgomery form here
    icrt1[i] = primeutil::InvMod(prod, mod_prime);
  }

  DeviceVector<word> icrt1_dv(num_total_primes);
  CopyHostToDevice(icrt1_dv, icrt1);

  tmp.ModifyNP(np);
  tmp.SetNumSlots(ptxt.GetNumSlots());
  tmp.SetScale(ptxt.GetScale());

  auto mx_temp = tmp.View();
  ntt_handler_.INTTAndMultConst(mx_temp, np, ptxt.ConstView(),
                                icrt1_dv.ConstView());
  PlaintextToComplexVector(message, tmp);
  SpecialFFT(message);
}

template <typename word>
void Encoder<word>::SpecialIFFT(std::vector<Complex> &data) const {
  int num_slots = data.size();
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "Power of 2 num slots only");

  for (int stride = num_slots / 2; stride >= 1; stride /= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; j++) {
        int twiddle_index = (st8 - (param_.GetGaloisFactor(j) % st8)) * gap;
        auto x = data[i + j] + data[i + j + stride];
        auto y = data[i + j] - data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x;
        data[i + j + stride] = y;
      }
    }
  }
  BitReverseVector(data);
  for (int i = 0; i < num_slots; ++i) {
    data[i] /= num_slots;
  }
}

template <typename word>
void Encoder<word>::SpecialFFT(std::vector<Complex> &data) const {
  int num_slots = data.size();
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "Power of 2 num slots only");
  BitReverseVector(data);

  for (int stride = 1; stride < num_slots; stride *= 2) {
    int stride_group_size = stride * 2;
    int st8 = stride << 3;
    int gap = M_ / st8;
    for (int i = 0; i < num_slots; i += stride_group_size) {
      for (int j = 0; j < stride; j++) {
        int twiddle_index = (param_.GetGaloisFactor(j) % st8) * gap;
        auto x = data[i + j];
        auto y = data[i + j + stride];
        y *= twiddle_factors_[twiddle_index];
        data[i + j] = x + y;
        data[i + j + stride] = x - y;
      }
    }
  }
}

template <typename word>
void Encoder<word>::ComplexVectorToPlaintext(Plaintext<word> &ptxt, int level,
                                             double scale,
                                             const std::vector<Complex> &data,
                                             int num_aux /*= 0*/) const {
  int num_slots = data.size();
  int degree = param_.degree_;
  int half_degree = degree / 2;
  int gap = half_degree / num_slots;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "ComplexVectorToPlaintext: Power of 2 num slots only");
  AssertTrue(num_slots <= half_degree,
             "ComplexVectorToPlaintext: Too many slots");

  NPInfo np = param_.LevelToNP(level, num_aux);
  auto primes = param_.GetPrimeVector(np);
  int num_total_primes = np.GetNumTotal();
  HostVector<word> mx(num_total_primes * degree, 0);

  std::vector<BigInt> big_primes;

  for (int j = 0; j < num_total_primes; j++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[j]));
  }

  for (int i = 0; i < num_slots; i++) {
    Complex value = data[i] * scale;
    double real = value.real();
    double imag = value.imag();

    BigInt big_real(real);
    BigInt big_imag(imag);

    int real_degree_index = i * gap;
    int imag_degree_index = real_degree_index + half_degree;

    BigInt real_tmp(0.0);
    BigInt imag_tmp(0.0);

    for (int j = 0; j < num_total_primes; j++) {
      const BigInt &prime = big_primes[j];
      BigInt::Mod(real_tmp, big_real, prime);
      BigInt::Mod(imag_tmp, big_imag, prime);
      word real_mod = real_tmp.GetUnsigned();
      word imag_mod = imag_tmp.GetUnsigned();
      mx[real_degree_index + j * param_.degree_] = real_mod;
      mx[imag_degree_index + j * param_.degree_] = imag_mod;
    }
  }

  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(num_slots);
  ptxt.SetScale(scale);
  CopyHostToDevice(ptxt.mx_, mx);
}

template <typename word>
void Encoder<word>::PlaintextToComplexVector(
    std::vector<Complex> &data, const Plaintext<word> &ptxt) const {
  int num_slots = ptxt.GetNumSlots();
  double scale = ptxt.GetScale();
  NPInfo np = ptxt.GetNP();
  int num_total_primes = np.GetNumTotal();
  auto primes = param_.GetPrimeVector(np);
  int degree = param_.degree_;
  int half_degree = degree / 2;
  int gap = half_degree / num_slots;

  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "ComplexVectorToPlaintext: Power of 2 num slots only");
  AssertTrue(num_slots <= half_degree,
             "PlaintextToComplexVector: Too many slots");

  data.resize(num_slots);

  HostVector<word> intt_res;
  CopyDeviceToHost(intt_res, ptxt.mx_);

  std::vector<BigInt> big_primes;
  BigInt prime_prod(static_cast<uint64_t>(1));
  for (int i = 0; i < num_total_primes; i++) {
    big_primes.emplace_back(static_cast<uint64_t>(primes[i]));
    BigInt::Mult(prime_prod, prime_prod, big_primes[i]);
  }
  BigInt half_prime_prod(static_cast<uint64_t>(0));
  BigInt::Div2(half_prime_prod, prime_prod);

  BigInt tmp(static_cast<uint64_t>(0));
  for (int i = 0; i < num_slots; i++) {
    BigInt real_icrt(static_cast<uint64_t>(0));
    BigInt imag_icrt(static_cast<uint64_t>(0));
    for (int j = 0; j < num_total_primes; j++) {
      BigInt real_value(static_cast<uint64_t>(intt_res[j * degree + i * gap]));
      BigInt imag_value(
          static_cast<uint64_t>(intt_res[j * degree + i * gap + half_degree]));
      BigInt prod(static_cast<uint64_t>(1));
      for (int k = 0; k < num_total_primes; k++) {
        if (j == k) continue;
        BigInt::Mult(prod, prod, big_primes[k]);
      }
      BigInt::Mult(tmp, real_value, prod);
      BigInt::Add(real_icrt, real_icrt, tmp);
      BigInt::Mult(tmp, imag_value, prod);
      BigInt::Add(imag_icrt, imag_icrt, tmp);
    }
    BigInt::NormalizeMod(real_icrt, real_icrt, prime_prod, half_prime_prod);
    BigInt::NormalizeMod(imag_icrt, imag_icrt, prime_prod, half_prime_prod);

    double real = real_icrt.GetDouble() / scale;
    double imag = imag_icrt.GetDouble() / scale;
    data[i] = Complex(real, imag);
  }
}

template <typename word>
void Encoder<word>::EncodeConstant(Constant<word> &constant, int level,
                                   double scale, double number,
                                   int num_aux /*= 0*/) const {
  BigInt big_int(number * scale);
  NPInfo np = param_.LevelToNP(level, num_aux);
  int num_total_primes = np.GetNumTotal();

  auto primes = param_.GetPrimeVector(np);

  HostVector<word> cx(num_total_primes);

  for (int i = 0; i < num_total_primes; i++) {
    BigInt prime(static_cast<uint64_t>(primes[i]));

    BigInt big_mod(static_cast<uint64_t>(0));
    BigInt::Mod(big_mod, big_int, prime);

    word result = big_mod.GetUnsigned();
    cx[i] = primeutil::ToMontgomery(result, primes[i]);
  }

  constant.ModifyNP(np);
  constant.SetScale(scale);
  CopyHostToDevice(constant.cx_, cx);
}

template <typename word>
Complex Encoder<word>::GetTwiddleFactor(int index) const {
  return twiddle_factors_[index];
}

template class Encoder<uint32_t>;
template class Encoder<uint64_t>;

}  // namespace cheddar
