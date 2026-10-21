#pragma once

#include <errno.h>
#include <pthread.h>
#include <sys/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

#include "core/Type.h"

namespace cheddar {

/**
 * @brief ChaCha20 (RFC 8439) keystream as a C++ UniformRandomBitGenerator.
 * The 256-bit key and the 96-bit nonce are drawn from getrandom(2) (the kernel
 * CSPRNG); the 32-bit block counter starts at 0. After 2^32 blocks (256 GiB of
 * output) the generator is rekeyed from the OS, so the counter never wraps.
 * 64-bit outputs are two consecutive little-endian keystream words (low word
 * first).
 */
class ChaCha20Generator {
 public:
  using result_type = uint64_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  ChaCha20Generator() { Reseed(); }
  ChaCha20Generator(const ChaCha20Generator &) = delete;
  ChaCha20Generator &operator=(const ChaCha20Generator &) = delete;
  ~ChaCha20Generator() { Wipe(); }

  /// Deterministic initialization; for known-answer tests only.
  void SetKeyForTesting(const uint8_t key[32], const uint8_t nonce[12],
                        uint32_t counter) {
    Init(key, nonce, counter);
  }

  /// Discard the current state and draw a fresh key and nonce from the OS.
  /// Must be called in a forked child (the state is otherwise duplicated).
  void Reseed() {
    uint8_t seed[44];
    GetRandomBytes(seed, sizeof(seed));
    Init(seed, seed + 32, 0);
    volatile uint8_t *v = seed;
    for (size_t i = 0; i < sizeof(seed); i++) v[i] = 0;
  }

  result_type operator()() {
    if (pos_ > 14) Refill();
    result_type lo = block_[pos_];
    result_type hi = block_[pos_ + 1];
    pos_ += 2;
    return lo | (hi << 32);
  }

  /// RFC 8439 Section 2.3 block function, output as 16 words (before
  /// serialization). Exposed for known-answer tests.
  static void Block(const uint32_t in[16], uint32_t out[16]) {
    uint32_t x[16];
    std::memcpy(x, in, sizeof(x));
    for (int i = 0; i < 10; i++) {
      QuarterRound(x[0], x[4], x[8], x[12]);
      QuarterRound(x[1], x[5], x[9], x[13]);
      QuarterRound(x[2], x[6], x[10], x[14]);
      QuarterRound(x[3], x[7], x[11], x[15]);
      QuarterRound(x[0], x[5], x[10], x[15]);
      QuarterRound(x[1], x[6], x[11], x[12]);
      QuarterRound(x[2], x[7], x[8], x[13]);
      QuarterRound(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; i++) out[i] = x[i] + in[i];
  }

 private:
  static uint32_t Rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

  static void QuarterRound(uint32_t &a, uint32_t &b, uint32_t &c,
                           uint32_t &d) {
    a += b; d ^= a; d = Rotl(d, 16);
    c += d; b ^= c; b = Rotl(b, 12);
    a += b; d ^= a; d = Rotl(d, 8);
    c += d; b ^= c; b = Rotl(b, 7);
  }

  static uint32_t LoadLE32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
  }

  // Fill buf with len bytes from getrandom(2). There is no weaker fallback: if
  // the kernel cannot provide entropy we abort instead of generating keys.
  static void GetRandomBytes(uint8_t *buf, size_t len) {
    size_t got = 0;
    while (got < len) {
      ssize_t r = getrandom(buf + got, len - got, 0);
      if (r < 0) {
        if (errno == EINTR) continue;
        std::perror("getrandom");
        std::cerr << "Random: cannot obtain OS entropy; aborting" << std::endl;
        std::abort();
      }
      got += static_cast<size_t>(r);
    }
  }

  void Init(const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
    state_[0] = 0x61707865;  // "expand 32-byte k"
    state_[1] = 0x3320646e;
    state_[2] = 0x79622d32;
    state_[3] = 0x6b206574;
    for (int i = 0; i < 8; i++) state_[4 + i] = LoadLE32(key + 4 * i);
    state_[12] = counter;
    for (int i = 0; i < 3; i++) state_[13 + i] = LoadLE32(nonce + 4 * i);
    pos_ = 16;
    exhausted_ = false;
  }

  void Refill() {
    if (exhausted_) Reseed();
    Block(state_, block_);
    pos_ = 0;
    if (++state_[12] == 0) exhausted_ = true;  // rekey before reuse
  }

  void Wipe() {
    volatile uint8_t *s = reinterpret_cast<uint8_t *>(state_);
    for (size_t i = 0; i < sizeof(state_); i++) s[i] = 0;
    volatile uint8_t *b = reinterpret_cast<uint8_t *>(block_);
    for (size_t i = 0; i < sizeof(block_); i++) b[i] = 0;
  }

  uint32_t state_[16];
  uint32_t block_[16];
  int pos_;
  bool exhausted_;
};

/**
 * @brief Random sampling front end for key generation, errors and test
 * messages. The bit source is a ChaCha20 generator seeded from the OS
 * (getrandom(2)); the sampling algorithms are unchanged. All methods are
 * thread-safe (one process-wide generator protected by a mutex).
 */
class Random {
 public:
  Random() = default;
  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;

  template <typename word>
  static void SampleUniformWord(word *ptr, int length, word range_min,
                                word range_max) {
    std::uniform_int_distribution<word> dist(range_min, range_max);
    std::lock_guard<std::mutex> lock(Shared().mutex);
    auto &gen = Shared().gen;
    std::generate(ptr, ptr + length, [&]() { return dist(gen); });
  }

  template <typename T>
  static void SampleWithoutReplacement(T *ptr, int length, int range_min,
                                       int range_max) {
    if (range_max - range_min + 1 < length) {
      std::cerr << "Invalid length";
      exit(EXIT_FAILURE);
    }
    std::vector<int> values(range_max - range_min + 1);
    std::iota(values.begin(), values.end(), range_min);
    {
      std::lock_guard<std::mutex> lock(Shared().mutex);
      std::shuffle(values.begin(), values.end(), Shared().gen);
    }
    std::copy(values.begin(), values.begin() + length, ptr);
  }

  template <typename T>
  static void SampleUniformReal(T *ptr, int length, double range_min,
                                double range_max) {
    std::uniform_real_distribution<double> dist(range_min, range_max);
    std::lock_guard<std::mutex> lock(Shared().mutex);
    auto &gen = Shared().gen;
    std::generate(ptr, ptr + length,
                  [&]() { return static_cast<T>(dist(gen)); });
  }

  template <typename T>
  static void SampleUniformComplex(T *ptr, int length, double range_min,
                                   double range_max) {
    std::uniform_real_distribution<double> dist(range_min, range_max);
    std::lock_guard<std::mutex> lock(Shared().mutex);
    auto &gen = Shared().gen;
    std::generate(ptr, ptr + length, [&]() {
      double re = dist(gen);  // fixed evaluation order
      double im = dist(gen);
      return static_cast<T>(Complex(re, im));
    });
  }

  template <typename T>
  static void SampleDiscreteNormal(T *ptr, int length, double mean,
                                   double stddev, double max_sigma = 6) {
    std::normal_distribution<double> dist(0, stddev);
    std::lock_guard<std::mutex> lock(Shared().mutex);
    auto &gen = Shared().gen;
    std::generate(ptr, ptr + length, [&]() {
      double val = dist(gen);
      if (val > max_sigma * stddev) {
        val = max_sigma * stddev;
      } else if (val < -max_sigma * stddev) {
        val = -max_sigma * stddev;
      }
      val += mean;
      return static_cast<T>(std::llround(val));
    });
  }

 private:
  struct SharedState {
    ChaCha20Generator gen;
    std::mutex mutex;
    SharedState() {
      // A forked child must not continue the parent's keystream.
      int error = pthread_atfork([] { Shared().mutex.lock(); },
                                 [] { Shared().mutex.unlock(); },
                                 [] {
                                   Shared().gen.Reseed();
                                   Shared().mutex.unlock();
                                 });
      if (error != 0) {
        std::cerr << "Random: cannot register fork handlers: "
                  << std::strerror(error) << "; aborting" << std::endl;
        std::abort();
      }
    }
  };
  static SharedState &Shared() {
    static SharedState state;  // thread-safe initialization
    return state;
  }
};

}  // namespace cheddar
