#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <set>
#include <thread>
#include <vector>

#include "Random.h"

using namespace cheddar;

namespace {

void SetTestKey(ChaCha20Generator &gen, uint32_t counter,
                const uint8_t nonce[12], bool zero_key = false) {
  uint8_t key[32];
  for (int i = 0; i < 32; i++) key[i] = zero_key ? 0 : i;
  gen.SetKeyForTesting(key, nonce, counter);
}

uint64_t Word64(uint32_t lo, uint32_t hi) {
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

}  // namespace

// RFC 8439 Section 2.3.2: block function test vector.
TEST(ChaCha20Test, Rfc8439BlockFunction) {
  const uint8_t nonce[12] = {0, 0, 0, 9, 0, 0, 0, 0x4a, 0, 0, 0, 0};
  uint32_t in[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                     0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                     0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
                     0x00000001, 0x09000000, 0x4a000000, 0x00000000};
  const uint32_t expected[16] = {
      0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033,
      0x9aaa2204, 0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
      0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2};
  uint32_t out[16];
  ChaCha20Generator::Block(in, out);
  for (int i = 0; i < 16; i++) EXPECT_EQ(out[i], expected[i]) << "word " << i;

  // Same vector through the generator (key setup, nonce load, counter = 1):
  // output words are the block words in order, low word first.
  ChaCha20Generator gen;
  SetTestKey(gen, 1, nonce);
  for (int i = 0; i < 16; i += 2)
    EXPECT_EQ(gen(), Word64(expected[i], expected[i + 1])) << "word " << i;
}

// RFC 8439 Appendix A.1, test vectors #1 and #2 (zero key and nonce).
TEST(ChaCha20Test, Rfc8439AppendixA1) {
  const uint8_t nonce[12] = {};
  ChaCha20Generator gen;
  SetTestKey(gen, 0, nonce, true);
  // keystream bytes 76 b8 e0 ad a0 f1 3d 90 40 5d 6a e5 53 86 bd 28
  EXPECT_EQ(gen(), Word64(0xade0b876, 0x903df1a0));
  EXPECT_EQ(gen(), Word64(0xe56a5d40, 0x28bd8653));
  // block counter 1 starts with 9f 07 e7 be 55 51 38 7a 98 ba 97 7c 73 2d 08 0d
  SetTestKey(gen, 1, nonce, true);
  EXPECT_EQ(gen(), Word64(0xbee7079f, 0x7a385155));
  EXPECT_EQ(gen(), Word64(0x7c97ba98, 0x0d082d73));
  // counter carries over from block to block
  ChaCha20Generator a, b;
  SetTestKey(a, 0, nonce, true);
  SetTestKey(b, 1, nonce, true);
  for (int i = 0; i < 8; i++) a();
  for (int i = 0; i < 20; i++) EXPECT_EQ(a(), b());
}

TEST(ChaCha20Test, FreshInstancesDiffer) {
  ChaCha20Generator a, b;
  EXPECT_NE(a(), b());
  EXPECT_NE(a(), b());
}

TEST(RandomTest, UniformWordRangeAndChiSquare) {
  const uint32_t q = 1000003;  // not a power of two
  const int n = 4000000, bins = 100;
  std::vector<uint32_t> v(n);
  Random::SampleUniformWord<uint32_t>(v.data(), n, 5, q - 1);
  std::vector<double> count(bins, 0);
  for (auto x : v) {
    ASSERT_GE(x, 5u);
    ASSERT_LT(x, q);
    count[static_cast<uint64_t>(x - 5) * bins / (q - 5)]++;
  }
  double chi = 0, e = static_cast<double>(n) / bins;
  for (double c : count) chi += (c - e) * (c - e) / e;
  // 99 dof: mean 99, sd ~14; 200 is far beyond any plausible fluctuation
  EXPECT_LT(chi, 200);

  std::vector<uint64_t> w(1000);
  Random::SampleUniformWord<uint64_t>(w.data(), 1000, 0, UINT64_MAX);
  Random::SampleUniformWord<uint64_t>(w.data(), 1000, 7, 7);
  for (auto x : w) EXPECT_EQ(x, 7u);
}

TEST(RandomTest, TernaryBitsBalanced) {
  const int n = 1 << 20;
  std::vector<uint32_t> v(n);
  Random::SampleUniformWord<uint32_t>(v.data(), n, 0, 1);
  double ones = 0;
  for (auto x : v) {
    ASSERT_LE(x, 1u);
    ones += x;
  }
  EXPECT_NEAR(ones / n, 0.5, 6 * 0.5 / std::sqrt(n));
}

TEST(RandomTest, WithoutReplacementIsDistinctAndInRange) {
  std::vector<int> v(32);
  for (int rep = 0; rep < 50; rep++) {
    Random::SampleWithoutReplacement(v.data(), 32, 10, 1000);
    std::set<int> s(v.begin(), v.end());
    EXPECT_EQ(s.size(), 32u);
    for (int x : v) {
      EXPECT_GE(x, 10);
      EXPECT_LE(x, 1000);
    }
  }
  // each element of a small range appears first about equally often
  std::vector<int> first(8, 0);
  int one;
  for (int rep = 0; rep < 16000; rep++) {
    Random::SampleWithoutReplacement(&one, 1, 0, 7);
    first[one]++;
  }
  for (int c : first) EXPECT_NEAR(c, 2000, 6 * std::sqrt(2000.0 * 7 / 8));
}

TEST(RandomTest, DiscreteNormalMomentsAndClamp) {
  const int n = 2000000;
  const double sigma = 3.2;
  std::vector<int> v(n);
  Random::SampleDiscreteNormal(v.data(), n, 0, sigma);
  double mean = 0, var = 0;
  int mx = 0;
  for (int x : v) {
    mean += x;
    mx = std::max(mx, std::abs(x));
  }
  mean /= n;
  for (int x : v) var += (x - mean) * (x - mean);
  var /= n;
  EXPECT_NEAR(mean, 0, 6 * sigma / std::sqrt(n));
  const double expected_var = sigma * sigma + 1.0 / 12;  // rounding adds 1/12
  // Approximate standard error of the variance for a Gaussian sample.
  const double var_standard_error = expected_var * std::sqrt(2.0 / (n - 1));
  EXPECT_NEAR(var, expected_var, 6 * var_standard_error);
  EXPECT_LE(mx, static_cast<int>(std::llround(6 * sigma)));
}

TEST(RandomTest, UniformRealAndComplex) {
  const int n = 1000000;
  std::vector<double> v(n);
  Random::SampleUniformReal(v.data(), n, -2.0, 2.0);
  double mean = 0, var = 0;
  for (double x : v) {
    ASSERT_GE(x, -2.0);
    ASSERT_LE(x, 2.0);
    mean += x;
  }
  mean /= n;
  for (double x : v) var += (x - mean) * (x - mean);
  var /= n;
  EXPECT_NEAR(mean, 0, 0.01);
  EXPECT_NEAR(var, 16.0 / 12, 0.01);
  std::vector<Complex> c(n);
  Random::SampleUniformComplex(c.data(), n, -1.0, 1.0);
  double re = 0, im = 0, cross = 0;
  for (auto &x : c) {
    re += x.real();
    im += x.imag();
    cross += x.real() * x.imag();
  }
  EXPECT_NEAR(re / n, 0, 0.005);
  EXPECT_NEAR(im / n, 0, 0.005);
  EXPECT_NEAR(cross / n, 0, 0.005);  // components uncorrelated
}

TEST(RandomTest, ThreadSafeNoDuplicatedBlocks) {
  const int kThreads = 8, kLen = 1 << 16;
  std::vector<std::vector<uint64_t>> out(kThreads,
                                         std::vector<uint64_t>(kLen));
  std::vector<std::thread> th;
  for (int t = 0; t < kThreads; t++)
    th.emplace_back([&, t] {
      for (int rep = 0; rep < 8; rep++)
        Random::SampleUniformWord<uint64_t>(out[t].data(), kLen, 0,
                                            UINT64_MAX);
    });
  for (auto &x : th) x.join();
  std::set<uint64_t> all;
  for (auto &o : out) all.insert(o.begin(), o.end());
  // 64-bit values: any repeat would indicate shared/duplicated keystream
  EXPECT_EQ(all.size(), static_cast<size_t>(kThreads) * kLen);
}

TEST(RandomTest, ForkedChildDoesNotReplayParentStream) {
  uint64_t warm;
  Random::SampleUniformWord<uint64_t>(&warm, 1, 0, UINT64_MAX);
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  uint64_t v[4];
  Random::SampleUniformWord<uint64_t>(v, 4, 0, UINT64_MAX);
  if (pid == 0) {
    ssize_t r = write(fds[1], v, sizeof(v));
    _exit(r == sizeof(v) ? 0 : 1);
  }
  uint64_t cv[4];
  ASSERT_EQ(read(fds[0], cv, sizeof(cv)), static_cast<ssize_t>(sizeof(cv)));
  int status;
  waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  for (int i = 0; i < 4; i++) EXPECT_NE(v[i], cv[i]);
}
