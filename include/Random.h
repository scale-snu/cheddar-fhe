#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "Export.h"
#include "core/Type.h"

namespace cheddar {

/**
 * @brief As this is not a cryptographically secure random number generator, it
 * should be only used for testing purposes only. Do not use this for
 * production.
 *
 */
class API_EXPORT Random {
 public:
  Random() = default;
  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;

  template <typename word>
  static void SampleUniformWord(word *ptr, int length, word range_min,
                                word range_max) {
    std::uniform_int_distribution<word> dist(range_min, range_max);
    std::generate(ptr, ptr + length, [&]() { return dist(gen_); });
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
    std::shuffle(values.begin(), values.end(), gen_);
    std::copy(values.begin(), values.begin() + length, ptr);
  }

  template <typename T>
  static void SampleUniformReal(T *ptr, int length, double range_min,
                                double range_max) {
    std::uniform_real_distribution<double> dist(range_min, range_max);
    std::generate(ptr, ptr + length,
                  [&]() { return static_cast<T>(dist(gen_)); });
  }

  template <typename T>
  static void SampleUniformComplex(T *ptr, int length, double range_min,
                                   double range_max) {
    std::uniform_real_distribution<double> dist(range_min, range_max);
    std::generate(ptr, ptr + length, [&]() {
      return static_cast<T>(Complex(dist(gen_), dist(gen_)));
    });
  }

  template <typename T>
  static void SampleDiscreteNormal(T *ptr, int length, double mean,
                                   double stddev, double max_sigma = 6) {
    std::normal_distribution<double> dist(0, stddev);
    std::generate(ptr, ptr + length, [&]() {
      double val = dist(gen_);
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
  static inline std::random_device rd_{};
  static inline std::mt19937_64 gen_{rd_()};
  // static inline std::mt19937_64 gen_{0};
};

}  // namespace cheddar