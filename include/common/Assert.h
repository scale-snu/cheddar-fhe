#pragma once

#include <iostream>
#include <string>

namespace cheddar {

// Exit if fails
inline void AssertTrue(bool expr, const std::string &msg) {
  if (!expr) {
    std::cerr << "ERROR: " << msg << std::endl << std::flush;
    std::exit(EXIT_FAILURE);
  }
}
inline void AssertFalse(bool expr, const std::string &msg) {
  AssertTrue(!expr, msg);
}
inline void Fail(const std::string &msg) {
  std::cerr << "ERROR: " << msg << std::endl << std::flush;
  std::exit(EXIT_FAILURE);
}

// Only produce warnings
inline void CheckTrue(bool expr, const std::string &msg) {
  if (!expr) {
    std::clog << "WARN: " << msg << std::endl;
  }
}
inline void CheckFalse(bool expr, const std::string &msg) {
  CheckTrue(!expr, msg);
}
inline void Warn(const std::string &msg) {
  std::clog << "WARN: " << msg << std::endl;
}

template <typename Container1, typename Container2>
inline void AssertSameNP(const Container1 &a, const Container2 &b) {
  AssertTrue(a.GetNP() == b.GetNP(), "Number of primes differ");
}

template <typename Container1, typename Container2, typename... Args>
inline void AssertSameNP(const Container1 &a, const Container2 &b,
                         Args... args) {
  AssertSameNP(a, b);
  AssertSameNP(b, args...);
}

}  // namespace cheddar
