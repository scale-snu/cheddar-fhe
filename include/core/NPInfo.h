#pragma once

#include "Export.h"

namespace cheddar {

struct API_EXPORT NPInfo {
  int num_main_ = 0;
  int num_ter_ = 0;
  int num_aux_ = 0;

  /**
   * @brief Get the number of Q primes
   *
   * @return int num_q = num_main + num_ter
   */
  int GetNumQ() const;

  /**
   * @brief Get the number of total primes
   *
   * @return int num_total = num_main + num_ter + num_aux
   */
  int GetNumTotal() const;

  /**
   * @brief Construct a new NPInfo object
   *
   * @param num_main number of main primes
   * @param num_ter number of terminal primes
   * @param num_aux number of auxiliary primes
   */
  explicit NPInfo(int num_main = 0, int num_ter = 0, int num_aux = 0);

  // custom copy constructor and assignment operator
  NPInfo(const NPInfo &other);
  NPInfo &operator=(const NPInfo &other);

  // custom comparison operators
  bool operator==(const NPInfo &other) const;
  bool IsSubsetOf(const NPInfo &other) const;
  bool IsSupersetOf(const NPInfo &other) const;
};

}  // namespace cheddar
