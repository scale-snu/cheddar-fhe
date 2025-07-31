#pragma once

#include "Export.h"
#include "core/DeviceVector.h"
#include "core/NPInfo.h"

namespace cheddar {

/**
 * @brief An abstract base class for all data (Constant, Ciphertext, Plaintext,
 * EvaluationKey).
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Container {
 protected:
  static inline int degree_ = 0;
  double scale_ = 1.0;
  NPInfo num_primes_;

 public:
  explicit Container(const NPInfo &num_primes = NPInfo{});

  // movable, but not copyable
  Container(Container &&) = default;
  Container &operator=(Container &&) = default;

  virtual ~Container() = default;

  /**
   * @brief A pure virtual function to get the NPInfo of the container.
   *
   * @return NPInfo the num primes info
   */
  virtual NPInfo GetNP() const = 0;

  /**
   * @brief A pure virtual funtion to modify the NPInfo of the container.
   *
   * @param num_primes the new NPInfo
   */
  virtual void ModifyNP(const NPInfo &num_primes) = 0;

  /**
   * @brief Getter for the scale of the container.
   *
   * @return double scale
   */
  double GetScale() const;

  /**
   * @brief Setter for the scale of the container.
   *
   * @param scale the new scale
   */
  void SetScale(double scale);

  /**
   * @brief Static function used by Context
   *
   * @param degree the polynomial degree
   */
  static void SetDegree(int degree);
};

/**
 * @brief A constant represented in RNS form.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Constant : public Container<word> {
 private:
  using Base = Container<word>;

 public:
  /**
   * @brief Constructor for Constant.
   *
   * @param num_primes the NPInfo for the Constant
   */
  explicit Constant(const NPInfo &num_primes = NPInfo{});

  // movable, but not copyable
  Constant(Constant &&) = default;
  Constant &operator=(Constant &&) = default;

  virtual ~Constant() = default;

  // member data structures (public)
  DeviceVector<word> cx_;

  /**
   * @brief Get the NPInfo of the Constant.
   *
   * @return NPInfo the num primes info
   */
  NPInfo GetNP() const override;

  /**
   * @brief Modify the NPInfo of the Constant. The data in memory will be
   * resized accordingly.
   *
   * @param num_primes the new NPInfo
   */
  void ModifyNP(const NPInfo &num_primes) override;

  /**
   * @brief Get the view of the Constant data.
   *
   * @param np_front_ignore the number of primes to disregard from the front
   * @return DvView<word> the view of the Constant data
   */
  DvView<word> View(int np_front_ignore = 0);

  /**
   * @brief Get the read-only view of the Constant data.
   *
   * @param np_front_ignore the number of primes to disregard from the front
   * @return DvConstView<word> the read-only view of the Constant data
   */
  DvConstView<word> ConstView(int np_front_ignore = 0) const;
};

/**
 * @brief A ciphertext represented in RNS form. It contains two or three
 * polynomials.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Ciphertext : public Container<word> {
 private:
  using Base = Container<word>;
  int num_slots_ = Base::degree_ / 2;

 public:
  /**
   * @brief Construct a new Ciphertext object.
   *
   * @param num_primes the NPInfo for the Ciphertext
   * @param has_rx whether the ciphertext has the third polynomial rx (default:
   * false)
   */
  explicit Ciphertext(const NPInfo &num_primes = NPInfo{}, bool has_rx = false);

  // movable, but not copyable
  Ciphertext(Ciphertext &&) = default;
  Ciphertext &operator=(Ciphertext &&) = default;

  virtual ~Ciphertext() = default;

  // member variables (public)
  DeviceVector<word> bx_;
  DeviceVector<word> ax_;
  DeviceVector<word> rx_;

  /**
   * @brief Check whether the ciphertext has the third polynomial rx.
   *
   * @return true if the ciphertext has rx
   * @return false if the ciphertext does not have rx
   */
  bool HasRx() const;

  /**
   * @brief Allocate memory for the third polynomial rx.
   *
   */
  void PrepareRx();

  /**
   * @brief Remove the third polynomial rx from the ciphertext.
   *
   */
  void RemoveRx();

  /**
   * @brief Get the NPInfo of the Ciphertext.
   *
   * @return NPInfo the num primes info
   */
  NPInfo GetNP() const override;

  /**
   * @brief Modify the NPInfo of the Constant. The data in memory will be
   * resized accordingly. rx will not be resized if its prior size is 0.
   *
   * @param num_primes the new NPInfo
   */
  void ModifyNP(const NPInfo &num_primes) override;

  /**
   * @brief Get the number of slots in the ciphertext.
   *
   * @return int the number of slots
   */
  int GetNumSlots() const;

  /**
   * @brief Set the number of slots in the ciphertext.
   *
   * @param num_slots the new number of slots
   */
  void SetNumSlots(int num_slots);

  // view functions (implementation details)
  DvView<word> BxView(int np_front_ignore = 0);
  DvConstView<word> BxConstView(int np_front_ignore = 0) const;
  DvView<word> AxView(int np_front_ignore = 0);
  DvConstView<word> AxConstView(int np_front_ignore = 0) const;
  DvView<word> RxView(int np_front_ignore = 0);
  DvConstView<word> RxConstView(int np_front_ignore = 0) const;
  std::vector<DvView<word>> ViewVector(int np_front_ignore = 0,
                                       bool ignore_rx = false);
  std::vector<DvConstView<word>> ConstViewVector(int np_front_ignore = 0,
                                                 bool ignore_rx = false) const;
};

/**
 * @brief A plaintext represented in RNS form. It contains one polynomial.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Plaintext : public Container<word> {
 private:
  using Base = Container<word>;
  int num_slots_ = Base::degree_ / 2;

 public:
  /**
   * @brief Construct a new Plaintext object.
   *
   * @param num_primes the NPInfo for the Plaintext
   */
  explicit Plaintext(const NPInfo &num_primes = NPInfo{});

  // movable, but not copyable
  Plaintext(Plaintext &&) = default;
  Plaintext &operator=(Plaintext &&) = default;

  virtual ~Plaintext() = default;

  // member variables (public)
  DeviceVector<word> mx_;

  /**
   * @brief Get the NPInfo of the plaintext.
   *
   * @return NPInfo the num primes info
   */
  NPInfo GetNP() const override;

  /**
   * @brief Modify the NPInfo of the Plaintext. The data in memory will be
   * resized accordingly.
   *
   * @param num_primes the new NPInfo
   */
  void ModifyNP(const NPInfo &num_primes) override;

  /**
   * @brief Get the number of slots in the plaintext.
   *
   * @return int the number of slots
   */
  int GetNumSlots() const;

  /**
   * @brief Set the number of slots in the plaintext.
   *
   * @param num_slots the new number of slots
   */
  void SetNumSlots(int num_slots);

  // view functions (implementation details)
  DvView<word> View(int np_front_ignore = 0);
  DvConstView<word> ConstView(int np_front_ignore = 0) const;
};

/**
 * @brief An evaluation key represented in RNS form. It contains two length-beta
 * vectors of polynomials.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT EvaluationKey : public Container<word> {
 private:
  using Base = Container<word>;

  // scales do not matter in evaluation keys
  using Base::GetScale;
  using Base::SetScale;

 public:
  /**
   * @brief Construct a new EvaluationKey object.
   *
   * @param num_primes the NPInfo for the EvaluationKey
   * @param beta the number of polynomials in the evaluation key
   */
  explicit EvaluationKey(const NPInfo &num_primes = NPInfo{}, int beta = 0);

  EvaluationKey(EvaluationKey &&) = default;
  EvaluationKey &operator=(EvaluationKey &&) = default;

  virtual ~EvaluationKey() = default;

  std::vector<DeviceVector<word>> bx_;
  std::vector<DeviceVector<word>> ax_;

  /**
   * @brief Get beta, the number of polynomials in each vector of the evaluation
   * key.
   *
   * @return int beta
   */
  int GetBeta() const;

  /**
   * @brief Get the NPInfo of the EvaluationKey.
   *
   * @return NPInfo the num primes info
   */
  NPInfo GetNP() const override;

  /**
   * @brief Modify the NPInfo of the EvaluationKey. The data in memory will be
   * resized accordingly.
   *
   * @param num_primes the new NPInfo
   */
  void ModifyNP(const NPInfo &num_primes) override;

  // view functions (implementation details)
  DvView<word> BxView(int index, int np_front_ignore = 0);
  DvConstView<word> BxConstView(int index, int np_front_ignore = 0) const;
  DvView<word> AxView(int index, int np_front_ignore = 0);
  DvConstView<word> AxConstView(int index, int np_front_ignore = 0) const;
  std::vector<DvView<word>> ViewVector(int index, int np_front_ignore = 0);
  std::vector<DvConstView<word>> ConstViewVector(int index,
                                                 int np_front_ignore = 0) const;
};

}  // namespace cheddar