#pragma once

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "UserInterface.h"

#ifdef ENABLE_EXTENSION
#include "extension/BootContext.h"
#endif

using word = uint32_t;
using namespace cheddar;

#define __ProfileStart(name, warm_up, init)                 \
  {                                                         \
    std::cout << ">>>>> " << name << " <<<<<" << std::endl; \
    cudaDeviceSynchronize();                                \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (int __w = 0; __w < warm_up + 1; __w++) {           \
      init;                                                 \
      if (__w == warm_up) {                                 \
        cudaDeviceSynchronize();                            \
        start = std::chrono::high_resolution_clock::now();  \
      }

// Actual region of interest (RoI) execution goes here

#define __ProfileEnd(name)                                                  \
  cudaDeviceSynchronize();                                                  \
  }                                                                         \
  auto end = std::chrono::high_resolution_clock::now();                     \
  std::cout << "Wall clock time (+ sync overhead): "                        \
            << std::chrono::duration_cast<std::chrono::microseconds>(end -  \
                                                                     start) \
                   .count()                                                 \
            << "us" << std::endl;                                           \
  }

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "ERROR:" << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void PrintVector(const std::vector<T> &vec, int print_num = 5) {
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "[ ";
  int size = vec.size();
  if (size <= 2 * print_num) {
    for (const auto &elem : vec) {
      std::cout << std::setw(10) << elem << ", ";
    }
  } else {
    for (int i = 0; i < print_num; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
    std::cout << " ..., ";
    for (int i = size - print_num; i < size; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
  }
  std::cout << "] ( size: " << size << " )" << std::endl;
}

class Testbed : public testing::TestWithParam<const char *> {
 public:
  int log_degree_;
  double default_scale_;
  int default_encryption_level_;
  std::unique_ptr<Parameter<word>> param_ = nullptr;
  ContextPtr<word> context_ = nullptr;
  std::unique_ptr<UserInterface<word>> interface_ = nullptr;
  std::vector<word> main_primes_;
  std::vector<word> ter_primes_;
  std::vector<word> aux_primes_;
  std::vector<std::pair<int, int>> level_config_;
  std::pair<int, int> additional_base_;

 protected:
  static inline constexpr double max_error_ = 1e-3;

  void SetUp() override {
    std::string json_path(GetParam());
    std::ifstream json_file(json_path);
    Check(json_file.is_open(), "Failed to open JSON file: " + json_path);
    json json_data = json::parse(json_file);
    json_file.close();

    // Parsing...
    Check(json_data.contains("log_degree"), "Missing log_degree in JSON file");
    Check(json_data["log_degree"].is_number_integer(),
          "log_degree should be an integer");
    log_degree_ = json_data["log_degree"];

    Check(json_data.contains("log_default_scale"),
          "Missing log_default_scale in JSON file");
    Check(json_data["log_default_scale"].is_number_integer(),
          "log_default_scale should be an integer");
    int log_default_scale = json_data["log_default_scale"];
    default_scale_ = (UINT64_C(1) << log_default_scale);

    Check(json_data.contains("default_encryption_level"),
          "Missing default_encryption_level in JSON file");
    Check(json_data["default_encryption_level"].is_number_integer(),
          "default_encryption_level should be an integer");
    default_encryption_level_ = json_data["default_encryption_level"];

    main_primes_.clear();
    Check(json_data.contains("main_primes"),
          "Missing main_primes in JSON file");
    auto main_primes = json_data["main_primes"];
    Check(main_primes.is_array(), "main_primes should be an array");
    std::vector<word> main_primes_;
    for (const auto &prime : main_primes) {
      Check(prime.is_number_integer(),
            "main_primes should be an array of integers");
      main_primes_.push_back(prime);
    }

    ter_primes_.clear();
    if (json_data.contains("terminal_primes")) {
      auto ter_primes = json_data["terminal_primes"];
      Check(ter_primes.is_array(), "aux_primes should be an array");
      for (const auto &prime : ter_primes) {
        Check(prime.is_number_integer(),
              "terminal_primes should be an array of integers");
        ter_primes_.push_back(prime);
      }
    }

    aux_primes_.clear();
    Check(json_data.contains("auxiliary_primes"),
          "Missing auxiliary_primes in JSON file");
    auto aux_primes = json_data["auxiliary_primes"];
    Check(aux_primes.is_array(), "aux_primes should be an array");
    for (const auto &prime : aux_primes) {
      Check(prime.is_number_integer(),
            "auxiliary_primes should be an array of integers");
      aux_primes_.push_back(prime);
    }

    level_config_.clear();
    Check(json_data.contains("level_config"),
          "Missing level_config in JSON file");
    auto level_config = json_data["level_config"];
    Check(level_config.is_array(), "level_config should be an array");
    for (const auto &pair : level_config) {
      Check(pair.is_array() && pair.size() == 2,
            "level_config should be an array of pairs");
      level_config_.emplace_back(pair[0], pair[1]);
    }

    additional_base_ = {0, 0};
    if (json_data.contains("additional_base")) {
      auto additional_base = json_data["additional_base"];
      Check(additional_base.is_array() && additional_base.size() == 2,
            "additional_base should be an array of pairs");
      additional_base_ = {additional_base[0], additional_base[1]};
    }

    // Initialize Parameter
    param_ = std::make_unique<Parameter<word>>(
        log_degree_, default_scale_, default_encryption_level_, level_config_,
        main_primes_, aux_primes_, ter_primes_, additional_base_);

    if (json_data.contains("dense_hamming_weight")) {
      Check(json_data["dense_hamming_weight"].is_number_integer(),
            "dense_hamming_weight should be an integer");
      int dense_h = json_data["dense_hamming_weight"];
      param_->SetDenseHammingWeight(dense_h);
    }
    if (json_data.contains("sparse_hamming_weight")) {
      Check(json_data["sparse_hamming_weight"].is_number_integer(),
            "sparse_hamming_weight should be an integer");
      int sparse_h = json_data["sparse_hamming_weight"];
      param_->SetSparseHammingWeight(sparse_h);
    }

#ifdef ENABLE_EXTENSION
    bool enable_boot = false;
    if (json_data.contains("boot")) {
      Check(json_data["boot"].is_boolean(), "boot should be a boolean");
      enable_boot = json_data["boot"];
    }

    if (enable_boot) {
      std::cout << "Bootstrapping enabled" << std::endl;
      Check(json_data.contains("num_cts_levels"),
            "Missing num_cts_levels in JSON file");
      Check(json_data["num_cts_levels"].is_number_integer(),
            "num_cts_levels should be an integer");
      int num_cts_levels = json_data["num_cts_levels"];
      Check(json_data.contains("num_stc_levels"),
            "Missing num_stc_levels in JSON file");
      Check(json_data["num_stc_levels"].is_number_integer(),
            "num_stc_levels should be an integer");
      int num_stc_levels = json_data["num_stc_levels"];
      context_ = BootContext<word>::Create(
          *param_,
          BootParameter(param_->max_level_, num_cts_levels, num_stc_levels));
    } else {
      context_ = Context<word>::Create(*param_);
    }
#else
    context_ = Context<word>::Create(*param_);
#endif
    interface_ = std::make_unique<UserInterface<word>>(context_);
  }

  void TearDown() override {
    interface_.reset();
    std::cout << "Context use count (should be 1 to prevent memory leak): "
              << context_.use_count() << std::endl;
    context_.reset();
    param_.reset();
  }

 public:
  int GetDnum() const { return param_->dnum_; }
  int GetAlpha() const { return param_->alpha_; }
  int GetNumTotalLevels() const { return param_->max_level_; }

  double DetermineScale(int level) const {
    if (level <= default_encryption_level_) {
      return param_->GetScale(level);
    } else {
      // We just use rescale prime product as the scale for test purposes.
      return param_->GetRescalePrimeProd(level);
    }
  }

  void GenerateRandomMessage(std::vector<Complex> &res, int num_slots = -1,
                             double range_min = -1.0, double range_max = 1.0,
                             bool complex = true) {
    int degree = 1 << log_degree_;
    if (num_slots == -1) num_slots = degree / 2;

    res.resize(num_slots);
    std::fill(res.begin(), res.end(), Complex(0, 0));
    if (complex) {
      Random::SampleUniformComplex(res.data(), num_slots, range_min, range_max);
    } else {
      Random::SampleUniformReal(res.data(), num_slots, range_min, range_max);
    }
  }

  void EncodeConstant(Constant<word> &constant, double number, int level,
                      bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_aux = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.EncodeConstant(constant, level, scale, number, num_aux);
  }

  void Encode(Plaintext<word> &res, const std::vector<Complex> &msg, int level,
              bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_p_primes = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.Encode(res, level, scale, msg, num_p_primes);
  }

  void EncodeAndEncrypt(Ciphertext<word> &res, const std::vector<Complex> &msg,
                        int level, bool mod_up = false) const {
    Plaintext<word> ptxt;
    Encode(ptxt, msg, level, mod_up);
    interface_->Encrypt(res, ptxt);
  }

  void Decode(std::vector<Complex> &res, const Plaintext<word> &ptxt) const {
    context_->encoder_.Decode(res, ptxt);
  }

  void DecryptAndDecode(std::vector<Complex> &res,
                        const Ciphertext<word> &ctxt) const {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, ctxt);
    context_->encoder_.Decode(res, ptxt);
  }

  void CompareMessages(const std::vector<Complex> &msg1,
                       const std::vector<Complex> &msg2, bool print = true,
                       double max_error = max_error_) const {
    int degree = 1 << log_degree_;
    if (print) {
      std::cout << std::endl;
      std::cout << "expected: ";
      PrintVector(msg1);
      std::cout << "obtained: ";
      PrintVector(msg2);
    }

    ASSERT_EQ(msg1.size(), msg2.size()) << "Different message sizes";

    int size = msg1.size();

    bool equal = true;
    double real_diff_min = msg1[0].real() - msg2[0].real();
    double real_diff_max = real_diff_min;
    double imag_diff_min = msg1[0].imag() - msg2[0].imag();
    double imag_diff_max = imag_diff_min;
    double abs_diff_min = std::abs(msg1[0] - msg2[0]);
    double abs_diff_max = abs_diff_min;

    double diff_magnitude_sum = 0;
    double diff_magnitude_sq_sum = 0;
    double msg1_magnitude_sq_sum = 0;

    for (int i = 0; i < size; ++i) {
      Complex diff = msg1[i] - msg2[i];
      if (std::abs(diff.real()) > max_error ||
          std::abs(diff.imag()) > max_error) {
        equal = false;
      }
      real_diff_min = std::min(real_diff_min, diff.real());
      real_diff_max = std::max(real_diff_max, diff.real());
      imag_diff_min = std::min(imag_diff_min, diff.imag());
      imag_diff_max = std::max(imag_diff_max, diff.imag());
      auto abs_diff = std::abs(diff);
      abs_diff_min = std::min(abs_diff_min, abs_diff);
      abs_diff_max = std::max(abs_diff_max, abs_diff);
      diff_magnitude_sum += abs_diff;
      diff_magnitude_sq_sum +=
          diff.real() * diff.real() + diff.imag() * diff.imag();
      msg1_magnitude_sq_sum +=
          msg1[i].real() * msg1[i].real() + msg1[i].imag() * msg1[i].imag();
    }

    // printing error stats
    if (print) {
      std::cout << std::scientific << std::setprecision(5);
      std::cout << "------------ Error stats (diff = expected - obtained) "
                   "------------"
                << std::endl;
      std::cout << "Diff real range: [ " << real_diff_min << ", "
                << real_diff_max << " ]" << std::endl;
      std::cout << "Diff imag range: [ " << imag_diff_min << ", "
                << imag_diff_max << " ]" << std::endl;
      std::cout << "Diff magnitude (sqrt(real^2 + imag^2)) range: ["
                << abs_diff_min << ", " << abs_diff_max << " ]" << std::endl;
      std::cout << "Average diff magnitude: " << diff_magnitude_sum / size
                << std::endl;
      std::cout << "SNR (E[(msg1 magnitude)^2] / E[(diff magnitude)^2]) = "
                << msg1_magnitude_sq_sum / diff_magnitude_sq_sum << std::endl;
      std::cout << "-----------------------------------------------------------"
                   "-------"
                << std::endl;
      std::cout << std::fixed << std::endl;
    }

    ASSERT_EQ(equal, true) << "Messages are not equal";
  }
};
