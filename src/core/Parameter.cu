#include <algorithm>
#include <numeric>

#include "core/Parameter.h"
#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"

namespace {
bool CompareNPPair(const std::pair<int, int>& a, const std::pair<int, int>& b) {
  // a < b: Has more total primes or, if equal, has more main primes
  int a_total = a.first + a.second;
  int b_total = b.first + b.second;
  return (a_total < b_total) || (a_total == b_total && a.first < b.first);
}
}  // namespace

namespace cheddar {

template <typename word>
Parameter<word>::Parameter(
    int log_degree, double base_scale, int default_encryption_level,
    std::vector<std::pair<int, int>> level_config,
    const std::vector<word>& main_primes, const std::vector<word>& aux_primes,
    const std::vector<word>& ter_primes /*= std::vector<word>{}*/,
    const std::pair<int, int>& additional_base /*= std::pair<int, int>{0, 0}*/)
    : log_degree_{log_degree},
      degree_{1 << log_degree},
      dnum_{DivCeil(main_primes.size() + ter_primes.size(), aux_primes.size())},
      L_{static_cast<int>(main_primes.size() + ter_primes.size())},
      alpha_{static_cast<int>(aux_primes.size())},
      base_scale_{base_scale},
      default_encryption_level_{default_encryption_level},
      max_level_{static_cast<int>(level_config.size()) - 1},
      main_primes_{main_primes},
      ter_primes_{ter_primes},
      aux_primes_{aux_primes},
      level_config_{level_config},
      additional_base_{additional_base},
      dense_h_{1 << (log_degree - 1)},
      sparse_h_{32} {

  // 0. Check for various constraints
  // Check if the primes are valid
  word max_prime = 0;
  auto max_func = [](word a, word b) -> word { return Max<word>(a, b); };

  std::accumulate(main_primes.cbegin(), main_primes.cend(), max_prime,
                  max_func);
  std::accumulate(aux_primes.cbegin(), aux_primes.cend(), max_prime, max_func);
  std::accumulate(ter_primes.cbegin(), ter_primes.cend(), max_prime, max_func);
  AssertTrue(max_prime < (((word)1) << (word_size_ * 8 - extra_bits_)),
             "Max prime is too large for the word size");

  // Check if the primes are NTT-friendly
  word ntt_base = degree_ * 2;
  auto assert_ntt_friendly = [ntt_base](word prime) {
    AssertTrue(prime % ntt_base == 1, "Prime is not NTT-friendly");
  };
  std::for_each(main_primes.cbegin(), main_primes.cend(), assert_ntt_friendly);
  std::for_each(aux_primes.cbegin(), aux_primes.cend(), assert_ntt_friendly);
  std::for_each(ter_primes.cbegin(), ter_primes.cend(), assert_ntt_friendly);

  // Check if the level config is valid
  int level_config_size = level_config.size();
  int max_num_main = main_primes_.size();
  int max_num_ter = ter_primes_.size();
  std::pair<int, int> prev_config{0, 0};
  for (int i = 0; i < level_config_size - 1; i++) {
    const auto& config = level_config[i];
    int num_main = config.first;
    int num_ter = config.second;
    AssertTrue(num_main >= 0 && num_main <= max_num_main,
               "Invalid number of main primes (" + std::to_string(num_main) +
                   ") at level " + std::to_string(i));
    AssertTrue(num_ter >= 0 && num_ter <= max_num_ter,
               "Invalid number of terminal primes (" + std::to_string(num_ter) +
                   ") at level " + std::to_string(i));
    AssertTrue(CompareNPPair(prev_config, config),
               "Invalid level configuration at level " + std::to_string(i));
    prev_config = config;
  }
  // The last level should contain all the main and terminal primes
  int num_main = level_config.back().first;
  int num_ter = level_config.back().second;
  AssertTrue(num_main == max_num_main, "Invalid number of main primes (" +
                                           std::to_string(num_main) +
                                           ") at the last level");
  AssertTrue(num_ter == max_num_ter, "Invalid number of terminal primes (" +
                                         std::to_string(num_ter) +
                                         ") at the last level");

  // 1. Populate galois factors
  int max_num_slots = degree_ / 2;
  int m = degree_ * 2;
  galois_factors_.resize(max_num_slots + 1);
  galois_factors_[max_num_slots] = 1;
  galois_factors_[0] = 1;
  for (int i = 1; i < max_num_slots; i++) {
    galois_factors_[i] = (galois_factors_[i - 1] * galois_number_) % m;
  }

  // 2. Flatten the primes and calculate the inverse primes
  q_primes_.insert(q_primes_.end(), ter_primes_.rbegin(), ter_primes_.rend());
  q_primes_.insert(q_primes_.end(), main_primes_.cbegin(), main_primes_.cend());
  inv_q_primes_.resize(q_primes_.size());
  std::transform(q_primes_.cbegin(), q_primes_.cend(), inv_q_primes_.begin(),
                 primeutil::InvModBase<word>);
  inv_aux_primes_.resize(alpha_);
  std::transform(aux_primes_.cbegin(), aux_primes_.cend(),
                 inv_aux_primes_.begin(), primeutil::InvModBase<word>);

  // 3. Populating per-level primes and inv_primes DeviceVectors
  for (int nm = 0; nm <= max_num_main + max_num_ter; nm++) {
    if (nm == 0) {
      primes_dv_.push_back(nullptr);
      inv_primes_dv_.push_back(nullptr);
      continue;
    }
    word* prime_ptr;
    signed_word* inv_prime_ptr;
    std::vector<word> primes(q_primes_.cbegin(), q_primes_.cbegin() + nm);
    std::vector<signed_word> inv_primes(inv_q_primes_.cbegin(),
                                        inv_q_primes_.cbegin() + nm);
    primes.insert(primes.end(), aux_primes_.cbegin(), aux_primes_.cend());
    inv_primes.insert(inv_primes.end(), inv_aux_primes_.cbegin(),
                      inv_aux_primes_.cend());
    cudaMalloc(&prime_ptr, primes.size() * sizeof(word));
    cudaMalloc(&inv_prime_ptr, inv_primes.size() * sizeof(signed_word));
    cudaMemcpy(prime_ptr, primes.data(), primes.size() * sizeof(word),
               cudaMemcpyHostToDevice);
    cudaMemcpy(inv_prime_ptr, inv_primes.data(),
               inv_primes.size() * sizeof(signed_word), cudaMemcpyHostToDevice);
    primes_dv_.push_back(prime_ptr);
    inv_primes_dv_.push_back(inv_prime_ptr);
  }

  // 4. Short base may be used for bootstrapping (SSE)
  short_base_np_.num_main_ = level_config_.at(0).first - additional_base_.first;
  short_base_np_.num_ter_ =
      level_config_.at(0).second - additional_base_.second;
  short_base_np_.num_aux_ = short_base_np_.num_main_ + short_base_np_.num_ter_;
  AssertTrue(additional_base_.first >= 0 && additional_base_.second >= 0 &&
                 short_base_np_.num_main_ >= 0 &&
                 short_base_np_.num_ter_ >= 0 &&
                 short_base_np_.num_aux_ <= alpha_,
             "Invalid additional base configuration");

  // 5. Computing scale-related constants
  rescale_prime_prod_.resize(max_level_ + 1);
  // rescale_prime_prod_[0] is undefined
  scale_.resize(default_encryption_level_ + 1);
  scale_[0] = base_scale_;

  double new_scale = base_scale_;

  for (int i = 1; i <= max_level_; i++) {
    NPInfo np_upper = LevelToNP(i);
    NPInfo np_lower = LevelToNP(i - 1);
    int ter_diff = np_upper.num_ter_ - np_lower.num_ter_;
    int main_diff = np_upper.num_main_ - np_lower.num_main_;

    double scale_numerator = 1.0;
    double scale_denominator = 1.0;

    if (ter_diff >= 0) {
      for (int j = 0; j < ter_diff; j++) {
        scale_numerator *= ter_primes_[np_lower.num_ter_ + j];
      }
    } else {
      for (int j = 0; j < -ter_diff; j++) {
        scale_denominator *= ter_primes_[np_upper.num_ter_ + j];
      }
    }

    if (main_diff >= 0) {
      for (int j = 0; j < main_diff; j++) {
        scale_numerator *= main_primes_[np_lower.num_main_ + j];
      }
    } else {
      for (int j = 0; j < -main_diff; j++) {
        scale_denominator *= main_primes_[np_upper.num_main_ + j];
      }
    }

    double prod = scale_numerator / scale_denominator;
    rescale_prime_prod_[i] = prod;

    if (i <= default_encryption_level_) {
      new_scale = std::sqrt(new_scale * prod);
      scale_[i] = new_scale;
    }
  }
}

template <typename word>
int Parameter<word>::GetGaloisFactor(int i) const {
  AssertTrue(i >= 0 && i <= degree_ / 2,
             "GetGaloisFactor: Invalid Galois factor index");
  return galois_factors_.at(i);
}

template <typename word>
double Parameter<word>::GetScale(int level) const {
  if (level == -1) level = 0;
  AssertTrue(level >= 0 && level <= default_encryption_level_,
             "GetScale: Invalid level");
  return scale_.at(level);
}

template <typename word>
double Parameter<word>::GetRescalePrimeProd(int level) const {
  AssertTrue(level > 0 && level <= max_level_,
             "GetRescalePrimeProd: Invalid level");
  return rescale_prime_prod_.at(level);
}

template <typename word>
int Parameter<word>::GetDenseHammingWeight() const {
  return dense_h_;
}

template <typename word>
int Parameter<word>::GetSparseHammingWeight() const {
  return sparse_h_;
}

template <typename word>
void Parameter<word>::SetDenseHammingWeight(int h) {
  AssertTrue(h >= sparse_h_ && h <= degree_,
             "SetDenseHammingWeight: Dense Hamming weight must be greater than "
             "sparse Hamming weight and smaller than degree");
  dense_h_ = h;
}

template <typename word>
void Parameter<word>::SetSparseHammingWeight(int h) {
  AssertTrue(
      h > 0 && h <= dense_h_,
      "SetSparseHammingWeight: Sparse Hamming weight must be smaller than "
      "dense Hamming weight");
  sparse_h_ = h;
}

template <typename word>
bool Parameter<word>::IsUsingSparseSecretEncapsulation() const {
  return dense_h_ > sparse_h_;
}

template <typename word>
int Parameter<word>::GetMaxNumTer() const {
  return ter_primes_.size();
}

template <typename word>
int Parameter<word>::GetMaxNumMain() const {
  return main_primes_.size();
}

template <typename word>
int Parameter<word>::GetMaxNumQ() const {
  return main_primes_.size() + ter_primes_.size();
}

template <typename word>
int Parameter<word>::GetMaxNumAux() const {
  return alpha_;
}

template <typename word>
int Parameter<word>::GetSSENumAux() const {
  return level_config_.at(0).first + level_config_.at(0).second -
         additional_base_.first - additional_base_.second;
}

template <typename word>
void Parameter<word>::AssertValidNP(const NPInfo& np) const {
  AssertTrue(np.num_main_ >= 0 && np.num_main_ <= GetMaxNumMain(),
             "Invalid number of main primes");
  AssertTrue(np.num_ter_ >= 0 && np.num_ter_ <= GetMaxNumTer(),
             "Invalid number of terminal primes");
  AssertTrue(np.num_aux_ >= 0 && np.num_aux_ <= GetMaxNumAux(),
             "Invalid number of aux primes");
}

template <typename word>
NPInfo Parameter<word>::LevelToNP(int level, int num_aux /*= 0*/) const {
  AssertTrue(num_aux >= 0 && num_aux <= alpha_, "Invalid number of aux primes");
  if (level == -1) {
    AssertTrue(num_aux <= short_base_np_.num_aux_,
               "Invalid number of aux primes for short base");
    return NPInfo(short_base_np_.num_main_, short_base_np_.num_ter_, num_aux);
  }
  const auto& config = level_config_.at(level);
  return NPInfo(config.first, config.second, num_aux);
}

template <typename word>
int Parameter<word>::NPToLevel(const NPInfo& np) const {
  // Perform a binary search to find the NPInfo
  auto found = std::equal_range(level_config_.cbegin(), level_config_.cend(),
                                std::make_pair(np.num_main_, np.num_ter_),
                                CompareNPPair);
  if (found.first == found.second) {
    // If not found, check for short base
    if (np.num_main_ == short_base_np_.num_main_ &&
        np.num_ter_ == short_base_np_.num_ter_ &&
        np.num_aux_ <= short_base_np_.num_aux_) {
      return -1;
    }
    // Otherwise, fails
    Fail("NPInfo not found");
  }
  return found.first - level_config_.cbegin();
}

template <typename word>
std::vector<word> Parameter<word>::GetPrimeVector(const NPInfo& np) const {
  AssertValidNP(np);
  std::vector<word> res(q_primes_.cbegin() + GetMaxNumTer() - np.num_ter_,
                        q_primes_.cbegin() + GetMaxNumTer() + np.num_main_);
  res.insert(res.end(), aux_primes_.cbegin(),
             aux_primes_.cbegin() + np.num_aux_);
  return res;
}

template <typename word>
const word* Parameter<word>::GetPrimesPtr(const NPInfo& np) const {
  int index = np.num_main_ + GetMaxNumTer();
  AssertTrue(index != 0, "Invalid np for GetPrimesPtr");
  return primes_dv_.at(index) + (GetMaxNumTer() - np.num_ter_);
}

template <typename word>
const make_signed_t<word>* Parameter<word>::GetInvPrimesPtr(
    const NPInfo& np) const {
  int index = np.num_main_ + GetMaxNumTer();
  AssertTrue(index != 0, "Invalid np for GetInvPrimesPtr");
  return inv_primes_dv_.at(index) + (GetMaxNumTer() - np.num_ter_);
}

template <typename word>
const word* Parameter<word>::__GetPrimesPtrModDownWithTerPrimes(
    const NPInfo& np_src, const NPInfo& np_non_intt) const {
  int index = GetMaxNumTer() - np_non_intt.num_ter_;
  return primes_dv_.at(index) + (GetMaxNumTer() - np_src.num_ter_);
}

template <typename word>
const make_signed_t<word>*
Parameter<word>::__GetInvPrimesPtrModDownWithTerPrimes(
    const NPInfo& np_src, const NPInfo& np_non_intt) const {
  int index = GetMaxNumTer() - np_non_intt.num_ter_;
  return inv_primes_dv_.at(index) + (GetMaxNumTer() - np_src.num_ter_);
}

template <typename word>
Parameter<word>::~Parameter() {
  for (auto prime_ptr : primes_dv_) {
    if (prime_ptr != nullptr) {
      cudaFree(prime_ptr);
    }
  }
  for (auto inv_prime_ptr : inv_primes_dv_) {
    if (inv_prime_ptr != nullptr) {
      cudaFree(inv_prime_ptr);
    }
  }
  primes_dv_.clear();
  inv_primes_dv_.clear();
}

template class Parameter<uint32_t>;
template class Parameter<uint64_t>;

}  // namespace cheddar
