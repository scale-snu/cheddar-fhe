#include "UserInterface.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"
#include "common/ConstantMemory.cuh"
#include "common/PrimeUtils.h"
#include "common/PtrList.h"

namespace cheddar {

namespace kernel {

// dst.ptrs_[0] --> bx (uninitialized)
// dst.ptrs_[1] --> ax (sampled random value)
// bx = -ax * sx + mx + ex
// For this function, we use prime_index modification
template <typename word>
__global__ void Encrypt(OutputPtrList<word, 2> dst, const word *primes,
                        const make_signed_t<word> *inv_primes, int num_q_primes,
                        const InputPtrList<word, 1> sx,
                        const InputPtrList<word, 1> mx,
                        const InputPtrList<word, 1> ex) {
  using signed_word = make_signed_t<word>;
  int log_degree = cm_log_degree();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);

  int sx_index = i;
  int mx_index = i;
  int ex_index = i;
  if (prime_index >= num_q_primes) {
    sx_index += sx.extra_;
    mx_index += mx.extra_;
    ex_index += ex.extra_;
  }

  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  word sx_value = basic::StreamingLoad(sx.ptrs_[0] + sx_index);
  word mx_value = basic::StreamingLoad(mx.ptrs_[0] + mx_index);
  word ex_value = basic::StreamingLoad(ex.ptrs_[0] + ex_index);
  word ax_value = basic::StreamingLoad(dst.ptrs_[1] + i);

  word res = basic::MultMontgomery(ax_value, sx_value, prime, inv_prime);
  res = basic::Sub(mx_value, res, prime);
  res = basic::Add(res, ex_value, prime);

  dst.ptrs_[0][i] = res;
}

// dst.ptrs_[0] --> bx (uninitialized)
// dst.ptrs_[1] --> ax (sampled random value)
// bx = -ax * sx + mx + ex
// For this function, we use prime_index modification
template <typename word>
__global__ void EncryptZero(OutputPtrList<word, 2> dst, const word *primes,
                            const make_signed_t<word> *inv_primes,
                            int num_q_primes, const InputPtrList<word, 1> sx,
                            const InputPtrList<word, 1> ex) {
  using signed_word = make_signed_t<word>;
  int log_degree = cm_log_degree();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);

  int sx_index = i;
  int ex_index = i;
  if (prime_index >= num_q_primes) {
    sx_index += sx.extra_;
    ex_index += ex.extra_;
  }

  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);

  word sx_value = basic::StreamingLoad(sx.ptrs_[0] + sx_index);
  word ex_value = basic::StreamingLoad(ex.ptrs_[0] + ex_index);
  word ax_value = basic::StreamingLoad(dst.ptrs_[1] + i);

  word res = basic::MultMontgomery(ax_value, sx_value, prime, inv_prime);
  res = basic::Sub(ex_value, res, prime);

  dst.ptrs_[0][i] = res;
}

// Contiguous memory access is guaranteed
template <typename word>
__global__ void AddEvkPart(word *dst, const word *primes,
                           const make_signed_t<word> *inv_primes,
                           const word *src, const word *p_prod) {
  using signed_word = make_signed_t<word>;
  int log_degree = cm_log_degree();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int prime_index = (i >> log_degree);
  const word prime = basic::StreamingLoadConst(primes + prime_index);
  const signed_word inv_prime =
      basic::StreamingLoadConst(inv_primes + prime_index);
  const word p_prod_value = basic::StreamingLoadConst(p_prod + prime_index);

  word src_value = basic::StreamingLoad(src + i);
  word dst_value = basic::StreamingLoad(dst + i);

  word res = basic::MultMontgomery(src_value, p_prod_value, prime, inv_prime);
  res = basic::Add(dst_value, res, prime);

  dst[i] = res;
}

}  // namespace kernel

template <typename word>
UserInterface<word>::UserInterface(ContextPtr<word> context)
    : context_{std::move(context)} {
  const auto &param = context_->param_;
  all_primes_ =
      param.GetPrimeVector(param.LevelToNP(param.max_level_, param.alpha_));

  if (!cm_populated_) {
    PopulateConstantMemory(context_->param_);
    cm_populated_ = true;
  }

  Warn(
      "UserInterface is for testing purposes only. "
      "Do not use in production.");
  PrepareSecrets();
  PrepareBasicEvks();
}

template <typename word>
void UserInterface<word>::Encrypt(Ct &ctxt, const Pt &ptxt) const {
  NPInfo np = ptxt.GetNP();

  // Setting metadata
  ctxt.RemoveRx();
  ctxt.ModifyNP(np);
  ctxt.SetScale(ptxt.GetScale());
  ctxt.SetNumSlots(ptxt.GetNumSlots());

  SampleRandomPolynomial(ctxt.ax_, np);
  int num_q_primes = np.num_main_ + np.num_ter_;
  int num_total_primes = np.GetNumTotal();
  int num_aux = np.num_aux_;
  int degree = context_->param_.degree_;

  Dv ex_dv(num_total_primes * degree);
  SampleError(ex_dv, np);

  // Prepare PtrLists
  const word *primes = context_->param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = context_->param_.GetInvPrimesPtr(np);

  InputPtrList<word, 1> sx;
  sx.ptrs_[0] = main_secret_.data() +
                (context_->param_.GetMaxNumTer() - np.num_ter_) * degree;
  sx.extra_ = (context_->param_.GetMaxNumMain() - np.num_main_) * degree;
  InputPtrList<word, 1> mx(ptxt.ConstView());
  InputPtrList<word, 1> ex(ex_dv.ConstView(num_aux * degree));
  auto ctxt_temp = ctxt.ViewVector();
  OutputPtrList<word, 2> dst(ctxt_temp);

  int grid_dim = num_total_primes * degree / kernel_block_dim_;

  // bx = -ax * sx + mx + ex
  kernel::Encrypt<word><<<grid_dim, kernel_block_dim_>>>(
      dst, primes, inv_primes, num_q_primes, sx, mx, ex);
}

template <typename word>
void UserInterface<word>::Decrypt(Pt &ptxt, const Ct &ctxt) const {
  NPInfo np = ctxt.GetNP();
  AssertTrue(np.num_aux_ == 0, "Decrypt: ModDown required before decryption");
  AssertTrue(!ctxt.HasRx(), "Decrypt: Rx should be removed before decryption");
  int degree = context_->param_.degree_;

  // Setting metadata
  ptxt.ModifyNP(np);
  ptxt.SetScale(ctxt.GetScale());
  ptxt.SetNumSlots(ctxt.GetNumSlots());
  int secret_front_ignore = context_->param_.GetMaxNumTer() - np.num_ter_;

  // ax * sx + bx ~= mx
  std::vector<DvView<word>> paccum_res{ptxt.View()};
  context_->elem_handler_.PAccum(
      paccum_res, np,
      {std::vector<DvConstView<word>>{ctxt.AxConstView()},
       std::vector<DvConstView<word>>{ctxt.BxConstView()}},
      {MainSecretConstView(secret_front_ignore)});
}

template <typename word>
const EvaluationKey<word> &UserInterface<word>::GetRotationKey(
    int rot_idx) const {
  return evk_map_.GetRotationKey(rot_idx);
}

template <typename word>
const EvaluationKey<word> &UserInterface<word>::GetMultiplicationKey() const {
  return evk_map_.GetMultiplicationKey();
}

template <typename word>
const EvaluationKey<word> &UserInterface<word>::GetConjugationKey() const {
  return evk_map_.GetConjugationKey();
}

template <typename word>
const EvaluationKey<word> &UserInterface<word>::GetDenseToSparseKey() const {
  return evk_map_.GetDenseToSparseKey();
}

template <typename word>
const EvaluationKey<word> &UserInterface<word>::GetSparseToDenseKey() const {
  return evk_map_.GetSparseToDenseKey();
}

template <typename word>
const EvkMap<word> &UserInterface<word>::GetEvkMap() const {
  return evk_map_;
}

template <typename word>
void UserInterface<word>::PrepareSecrets() {
  int degree = context_->param_.degree_;
  int num_total_primes = all_primes_.size();
  int alpha = context_->param_.alpha_;

  NPInfo np_max = context_->param_.LevelToNP(context_->param_.max_level_,
                                             context_->param_.alpha_);

  // Sampling sparse ternary secret
  int hamming_weight = context_->param_.GetDenseHammingWeight();
  std::vector<int> indices(hamming_weight);
  std::vector<word> ternary_values(hamming_weight);
  Random::SampleWithoutReplacement(indices.data(), hamming_weight, 0,
                                   degree - 1);
  Random::SampleUniformWord<word>(ternary_values.data(), hamming_weight, 0, 1);

  HostVector<word> main_s(num_total_primes * degree, 0);
  for (int i = 0; i < num_total_primes; i++) {
    word prime = all_primes_[i];
    for (int j = 0; j < hamming_weight; j++) {
      // 0 --> 1, 1 --> (prime - 1)
      main_s[i * degree + indices[j]] = (prime - 2) * ternary_values[j] + 1;
    }
  }
  CopyHostToDevice(main_secret_, main_s);
  auto main_sx_view = MainSecretView();
  context_->ntt_handler_.NTT(main_sx_view, np_max, MainSecretConstView(), true);

  if (!context_->param_.IsUsingSparseSecretEncapsulation()) {
    return;
  }

  // Sampling for sparse secret
  // We also prepare it for the maximum size
  hamming_weight = context_->param_.GetSparseHammingWeight();
  indices.resize(hamming_weight);
  ternary_values.resize(hamming_weight);
  Random::SampleWithoutReplacement(indices.data(), hamming_weight, 0,
                                   degree - 1);
  Random::SampleUniformWord<word>(ternary_values.data(), hamming_weight, 0, 1);
  HostVector<word> sparse_s(num_total_primes * degree, 0);
  for (int i = 0; i < num_total_primes; i++) {
    word prime = all_primes_[i];
    for (int j = 0; j < hamming_weight; j++) {
      // 0 --> 1, 1 --> (prime - 1)
      sparse_s[i * degree + indices[j]] = (prime - 2) * ternary_values[j] + 1;
    }
  }
  CopyHostToDevice(sparse_secret_, sparse_s);
  auto sparse_sx_view = SparseSecretView();
  context_->ntt_handler_.NTT(sparse_sx_view, np_max, SparseSecretConstView(),
                             true);
}

template <typename word>
void UserInterface<word>::PrepareRotationKey(int rot_idx, int max_level) {
  int L = context_->param_.L_;
  int alpha = context_->param_.alpha_;
  int max_num_ter = context_->param_.GetMaxNumTer();
  int degree = context_->param_.degree_;
  NPInfo np = GetNPForEvk(max_level);
  int half_degree = degree / 2;

  AssertTrue(rot_idx > 0 && rot_idx < half_degree,
             "Invalid rotation index " + std::to_string(rot_idx));
  if (evk_map_.find(rot_idx) != evk_map_.end()) {  // if already prepared
    const Evk &evk = evk_map_.at(rot_idx);
    if (np.IsSubsetOf(evk.GetNP())) {
      Warn("Rotation key for rotation index " + std::to_string(rot_idx) +
           " already prepared");
      return;
    }
  }

  Dv s_rot_dv(np.GetNumTotal() * degree);
  int aux_size = alpha * degree;
  std::vector<DvView<word>> s_rot{s_rot_dv.View(aux_size)};
  int ter_left = max_num_ter - np.num_ter_;

  // we permute it in the opposite direction
  context_->elem_handler_.Permute(s_rot, np, half_degree - rot_idx,
                                  {MainSecretConstView(ter_left)});
  PrepareEvk(rot_idx, np, s_rot_dv, main_secret_);
}

template <typename word>
void UserInterface<word>::PrepareRotationKey(const EvkRequest &evk_request) {
  for (const auto &[rot_idx, level] : evk_request) {
    if (rot_idx == 0) continue;
    AssertTrue(rot_idx > 0,
               "Invalid rotation index " + std::to_string(rot_idx));
    PrepareRotationKey(Abs(rot_idx), level);
  }
}

template <typename word>
void UserInterface<word>::PrepareBasicEvks() {
  int L = context_->param_.L_;
  int alpha = context_->param_.alpha_;
  int degree = context_->param_.degree_;
  NPInfo np = GetNPForEvk(context_->param_.max_level_);

  // Multiplication key
  Dv s_squared(np.GetNumTotal() * degree);
  std::vector<DvView<word>> s_squared_view{s_squared.View(alpha * degree)};
  std::vector<DvConstView<word>> sx_view{MainSecretConstView()};

  context_->elem_handler_.Mult(s_squared_view, np, sx_view, sx_view);
  PrepareEvk(EvkMap<word>::kMultiplicationKeyIndex, np, main_secret_,
             s_squared);

  // Conjugation key
  Dv s_conj(np.GetNumTotal() * degree);
  std::vector<DvView<word>> s_conj_view{s_conj.View(alpha * degree)};
  context_->elem_handler_.Permute(s_conj_view, np, -1, sx_view);
  PrepareEvk(EvkMap<word>::kConjugationKeyIndex, np, s_conj, main_secret_);

  if (context_->param_.IsUsingSparseSecretEncapsulation()) {
    // Dense to Sparse key
    NPInfo dts_np = GetNPForEvk(-1);

    PrepareEvk(EvkMap<word>::kDenseToSparseKeyIndex, dts_np, sparse_secret_,
               main_secret_);

    // Sparse to Dense key
    PrepareEvk(EvkMap<word>::kSparseToDenseKeyIndex, np, main_secret_,
               sparse_secret_);
    // We can simplify SparseToDenseKey
    // TODO: this key has beta times higher error than necessary
    auto &std_key = evk_map_.at(EvkMap<word>::kSparseToDenseKeyIndex);
    int beta = DivCeil(L, alpha);
    std::vector<DvView<word>> std_key_view = std_key.ViewVector(0);
    std::vector<std::vector<DvConstView<word>>> std_key_accum_inputs;
    for (int i = 0; i < beta; i++) {
      std_key_accum_inputs.push_back(std_key.ConstViewVector(i));
    }
    context_->elem_handler_.Accum(std_key_view, np, std_key_accum_inputs);
  }
}

// encryption_secret and target_secret are always prepared at the maximum level
// Or exactly follows the np
template <typename word>
void UserInterface<word>::PrepareEvk(int key_idx, const NPInfo &np,
                                     const Dv &encryption_secret,
                                     const Dv &target_secret) {
  int degree = context_->param_.degree_;
  // Beware that alpha != np.num_aux_
  int alpha = context_->param_.alpha_;
  int L = context_->param_.L_;
  int max_num_ter = context_->param_.GetMaxNumTer();
  int num_q = np.num_main_ + np.num_ter_;
  int beta = DivCeil(num_q, np.num_aux_);

  int ter_left = max_num_ter - np.num_ter_;
  int main_left = context_->param_.GetMaxNumMain() - np.num_main_;

  // Preparing input pointers
  int enc_s_size = encryption_secret.size();
  InputPtrList<word, 1> enc_s;
  if (enc_s_size == (L + alpha) * degree) {
    enc_s.ptrs_[0] = encryption_secret.data() + ter_left * degree;
    enc_s.extra_ = main_left * degree;
  } else if (enc_s_size == np.GetNumTotal() * degree) {
    enc_s.ptrs_[0] = encryption_secret.data();
    enc_s.extra_ = 0;
  } else {
    Fail("PrepareEvk: encryption secret is not sized correctly");
  }
  int target_s_size = target_secret.size();
  const word *target_s_ptr = nullptr;
  if (target_s_size == (L + alpha) * degree) {
    target_s_ptr = target_secret.data() + ter_left * degree;
  } else if (target_s_size == np.GetNumTotal() * degree) {
    target_s_ptr = target_secret.data();
  } else {
    Fail("PrepareEvk: target secret is not sized correctly");
  }

  Dv p_prod(num_q);
  // Prepare p_prod
  HostVector<word> h_p_prod(num_q);
  for (int i = 0; i < num_q; i++) {
    word mod_prime = all_primes_[i + ter_left];
    word p_prod_mod_qi = 1;
    for (int j = 0; j < np.num_aux_; j++) {
      p_prod_mod_qi =
          primeutil::MultMod(p_prod_mod_qi, all_primes_[j + L], mod_prime);
    }
    h_p_prod[i] = primeutil::ToMontgomery(p_prod_mod_qi, mod_prime);
  }
  CopyHostToDevice(p_prod, h_p_prod);

  // Initialize object in the EvkMap
  if (evk_map_.find(key_idx) != evk_map_.end()) {
    Warn("Overwriting the evk for key index " + std::to_string(key_idx));
    evk_map_.erase(key_idx);
  }
  evk_map_.try_emplace(key_idx, np, beta);
  Evk &evk = evk_map_.at(key_idx);

  // 1. Prepare primes
  const word *primes = context_->param_.GetPrimesPtr(np);
  const make_signed_t<word> *inv_primes = context_->param_.GetInvPrimesPtr(np);

  // Prepare each of beta pairs.
  // bx = -ax * encryption_secret + target_secret * const + error
  for (int i = 0; i < beta; i++) {
    // Preparing the encryption of 0
    SampleRandomPolynomial(evk.ax_.at(i), np);
    Dv ex_dv(np.GetNumTotal() * degree);
    SampleError(ex_dv, np);

    auto evk_temp = evk.ViewVector(i);
    OutputPtrList<word, 2> evk_ptr_list(evk_temp);
    InputPtrList<word, 1> ex(ex_dv.ConstView(np.num_aux_ * degree));

    int grid_dim = (num_q + np.num_aux_) * degree / kernel_block_dim_;

    kernel::EncryptZero<word><<<grid_dim, kernel_block_dim_>>>(
        evk_ptr_list, primes, inv_primes, num_q, enc_s, ex);

    int chunk_size = np.num_aux_;
    if (i == beta - 1) {
      chunk_size = num_q - (beta - 1) * np.num_aux_;
    }
    grid_dim = chunk_size * degree / kernel_block_dim_;

    kernel::AddEvkPart<word><<<grid_dim, kernel_block_dim_>>>(
        evk.bx_.at(i).data() + i * np.num_aux_ * degree,
        primes + i * np.num_aux_, inv_primes + i * np.num_aux_,
        target_s_ptr + i * np.num_aux_ * degree,
        p_prod.data() + i * np.num_aux_);
  }
}

// We can regard this random polynomial as having any form we want
// It can be regarded to be NTT-applied or not (with or without Montgomery form)
template <typename word>
void UserInterface<word>::SampleRandomPolynomial(Dv &poly,
                                                 const NPInfo &np) const {
  int degree = context_->param_.degree_;
  int max_num_ter = context_->param_.GetMaxNumTer();
  int num_q = np.num_main_ + np.num_ter_;
  int L = context_->param_.L_;
  int num_total_primes = num_q + np.num_aux_;
  AssertTrue(num_total_primes * degree == static_cast<int>(poly.size()),
             "SampleRandomPolynomial: Invalid poly size");
  int prime_offset = max_num_ter - np.num_ter_;

  HostVector<word> poly_host(num_total_primes * degree, 0);
  for (int i = 0; i < num_total_primes; i++) {
    int prime_index = i + prime_offset;
    if (i >= num_q) {
      prime_index = L + i - num_q;
    }
    word prime = all_primes_[prime_index];
    Random::SampleUniformWord<word>(poly_host.data() + i * degree, degree, 0,
                                    prime - 1);
  }
  CopyHostToDevice(poly, poly_host);
}

template <typename word>
void UserInterface<word>::SampleError(Dv &poly, const NPInfo &np) const {
  int degree = context_->param_.degree_;
  int param_max_level = context_->param_.max_level_;
  int max_num_ter = context_->param_.GetMaxNumTer();
  int num_q = np.num_main_ + np.num_ter_;
  int L = context_->param_.L_;
  int num_total_primes = num_q + np.num_aux_;
  AssertTrue(num_total_primes * degree == static_cast<int>(poly.size()),
             "SampleError: Invalid poly size");
  int prime_offset = max_num_ter - np.num_ter_;

  HostVector<word> poly_host(num_total_primes * degree, 0);

  std::vector<int> error(degree);
  Random::SampleDiscreteNormal(error.data(), degree, 0,
                               kErrorStandardDeviation);
  for (int i = 0; i < num_total_primes; i++) {
    int prime_index = i + prime_offset;
    if (i >= num_q) {
      prime_index = L + i - num_q;
    }
    word prime = all_primes_[prime_index];
    for (int j = 0; j < degree; j++) {
      poly_host[i * degree + j] =
          ((error[j] < 0) ? prime - static_cast<word>(-error[j])
                          : static_cast<word>(error[j]));
    }
  }
  CopyHostToDevice(poly, poly_host);
  int aux_size = np.num_aux_ * degree;
  auto poly_view = poly.View(aux_size);
  context_->ntt_handler_.NTT(poly_view, np, poly.ConstView(aux_size), true);
}

template <typename word>
NPInfo UserInterface<word>::GetNPForEvk(int max_level) const {
  // DtS case
  if (max_level == -1) {
    NPInfo short_base = context_->param_.LevelToNP(-1);
    short_base.num_aux_ = short_base.num_main_ + short_base.num_ter_;
    return short_base;
  }

  // Normal case
  NPInfo res = context_->param_.LevelToNP(0);
  for (int i = 1; i <= max_level; i++) {
    NPInfo np_i = context_->param_.LevelToNP(i);
    res.num_main_ = Max(res.num_main_, np_i.num_main_);
  }
  res.num_ter_ = context_->param_.GetMaxNumTer();
  res.num_aux_ = context_->param_.alpha_;

  return res;
}

template <typename word>
DvView<word> UserInterface<word>::MainSecretView(int front_ignore /*= 0*/) {
  int degree = context_->param_.degree_;
  int alpha = context_->param_.alpha_;
  return main_secret_.View(alpha * degree, front_ignore * degree);
}

template <typename word>
DvConstView<word> UserInterface<word>::MainSecretConstView(
    int front_ignore /*= 0*/) const {
  int degree = context_->param_.degree_;
  int alpha = context_->param_.alpha_;
  return main_secret_.ConstView(alpha * degree, front_ignore * degree);
}

template <typename word>
DvView<word> UserInterface<word>::SparseSecretView(int front_ignore /*= 0*/) {
  AssertTrue(context_->param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret is not used");
  int degree = context_->param_.degree_;
  int alpha = context_->param_.alpha_;
  return sparse_secret_.View(alpha * degree, front_ignore * degree);
}

template <typename word>
DvConstView<word> UserInterface<word>::SparseSecretConstView(
    int front_ignore /*= 0*/) const {
  AssertTrue(context_->param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret is not used");
  int degree = context_->param_.degree_;
  int alpha = context_->param_.alpha_;
  return sparse_secret_.ConstView(alpha * degree, front_ignore * degree);
}

template class UserInterface<uint32_t>;
template class UserInterface<uint64_t>;

}  // namespace cheddar