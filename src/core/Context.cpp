#include "core/Context.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "common/PrimeUtils.h"

namespace cheddar {

template <typename word>
void Context<word>::MatchResultWith(Ct &res, const Ct &a) const {
  // The ordering is important here.
  if (a.HasRx()) {
    res.ModifyNP(a.GetNP());
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(a.GetNP());
  }
}

template <typename word>
void Context<word>::MatchResultWith(Ct &res, const Ct &a, const Ct &b) const {
  if (a.HasRx() || b.HasRx()) {
    res.ModifyNP(a.GetNP());
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(a.GetNP());
  }
}

template <typename word>
DvConstView<word> Context<word>::GetPProd(NPInfo &np) const {
  int prime_offset = param_.GetMaxNumTer() - np.num_ter_;
  int num_q_primes = np.GetNumQ();
  return DvConstView<word>(p_prod_.data() + prime_offset, num_q_primes);
}

template <typename word>
const ModSwitchHandler<word> &Context<word>::GetDtSModSwitchHandler() const {
  AssertTrue(param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret encapsulation is not enabled");
  return mod_switch_handlers_.back();
}

template <typename word>
const ModSwitchHandler<word> &Context<word>::GetStDModSwitchHandler() const {
  AssertTrue(param_.IsUsingSparseSecretEncapsulation(),
             "Sparse secret encapsulation is not enabled");
  return mod_switch_handlers_.at(param_.max_level_);
}

template <typename word>
void Context<word>::AssertSameScale(const double &scale1,
                                    const double &scale2) const {
  static constexpr double kScaleErrorMargin = 1e-12;
  double diff = scale1 - scale2;
  diff = diff < 0 ? -diff : diff;
  AssertTrue(diff < kScaleErrorMargin * scale1, "Scale mismatch");
}

template <typename word>
std::shared_ptr<Context<word>> Context<word>::Create(
    const Parameter<word> &param) {
  return std::shared_ptr<Context<word>>(new Context<word>(param));
}

template <typename word>
Context<word>::Context(const Parameter<word> &param)
    : param_{param},
      memory_pool_(param_),
      elem_handler_(param_),
      ntt_handler_(param_),
      encoder_(param_, ntt_handler_) {
  // 0. Set some static variables
  Container<word>::SetDegree(param_.degree_);
  MultiLevelCiphertext<word>::StaticInit(param_, encoder_);

  // 1. Initialize mod_switch_handlers_
  for (int level = 0; level <= param_.max_level_; level++) {
    mod_switch_handlers_.emplace_back(param_, level, elem_handler_,
                                      ntt_handler_);
  }

  // 2. Initialize p_prod_
  NPInfo np = param_.LevelToNP(param_.max_level_, param_.alpha_);
  std::vector<word> primes = param_.GetPrimeVector(np);
  int num_q_primes = np.GetNumQ();
  HostVector<word> h_p_prod(num_q_primes, 1);
  for (int i = 0; i < num_q_primes; i++) {
    word mod_prime = primes[i];
    for (int j = 0; j < np.num_aux_; j++) {
      h_p_prod[i] =
          primeutil::MultMod(h_p_prod[i], primes[num_q_primes + j], mod_prime);
    }
    h_p_prod[i] = primeutil::ToMontgomery(h_p_prod[i], mod_prime);
  }
  CopyHostToDevice(p_prod_, h_p_prod);

  if (!param_.IsUsingSparseSecretEncapsulation()) return;

  // 3. Initialize level_down_consts_
  level_down_consts_.resize(param_.default_encryption_level_ + 1);
  for (int i = param_.default_encryption_level_; i > 0; i--) {
    double scale = param_.GetScale(i);
    encoder_.EncodeConstant(level_down_consts_[i], i, scale, 1.0);
  }

  // Extra data for SSE
  mod_switch_handlers_.emplace_back(param_, -1, elem_handler_, ntt_handler_);
  np = param_.LevelToNP(-1, param_.GetSSENumAux());
  primes = param_.GetPrimeVector(np);
  num_q_primes = np.GetNumQ();
  HostVector<word> h_p_prod_dts(num_q_primes, 1);
  for (int i = 0; i < num_q_primes; i++) {
    word mod_prime = primes[i];
    for (int j = 0; j < np.num_aux_; j++) {
      h_p_prod_dts[i] = primeutil::MultMod(h_p_prod_dts[i],
                                           primes[num_q_primes + j], mod_prime);
    }
    h_p_prod_dts[i] = primeutil::ToMontgomery(h_p_prod_dts[i], mod_prime);
  }
  CopyHostToDevice(p_prod_dts_, h_p_prod_dts);
}

template <typename word>
Context<word>::~Context() {
  MultiLevelCiphertext<word>::StaticDestroy();
}

template <typename word>
void Context<word>::Copy(Ct &res, const Ct &a) const {
  if (&res == &a) return;
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());

  CopyDeviceToDevice(res.bx_, a.bx_);
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  bool rx_add = a.HasRx() && b.HasRx();
  MatchResultWith(res, a, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  if (rx_add) {
    auto res_temp = res.ViewVector();
    elem_handler_.Add(res_temp, np, a.ConstViewVector(), b.ConstViewVector());
  } else {
    auto res_temp = res.ViewVector(0, true);
    elem_handler_.Add(res_temp, np, a.ConstViewVector(0, true),
                      b.ConstViewVector(0, true));
    if (a.HasRx()) {
      CopyDeviceToDevice(res.rx_, a.rx_);
    } else if (b.HasRx()) {
      CopyDeviceToDevice(res.rx_, b.rx_);
    }
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Add(res_temp, np, {a.BxConstView()}, {b.ConstView()});
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Add(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.AddConst(res_temp, np, {a.BxConstView()}, b.ConstView());
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  bool rx_sub = a.HasRx() && b.HasRx();
  MatchResultWith(res, a, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  if (rx_sub) {
    auto res_temp = res.ViewVector();
    elem_handler_.Sub(res_temp, np, a.ConstViewVector(), b.ConstViewVector());
  } else {
    auto res_temp = res.ViewVector(0, true);
    elem_handler_.Sub(res_temp, np, a.ConstViewVector(0, true),
                      b.ConstViewVector(0, true));
    if (a.HasRx()) {
      CopyDeviceToDevice(res.rx_, a.rx_);
    } else if (b.HasRx()) {
      auto res_temp = std::vector<DvView<word>>{res.RxView()};
      elem_handler_.Neg(res_temp, np, {b.RxConstView()});
    }
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Sub(res_temp, np, {a.BxConstView()}, {b.ConstView()});
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.SubConst(res_temp, np, {a.BxConstView()}, b.ConstView());
  CopyDeviceToDevice(res.ax_, a.ax_);
  if (a.HasRx()) {
    CopyDeviceToDevice(res.rx_, a.rx_);
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Pt &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, b);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.Sub(res_temp, np, {a.ConstView()}, {b.BxConstView()});
  if (b.HasRx()) {
    auto res_temp = std::vector<DvView<word>>{res.AxView(), res.RxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView(), b.RxConstView()});
  } else {
    auto res_temp = std::vector<DvView<word>>{res.AxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView()});
  }
}

template <typename word>
void Context<word>::Sub(Ct &res, const Const &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  MatchResultWith(res, b);
  res.SetNumSlots(b.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = std::vector<DvView<word>>{res.BxView()};
  elem_handler_.SubOppositeConst(res_temp, np, {b.BxConstView()},
                                 a.ConstView());
  if (b.HasRx()) {
    auto res_temp = std::vector<DvView<word>>{res.AxView(), res.RxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView(), b.RxConstView()});
  } else {
    auto res_temp = std::vector<DvView<word>>{res.AxView()};
    elem_handler_.Neg(res_temp, np, {b.AxConstView()});
  }
}

template <typename word>
void Context<word>::Neg(Ct &res, const Ct &a) const {
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Neg(res_temp, np, a.ConstViewVector());
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Ct &b) const {
  AssertSameNP(a, b);
  AssertFalse(a.HasRx() || b.HasRx(),
              "Relinearization required before Mult Ct x Ct");
  res.ModifyNP(a.GetNP());
  res.PrepareRx();
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Tensor(res_temp, np, a.ConstViewVector(0, true),
                       b.ConstViewVector(0, true));
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Pt &b) const {
  AssertSameNP(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.PMult(res_temp, np, a.ConstViewVector(), b.ConstView());
}

template <typename word>
void Context<word>::Mult(Ct &res, const Ct &a, const Const &b) const {
  AssertSameNP(a, b);
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() * b.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.MultConst(res_temp, np, a.ConstViewVector(), b.ConstView());
}

template <typename word>
bool Context<word>::IsMultUnsafeCompatible(int level1, int level2) const {
  if (level1 == level2) return true;

  int min_level = Min(level1, level2);
  int max_level = Max(level1, level2);

  NPInfo min_np = param_.LevelToNP(min_level, 0);
  NPInfo max_np = param_.LevelToNP(max_level, 0);

  return min_np.IsSubsetOf(max_np);
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Ct &b,
                               int level) const {
  AssertFalse(a.HasRx() || b.HasRx(),
              "Relinearization required before MultUnsafe Ct x Ct");
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  AssertTrue(a_np.num_aux_ == 0 && b_np.num_aux_ == 0,
             "MultUnsafe Ct x Ct should be only used for ciphertexts without "
             "aux primes");
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if ((&res == &a && a_level != level) || (&res == &b && b_level != level)) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  NPInfo res_np = param_.LevelToNP(level, 0);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Ct");

  // Actual computation
  res.ModifyNP(res_np);
  res.PrepareRx();
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.Tensor(res_temp, res_np,
                       a.ConstViewVector(a_front_ignore, true),
                       b.ConstViewVector(b_front_ignore, true));
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Pt &b,
                               int level) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if (&res == &a && a_level != level) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  int num_aux = Min(a_np.num_aux_, b_np.num_aux_);
  NPInfo res_np = param_.LevelToNP(level, num_aux);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Pt");

  // Actual computation
  if (a.HasRx()) {
    res.ModifyNP(res_np);
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(res_np);
  }
  res.SetNumSlots(Max(a.GetNumSlots(), b.GetNumSlots()));
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.PMult(res_temp, res_np, a.ConstViewVector(a_front_ignore),
                      b.ConstView(b_front_ignore));
}

template <typename word>
void Context<word>::MultUnsafe(Ct &res, const Ct &a, const Const &b,
                               int level) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();
  int a_level = param_.NPToLevel(a_np);
  int b_level = param_.NPToLevel(b_np);
  if (level == -1) {
    level = Min(a_level, b_level);
  }

  // In-place operation are not possible if the levels are different
  if (&res == &a && a_level != level) {
    Ct tmp;
    MultUnsafe(tmp, a, b, level);
    res = std::move(tmp);
    return;
  }

  // Target NPInfo
  int num_aux = Min(a_np.num_aux_, b_np.num_aux_);
  NPInfo res_np = param_.LevelToNP(level, num_aux);
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Const");

  // Actual computation
  if (a.HasRx()) {
    res.ModifyNP(res_np);
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(res_np);
  }
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() * b.GetScale());

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  auto res_temp = res.ViewVector();
  elem_handler_.MultConst(res_temp, res_np, a.ConstViewVector(a_front_ignore),
                          b.ConstView(b_front_ignore));
}

template <typename word>
void Context<word>::Permute(Ct &res, const Ct &a, int rot_dist) const {
  int num_slots = a.GetNumSlots();
  int rot_idx = rot_dist % num_slots;
  if (rot_idx < 0) rot_idx += num_slots;

  if (rot_idx == 0) {
    Copy(res, a);
    return;
  }
  // in-place operation is not supported
  if (&res == &a) {
    Ct tmp;
    Permute(tmp, a, rot_idx);
    res = std::move(tmp);
    return;
  }

  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = a.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Permute(res_temp, np, rot_idx, a.ConstViewVector());
  // elem_handler_.PermuteAccum(res_temp, np, {rot_idx}, {a.ConstViewVector()});
}

template <typename word>
void Context<word>::PermuteConjugate(Ct &res, const Ct &a) const {
  static constexpr int conj_rot_idx = -1;
  // in-place operation is not supported
  if (&res == &a) {
    Ct tmp;
    PermuteConjugate(tmp, a);
    res = std::move(tmp);
    return;
  }

  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.Permute(res_temp, np, conj_rot_idx, a.ConstViewVector());
}

template <typename word>
void Context<word>::MultImaginaryUnit(Ct &res, const Ct &a) const {
  MatchResultWith(res, a);
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale());
  NPInfo np = res.GetNP();

  auto res_temp = res.ViewVector();
  elem_handler_.MultImaginaryUnit(res_temp, np, a.ConstViewVector(),
                                  ntt_handler_.ImaginaryUnitConstView(np));
}

template <typename word>
void Context<word>::Relinearize(Ct &res, const Ct &a, const Evk &key) const {
  AssertTrue(a.HasRx(), "Relinearize requires aux");
  MultKey(res, a, key);
}

template <typename word>
void Context<word>::AdjustLevelForMultKey(int &level, const int num_q,
                                          const int num_aux) const {
  if (level == 0) {
    // This still has some issues if alpha == num_q at level 0
    if (num_aux != param_.alpha_) {
      level = -1;  // maybe...
    }
  }
  if (level == -1) {
    AssertTrue(num_aux == num_q, "Invalid setting for DTS");
  } else {
    AssertTrue(num_aux == param_.alpha_,
               "Invalid setting for MultKeyNoModDown");
  }
}

template <typename word>
void Context<word>::MultKey(Ct &res, const Ct &a, const Evk &key) const {
  NPInfo np = a.GetNP();
  int level = param_.NPToLevel(np);
  int num_aux = key.GetNP().num_aux_;
  int num_q = np.GetNumQ();
  AdjustLevelForMultKey(level, num_q, num_aux);
  const auto &mod_switcher =
      level == -1 ? GetDtSModSwitchHandler() : mod_switch_handlers_.at(level);

  Ct accum;
  MultKeyNoModDown(accum, a, key);

  // Prepare result
  res.RemoveRx();
  res.ModifyNP(np);
  res.SetScale(a.GetScale());
  res.SetNumSlots(a.GetNumSlots());

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();
  mod_switcher.ModDown(res_bx_view, accum.BxConstView());
  mod_switcher.ModDown(res_ax_view, accum.AxConstView());
}

template <typename word>
void Context<word>::MultKeyNoModDown(Ct &accum, const std::vector<Dv> &a_modup,
                                     const Ct &a_orig, const Evk &key) const {
  NPInfo a_orig_np = a_orig.GetNP();
  int level = param_.NPToLevel(a_orig_np);
  int num_main = a_orig_np.num_main_;
  int num_ter = a_orig_np.num_ter_;
  int num_aux = key.GetNP().num_aux_;
  int num_q = num_main + num_ter;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset = ((level == -1) ? 0 : (param_.GetMaxNumTer() - num_ter));

  AssertTrue(&accum != &a_orig,
             "In-place operation is not supported for MultKeyNoModDown");

  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);

  AssertTrue(key.GetBeta() >= beta && static_cast<int>(a_modup.size()) == beta,
             "Beta mismatch");

  NPInfo np(num_main, num_ter, num_aux);
  accum.RemoveRx();
  accum.ModifyNP(np);
  accum.SetScale(a_orig.GetScale());
  accum.SetNumSlots(a_orig.GetNumSlots());

  std::vector<DvView<word>> accum_views = accum.ViewVector();
  std::vector<std::vector<DvConstView<word>>> key_views;
  std::vector<DvConstView<word>> modup_view;

  for (int i = 0; i < beta; i++) {
    int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) continue;
    key_views.push_back(key.ConstViewVector(i, prime_offset));
    modup_view.push_back(a_modup.at(i).ConstView(num_aux * param_.degree_));
  }
  elem_handler_.PAccum(accum_views, np, key_views, modup_view);
}

template <typename word>
void Context<word>::MultKeyNoModDown(Ct &accum, const Ct &a,
                                     const Evk &key) const {
  NPInfo a_np = a.GetNP();
  AssertTrue(a_np.num_aux_ == 0,
             "MultKeyNoModDown is not supported for ciphertexts with p primes");
  AssertTrue(&accum != &a,
             "In-place operation is not supported for MultKeyNoModDown");

  int num_q = a_np.GetNumQ();
  int level = param_.NPToLevel(a_np);
  int num_aux = key.GetNP().num_aux_;
  AdjustLevelForMultKey(level, num_q, num_aux);
  int prime_offset =
      ((level == -1) ? 0 : (param_.GetMaxNumTer() - a_np.num_ter_));
  int padded_num_q = num_q + prime_offset;
  int beta = DivCeil(padded_num_q, num_aux);
  const auto &mod_switcher =
      level == -1 ? GetDtSModSwitchHandler() : mod_switch_handlers_.at(level);

  // Mod-up result preparation
  std::vector<Dv> mod_up_result;
  std::vector<DvView<word>> mod_up_result_view;
  for (int i = 0; i < beta; i++) {
    int prime_index_end = Min((i + 1) * num_aux, padded_num_q);
    if (prime_index_end <= prime_offset) {
      mod_up_result.emplace_back(0);
      mod_up_result_view.push_back(mod_up_result[i].View(0));
    } else {
      mod_up_result.emplace_back((num_q + num_aux) * param_.degree_);
      mod_up_result_view.push_back(
          mod_up_result[i].View(num_aux * param_.degree_));
    }
  }

  DvConstView<word> p_prod =
      (level == -1) ? p_prod_dts_.ConstView(0) : GetPProd(a_np);

  // relinearization or simple mult-key
  if (a.HasRx()) {
    mod_switcher.ModUp(mod_up_result_view, a.RxConstView());
    MultKeyNoModDown(accum, mod_up_result, a, key);
    // accum.bx_ += p_prod * a.bx_
    // accum.ax_ += p_prod * a.ax_
    DvView<word> accum_bx_view(accum.bx_.data(), num_q * param_.degree_, 0);
    DvView<word> accum_ax_view(accum.ax_.data(), num_q * param_.degree_, 0);
    std::vector<DvView<word>> caccum_res{accum_bx_view, accum_ax_view};
    std::vector<std::vector<DvConstView<word>>> src_const_views;
    src_const_views.push_back(a.ConstViewVector(0, true));
    src_const_views.push_back(accum.ConstViewVector());
    elem_handler_.CAccum(caccum_res, a_np, src_const_views, {p_prod});
  } else {
    mod_switcher.ModUp(mod_up_result_view, a.AxConstView());
    MultKeyNoModDown(accum, mod_up_result, a, key);
    // accum.bx_ += p_prod * a.bx_
    DvView<word> accum_bx_view(accum.bx_.data(), num_q * param_.degree_, 0);
    std::vector<DvView<word>> caccum_res{accum_bx_view};
    std::vector<std::vector<DvConstView<word>>> src_const_views;
    src_const_views.push_back(std::vector<DvConstView<word>>{a.BxConstView()});
    src_const_views.push_back(
        std::vector<DvConstView<word>>{accum.BxConstView()});
    elem_handler_.CAccum(caccum_res, a_np, src_const_views, {p_prod});
  }
}

template <typename word>
void Context<word>::RelinearizeRescale(Ct &res, const Ct &a,
                                       const Evk &key) const {
  AssertTrue(a.HasRx(), "RelinearizeRescale requires aux");

  int level = param_.NPToLevel(a.GetNP());

  Ct accum;
  MultKeyNoModDown(accum, a, key);

  // Prepare result
  res.RemoveRx();
  res.ModifyNP(param_.LevelToNP(level - 1));

  res.SetScale(a.GetScale() / param_.GetRescalePrimeProd(level));
  res.SetNumSlots(a.GetNumSlots());

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();
  mod_switch_handlers_.at(level).ModDownAndRescale(res_bx_view,
                                                   accum.BxConstView());
  mod_switch_handlers_.at(level).ModDownAndRescale(res_ax_view,
                                                   accum.AxConstView());
}

template <typename word>
void Context<word>::Rescale(Ct &res, const Ct &a) const {
  if (&res == &a) {
    Warn("Rescale is not adequate for in-place operations");
    Ct temp;
    Rescale(temp, a);
    res = std::move(temp);
    return;
  }
  AssertTrue(a.GetNP().num_aux_ == 0,
             "Rescale is not supported for ciphertexts "
             "with p primes");

  int level = param_.NPToLevel(a.GetNP());
  AssertTrue(level > 0, "Not enough q primes to rescale");

  if (a.HasRx()) {
    res.ModifyNP(param_.LevelToNP(level - 1));
    res.PrepareRx();
  } else {
    res.RemoveRx();
    res.ModifyNP(param_.LevelToNP(level - 1));
  }
  res.SetNumSlots(a.GetNumSlots());
  res.SetScale(a.GetScale() / param_.GetRescalePrimeProd(level));

  auto res_bx_view = res.BxView();
  auto res_ax_view = res.AxView();

  mod_switch_handlers_.at(level).Rescale(res_bx_view, a.BxConstView());
  mod_switch_handlers_.at(level).Rescale(res_ax_view, a.AxConstView());
  if (a.HasRx()) {
    auto res_rx_view = res.RxView();
    mod_switch_handlers_.at(level).Rescale(res_rx_view, a.RxConstView());
  }
}

template <typename word>
void Context<word>::HRot(Ct &res, const Ct &a, const Evk &rot_key,
                         int rot_dist) const {
  int num_slots = a.GetNumSlots();
  rot_dist %= num_slots;
  if (rot_dist < 0) rot_dist += num_slots;
  if (rot_dist == 0) {
    Warn("HRot is not necessary");
    Copy(res, a);
    return;
  }

  Ct tmp;
  MultKey(tmp, a, rot_key);
  Permute(res, tmp, rot_dist);
}

template <typename word>
void Context<word>::HConj(Ct &res, const Ct &a, const Evk &conj_key) const {
  Ct tmp;
  MultKey(tmp, a, conj_key);
  PermuteConjugate(res, tmp);
}

template <typename word>
void Context<word>::HRotAdd(Ct &res, const Ct &a, const Ct &b,
                            const Evk &rot_key, int rot_dist) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  int num_slots = Max(a.GetNumSlots(), b.GetNumSlots());
  rot_dist %= num_slots;
  if (rot_dist < 0) rot_dist += num_slots;
  res.SetNumSlots(num_slots);
  res.SetScale(a.GetScale());
  if (rot_dist == 0) {
    Warn("HRotAdd is not necessary");
    Add(res, a, b);
    return;
  }

  Ct tmp;
  MultKey(tmp, a, rot_key);

  NPInfo np = a.GetNP();
  if (b.HasRx()) {
    res.ModifyNP(np);
    res.PrepareRx();
    CopyDeviceToDevice(res.rx_, b.rx_);
  } else {
    res.RemoveRx();
    res.ModifyNP(np);
  }

  auto res_temp = res.ViewVector();
  elem_handler_.PermuteAccum(
      res_temp, np, {rot_dist},
      {tmp.ConstViewVector(), b.ConstViewVector(0, true)});
}

template <typename word>
void Context<word>::HConjAdd(Ct &res, const Ct &a, const Ct &b,
                             const Evk &conj_key) const {
  AssertSameNP(a, b);
  AssertSameScale(a, b);
  int num_slots = Max(a.GetNumSlots(), b.GetNumSlots());
  res.SetNumSlots(num_slots);
  res.SetScale(a.GetScale());

  Ct tmp;
  MultKey(tmp, a, conj_key);

  NPInfo np = a.GetNP();
  if (b.HasRx()) {
    res.ModifyNP(np);
    res.PrepareRx();
    CopyDeviceToDevice(res.rx_, b.rx_);
  } else {
    res.RemoveRx();
    res.ModifyNP(np);
  }

  auto res_temp = res.ViewVector();
  elem_handler_.PermuteAccum(
      res_temp, np, {-1}, {tmp.ConstViewVector(), b.ConstViewVector(0, true)});
}

template <typename word>
void Context<word>::HMult(Ct &res, const Ct &a, const Ct &b,
                          const Evk &mult_key, bool rescale) const {
  Mult(res, a, b);
  if (rescale) {
    RelinearizeRescale(res, res, mult_key);
  } else {
    Relinearize(res, res, mult_key);
  }
}
template <typename word>
void Context<word>::MadUnsafe(Ct &res, const Ct &a, const Const &b) const {
  const NPInfo &a_np = a.GetNP();
  const NPInfo &b_np = b.GetNP();

  NPInfo res_np = res.GetNP();
  AssertTrue(res_np.IsSubsetOf(a_np) && res_np.IsSubsetOf(b_np),
             "Incompatible levels for MultUnsafe Ct x Const");
  AssertTrue(res.GetNP().num_aux_ == 0 && a.GetNP().num_aux_ == 0,
             "MadUnsafe should be only used for q primes");
  AssertTrue(res.GetNumSlots() == a.GetNumSlots(),
             "MadUnsafe should be only used for the same number of slots");

  AssertSameScale(res, a.GetScale() * b.GetScale());

  if (a.HasRx()) {
    res.PrepareRx();
  }

  int a_front_ignore = a_np.num_ter_ - res_np.num_ter_;
  int b_front_ignore = b_np.num_ter_ - res_np.num_ter_;

  if (a.HasRx()) {
    auto res_view = res.ViewVector();
    auto res_const_view = res.ConstViewVector();
    auto a_view = a.ConstViewVector(a_front_ignore);
    auto b_view = b.ConstView(b_front_ignore);
    elem_handler_.CAccum(res_view, res.GetNP(), {a_view, res_const_view},
                         {b_view});

    // elem_handler_.MadConstThree(res.bx_, res.ax_, res.aux_, a.bx_, a.ax_,
    // a.aux_, b.cx_, num_q_primes);
  } else {
    auto res_view = res.ViewVector(0, true);
    auto res_const_view = res.ConstViewVector(0, true);
    auto a_view = a.ConstViewVector(a_front_ignore, true);
    auto b_view = b.ConstView(b_front_ignore);
    elem_handler_.CAccum(res_view, res.GetNP(), {a_view, res_const_view},
                         {b_view});
    // elem_handler_.MadConstTwo(res.bx_, res.ax_, a.bx_, a.ax_, b.cx_,
    //                           num_q_primes);
  }
}

template <typename word>
void Context<word>::LevelDown(Ct &res, const Ct &a, int target_level) const {
  Ct mult_temp, next;
  int level = param_.NPToLevel(a.GetNP());
  const Ct *prev_res = &a;
  AssertTrue(level >= target_level, "Invalid target level for LevelDown");
  for (int i = level; i > target_level; i--) {
    Mult(mult_temp, *prev_res, level_down_consts_[i]);
    Rescale(next, mult_temp);
    prev_res = &next;
  }
  Copy(res, *prev_res);
}

template <typename word>
void Context<word>::AddLowerLevelsUntil(MultiLevelCiphertext<word> &ml_ct,
                                        int min_level) const {
  if (ml_ct.Exists(min_level)) {
    return;
  }
  int max_level = ml_ct.GetMaxLevel();
  int old_min_level = ml_ct.GetMinLevel();

  AssertTrue(min_level <= max_level && min_level >= 0,
             "AddLowerLevelsUntil: Invalid level " + std::to_string(min_level));

  Ct tmp;
  for (int i = old_min_level - 1; i >= min_level; i--) {
    ml_ct.AllocateLevel(i);
    Mult(tmp, ml_ct.AtLevel(i + 1),
         MultiLevelCiphertext<word>::GetLevelDownConst(i + 1));
    Rescale(ml_ct.AtLevel(i), tmp);
  }
}

template class Context<uint32_t>;
template class Context<uint64_t>;

}  // namespace cheddar
