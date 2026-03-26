#include "core/Container.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
Container<word>::Container(const NPInfo &num_primes /*= NPInfo{}*/)
    : num_primes_{num_primes} {}

template <typename word>
double Container<word>::GetScale() const {
  return scale_;
}

template <typename word>
void Container<word>::SetScale(double scale) {
  scale_ = scale;
}

template <typename word>
void Container<word>::SetDegree(int degree) {
  AssertTrue(IsPowOfTwo(degree), "Container degree must be a power of two");
  degree_ = degree;
}

template <typename word>
Constant<word>::Constant(const NPInfo &num_primes /*= NPInfo{}*/)
    : Base(num_primes), cx_(num_primes.GetNumTotal()) {}

template <typename word>
NPInfo Constant<word>::GetNP() const {
  int total = this->num_primes_.GetNumTotal();
  AssertTrue(static_cast<int>(cx_.size()) == total,
             "Constant num primes mismatch");
  return this->num_primes_;
}

template <typename word>
void Constant<word>::ModifyNP(const NPInfo &num_primes) {
  this->num_primes_ = num_primes;
  cx_.resize(num_primes.GetNumTotal());
}

template <typename word>
DvView<word> Constant<word>::View(int np_front_ignore /*= 0*/) {
  return cx_.View(this->num_primes_.num_aux_, np_front_ignore);
}

template <typename word>
DvConstView<word> Constant<word>::ConstView(int np_front_ignore /*= 0*/) const {
  return cx_.ConstView(this->num_primes_.num_aux_, np_front_ignore);
}

template <typename word>
Ciphertext<word>::Ciphertext(const NPInfo &num_primes /*= NPInfo{}*/,
                             bool has_rx /*= false*/)
    : Base(num_primes),
      bx_(num_primes.GetNumTotal() * Base::degree_),
      ax_(num_primes.GetNumTotal() * Base::degree_),
      rx_(has_rx ? num_primes.GetNumTotal() * Base::degree_ : 0) {}

template <typename word>
bool Ciphertext<word>::HasRx() const {
  return rx_.size() != 0;
}

template <typename word>
void Ciphertext<word>::PrepareRx() {
  rx_.resize(bx_.size());
}

template <typename word>
void Ciphertext<word>::RemoveRx() {
  rx_.resize(0);
}

template <typename word>
NPInfo Ciphertext<word>::GetNP() const {
  AssertTrue(bx_.size() == ax_.size(), "Ciphertext DV size mismatch");
  if (HasRx()) {
    AssertTrue(bx_.size() == rx_.size(), "Ciphertext DV size mismatch");
  }
  int total = this->num_primes_.GetNumTotal();
  AssertTrue(static_cast<int>(bx_.size()) == total * Base::degree_,
             "Ciphertext num primes mismatch");
  return this->num_primes_;
}

template <typename word>
void Ciphertext<word>::ModifyNP(const NPInfo &num_primes) {
  this->num_primes_ = num_primes;
  int new_size = num_primes.GetNumTotal() * Base::degree_;

  bx_.resize(new_size);
  ax_.resize(new_size);
  if (HasRx()) {
    rx_.resize(new_size);
  }
}

template <typename word>
int Ciphertext<word>::GetNumSlots() const {
  return num_slots_;
}

template <typename word>
void Ciphertext<word>::SetNumSlots(int num_slots) {
  AssertTrue(IsPowOfTwo(num_slots),
             "Ciphertext num slots must be a power of two");
  num_slots_ = num_slots;
}

template <typename word>
DvView<word> Ciphertext<word>::BxView(int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return bx_.View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> Ciphertext<word>::BxConstView(
    int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return bx_.ConstView(aux_size, front_offset);
}

template <typename word>
DvView<word> Ciphertext<word>::AxView(int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return ax_.View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> Ciphertext<word>::AxConstView(
    int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return ax_.ConstView(aux_size, front_offset);
}

template <typename word>
DvView<word> Ciphertext<word>::RxView(int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return rx_.View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> Ciphertext<word>::RxConstView(
    int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return rx_.ConstView(aux_size, front_offset);
}

template <typename word>
std::vector<DvView<word>> Ciphertext<word>::ViewVector(
    int np_front_ignore /*= 0*/, bool ignore_rx /*= false*/) {
  if (ignore_rx || !HasRx()) {
    return {BxView(np_front_ignore), AxView(np_front_ignore)};
  } else {
    return {BxView(np_front_ignore), AxView(np_front_ignore),
            RxView(np_front_ignore)};
  }
}

template <typename word>
std::vector<DvConstView<word>> Ciphertext<word>::ConstViewVector(
    int np_front_ignore /*= 0*/, bool ignore_rx /*= false*/) const {
  if (ignore_rx || !HasRx()) {
    return {BxConstView(np_front_ignore), AxConstView(np_front_ignore)};
  } else {
    return {BxConstView(np_front_ignore), AxConstView(np_front_ignore),
            RxConstView(np_front_ignore)};
  }
}

template <typename word>
Plaintext<word>::Plaintext(const NPInfo &num_primes /*= NPInfo{}*/)
    : Base(num_primes), mx_(num_primes.GetNumTotal() * Base::degree_) {}

template <typename word>
NPInfo Plaintext<word>::GetNP() const {
  int total = this->num_primes_.GetNumTotal();
  AssertTrue(static_cast<int>(mx_.size()) == total * Base::degree_,
             "Plaintext num primes mismatch");
  return this->num_primes_;
}

template <typename word>
void Plaintext<word>::ModifyNP(const NPInfo &num_primes) {
  this->num_primes_ = num_primes;
  mx_.resize(num_primes.GetNumTotal() * Base::degree_);
}

template <typename word>
int Plaintext<word>::GetNumSlots() const {
  return num_slots_;
}

template <typename word>
void Plaintext<word>::SetNumSlots(int num_slots) {
  num_slots_ = num_slots;
}

template <typename word>
DvView<word> Plaintext<word>::View(int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return mx_.View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> Plaintext<word>::ConstView(
    int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  return mx_.ConstView(aux_size, front_offset);
}

template <typename word>
EvaluationKey<word>::EvaluationKey(const NPInfo &num_primes /*= NPInfo{}*/,
                                   int beta /*= 0*/)
    : Base(num_primes) {
  AssertTrue(beta >= 0, "Negative beta given for evaluation key");
  int size = num_primes.GetNumTotal() * Base::degree_;
  for (int i = 0; i < beta; i++) {
    bx_.emplace_back(size);
    ax_.emplace_back(size);
  }
}

template <typename word>
int EvaluationKey<word>::GetBeta() const {
  AssertTrue(bx_.size() == ax_.size(), "Evk beta mismatch");
  return bx_.size();
}

template <typename word>
NPInfo EvaluationKey<word>::GetNP() const {
  int beta = GetBeta();
  int total = this->num_primes_.GetNumTotal();

  if (beta != 0) {
    auto size = bx_[0].size();
    AssertTrue(static_cast<int>(size) == total * Base::degree_,
               "Evaluation key num primes mismatch");
    for (int i = 0; i < beta; i++) {
      AssertTrue(bx_[i].size() == size, "Evk DV size mismatch");
      AssertTrue(ax_[i].size() == size, "Evk DV size mismatch");
    }
  }
  return this->num_primes_;
}

template <typename word>
void EvaluationKey<word>::ModifyNP(const NPInfo &num_primes) {
  this->num_primes_ = num_primes;
  int beta = GetBeta();
  int new_size = num_primes.GetNumTotal() * Base::degree_;
  for (int i = 0; i < beta; i++) {
    bx_[i].resize(new_size);
    ax_[i].resize(new_size);
  }
}

template <typename word>
DvView<word> EvaluationKey<word>::BxView(int index,
                                         int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  AssertTrue(index < GetBeta(), "Evk index out of range");
  return bx_.at(index).View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> EvaluationKey<word>::BxConstView(
    int index, int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  AssertTrue(index < GetBeta(), "Evk index out of range");
  return bx_.at(index).ConstView(aux_size, front_offset);
}

template <typename word>
DvView<word> EvaluationKey<word>::AxView(int index,
                                         int np_front_ignore /*= 0*/) {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  AssertTrue(index < GetBeta(), "Evk index out of range");
  return ax_.at(index).View(aux_size, front_offset);
}

template <typename word>
DvConstView<word> EvaluationKey<word>::AxConstView(
    int index, int np_front_ignore /*= 0*/) const {
  int aux_size = this->num_primes_.num_aux_ * Base::degree_;
  int front_offset = np_front_ignore * Base::degree_;
  AssertTrue(index < GetBeta(), "Evk index out of range");
  return ax_.at(index).ConstView(aux_size, front_offset);
}

template <typename word>
std::vector<DvView<word>> EvaluationKey<word>::ViewVector(
    int index, int np_front_ignore /*= 0*/) {
  return {BxView(index, np_front_ignore), AxView(index, np_front_ignore)};
}

template <typename word>
std::vector<DvConstView<word>> EvaluationKey<word>::ConstViewVector(
    int index, int np_front_ignore /*= 0*/) const {
  return {BxConstView(index, np_front_ignore),
          AxConstView(index, np_front_ignore)};
}

template class Container<uint32_t>;
template class Container<uint64_t>;
template class Constant<uint32_t>;
template class Constant<uint64_t>;
template class Ciphertext<uint32_t>;
template class Ciphertext<uint64_t>;
template class Plaintext<uint32_t>;
template class Plaintext<uint64_t>;
template class EvaluationKey<uint32_t>;
template class EvaluationKey<uint64_t>;

}  // namespace cheddar