#pragma once

#include <map>
#include <memory>
#include <vector>

#include "Export.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "extension/BootParameter.h"
#include "extension/EvalMod.h"
#include "extension/EvalSpecialFFT.h"

namespace cheddar {

enum class API_EXPORT BootVariant {
  kNormal,             // Normal complex bootstrapping
  kImaginaryRemoving,  // Removes the imaginary part at the end
  kMergeTwoReal        // For developers' internal use
};

/**
 * @brief BootContext class for bootstrapping. This class is used to handle all
 * the precomputed data and to create an optimized computational flow for
 * bootstrapping. Minimum key-switching (min_ks) is supported, which
 * significantly reduces the number of evaluation keys at the cost of slower
 * execution time. To understand what min_ks does, refer to Kim, Jongmin, et al.
 * "ARK: Fully Homomorphic Encryption Accelerator with Runtime Data Generation
 * and Inter-Operation Key Reuse." 2022 55th IEEE/ACM International Symposium on
 * Microarchitecture (MICRO). IEEE, 2022.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT BootContext
    : public Context<word>,
      public std::enable_shared_from_this<BootContext<word>> {
 private:
  using Base = Context<word>;
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  int log_scaleup_;
  Constant<word> scaleup_const_;

  double cts_const_;
  double stc_const_;

  DeviceVector<word> mod_max_intt_const_;

  std::map<int, EvalSpecialFFT<word>> eval_fft_;
  std::map<int, BootVariant> boot_variant_;
  std::unique_ptr<EvalMod<word>> eval_mod_;

  BootContext(const Parameter<word> &, const BootParameter &);

  int GetBootEnabledNumSlots(int num_slots) const;
  double GetCtSConst() const;
  double GetStCConst(BootVariant variant = BootVariant::kNormal) const;
  void ModUpToMax(Ct &res, const Ct &input, const EvkMap<word> &evk_map) const;
  void CoeffToSlot(Ct &res, int num_slots, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void SlotToCoeff(Ct &res, int num_slots, const Ct &input,
                   const EvkMap<word> &evk_map, bool min_ks = false) const;
  void EvaluateMod(Ct &res, const Ct &input, const Evk &mult_key) const;

  ContextPtr<word> GetContext();
  ConstContextPtr<word> GetContext() const;

 public:
  const BootParameter boot_param_;

  /**
   * @brief Creates a new instance of BootContext.
   *
   * @param param CKKS parameters
   * @param boot_param bootstrapping parameters
   * @return std::shared_ptr<BootContext<word>> a shared pointer to the new
   * BootContext instance
   */
  static std::shared_ptr<BootContext<word>> Create(
      const Parameter<word> &param, const BootParameter &boot_param);

  BootContext(const BootContext &) = delete;
  BootContext &operator=(const BootContext &) = delete;

  BootContext(BootContext &&) = default;

  virtual ~BootContext() = default;

  // To perform bootstrapping, follow the following steps.

  // 1. first prepare evalmod (only once) and special fft (for each num_slots)

  /**
   * @brief Prepares homomorphic modular reduction evaluation in a BootContext
   *
   */
  void PrepareEvalMod();

  /**
   * @brief Prepares homomorphic special FFT and IFFT evaluations in a
   * BootContext.
   *
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @param variant boot variant (BootVariant::kNormal (default) /
   * BootVariant::kImaginaryRemoving / BootVariant::kMergeTwoReal)
   */
  void PrepareEvalSpecialFFT(int num_slots,
                             BootVariant variant = BootVariant::kNormal);

  // 2. Retrieve required rotation distances for performing bootstrapping.

  /**
   * @brief Add required rotation distances to an EvkRequest. The client needs
   * to create an EvkMap based on the information in the resulting EvkRequest.
   *
   * @param req EvkRequest to add the required rotation distances
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @param min_ks whether to use minimum key-switching
   */
  void AddRequiredRotations(EvkRequest &req, int num_slots,
                            bool min_ks = false) const;

  // 3. Actual evaluation

  /**
   * @brief Perform bootstrapping. PrepareEvalMod() and PrepareEvalSpecialFFT()
   * should have been already done. Also, the client should provide all the
   * required evaluation keys in the evk_map by using the information obtained
   * from AddRequiredRotations().
   *
   * @param res bootstrapping result ciphertext
   * @param input input ciphertext
   * @param evk_map client-provided EvkMap
   * @param min_ks whether to use minimum key-switching
   */
  void Boot(Ct &res, const Ct &input, const EvkMap<word> &evk_map,
            bool min_ks = false) const;

  // Other functions...

  /**
   * @brief Checks if bootstrapping is prepared for the given number of slots.
   *
   * @param num_slots number of slots in the ciphertext to be bootstrapped
   * @return true if bootstrapping is prepared for the given number of slots
   * @return false if bootstrapping is not prepared
   */
  bool IsBootPrepared(int num_slots) const;

  /**
   * @brief Performs the trace operation. For s = start_rot_dist, and n =
   * num_accum, res = (input << s) + (input << 2s) + ... + (input << ns).
   *
   * @param res result ciphertext
   * @param start_rot_dist starting rotation amount
   * @param num_accum must be a power of 2
   * @param input input ciphertext
   * @param evk_map client-provided EvkMap
   */
  void Trace(Ct &res, int start_rot_dist, int num_accum, const Ct &input,
             const EvkMap<word> &evk_map) const;
};

}  // namespace cheddar