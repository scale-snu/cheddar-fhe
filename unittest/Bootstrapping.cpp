#include "Testbed.h"

static constexpr int num_slots = 1 << 15;

static constexpr int warm_up = 5;

TEST_P(Testbed, Bootstrap) {
  std::cout << "Preparing for bootstrapping (num_slots: " << num_slots << ")"
            << std::endl;
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg1;
  GenerateRandomMessage(msg1, num_slots);
  Ciphertext<word> ct1;

  Ciphertext<word> ct_res;
  std::vector<Complex> res;

  __ProfileStart("Boot-Basic", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap());
  __ProfileEnd("Boot-Basic");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);

  __ProfileStart("Boot-MinKS", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap(), true);
  __ProfileEnd("Boot-MinKS");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });