#undef ENABLE_EXTENSION

#include <chrono>

#include "Testbed.h"

static constexpr int warm_up = 5;

TEST_P(Testbed, EncodeDecode) {
  std::cout << "Encode and Decode functions exist for test purposes and their "
               "performance is not a priority."
            << std::endl;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    GenerateRandomMessage(msg1);

    Plaintext<word> pt1;
    Encode(pt1, msg1, level);

    std::vector<Complex> res;
    Decode(res, pt1);
    CompareMessages(msg1, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, EncodeEncryptDecryptDecode) {
  std::cout << "Encode, Encrypt, Decrypt and Decode functions exist for test "
               "purposes and their performance is not a priority."
            << std::endl;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    GenerateRandomMessage(msg1);

    Ciphertext<word> ct1;
    EncodeAndEncrypt(ct1, msg1, level);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct1);
    CompareMessages(msg1, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtAddCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddCt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, ct2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtAddPt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddPt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtAddConst) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddConst at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtSubCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubCt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, ct2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtSubPt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubPt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtSubConst) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubConst at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, NegCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(-msg1[i]);
    }
    Ciphertext<word> ct1;
    Ciphertext<word> ct_res;

    std::string name = "NegCt at level" + std::to_string(level);
    auto prepare_cts = [&]() { EncodeAndEncrypt(ct1, msg1, level); };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Neg(ct_res, ct1);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtMultPt) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name =
        "CtMultPt (w/o rescaling) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Mult(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtMultConst) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name =
        "CtMultConst (w/o rescaling) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Mult(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, CtMultImaginaryUnit) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(Complex(-msg1[i].imag(), msg1[i].real()));
    }
    Ciphertext<word> ct1;
    Ciphertext<word> ct_res;

    std::string name = "CtMultImaginaryUnit at level" + std::to_string(level);
    auto prepare_cts = [&]() { EncodeAndEncrypt(ct1, msg1, level); };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->MultImaginaryUnit(ct_res, ct1);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, HMult) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res, ct_tmp;

    // Merged case;
    std::string name =
        "HMult(tensor + merged relin-rescale) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };
    __ProfileStart(name, warm_up, prepare_cts(););
    context_->HMult(ct_res, ct1, ct2, interface_->GetMultiplicationKey(), true);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);

    // Non-merged case;
    name = "HMult(tensor + relinearize) at level" + std::to_string(level);
    __ProfileStart(name, warm_up, prepare_cts(););
    context_->HMult(ct_tmp, ct1, ct2, interface_->GetMultiplicationKey(),
                    false);
    __ProfileEnd(name);

    DecryptAndDecode(res, ct_tmp);
    CompareMessages(true_res, res, level == param_->max_level_);

    name = "Rescale at level" + std::to_string(level);
    __ProfileStart(name, warm_up,
                   context_->HMult(ct_tmp, ct1, ct2,
                                   interface_->GetMultiplicationKey(), false););
    context_->Rescale(ct_res, ct_tmp);
    __ProfileEnd(name);

    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, HRot) {
  int num_slots = (1 << log_degree_) / 2;
  word test_rot_dist = 1234;
  interface_->PrepareRotationKey(test_rot_dist, param_->max_level_);

  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[(i + test_rot_dist) % num_slots]);
    }
    Ciphertext<word> ct1, ct_res;
    std::string name = "HRot at level" + std::to_string(level);
    __ProfileStart(name, warm_up, EncodeAndEncrypt(ct1, msg1, level););
    context_->HRot(ct_res, ct1, interface_->GetRotationKey(test_rot_dist),
                   test_rot_dist);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, HConj) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(std::conj(msg1[i]));
    }
    Ciphertext<word> ct1, ct_res;
    std::string name = "HConj at level" + std::to_string(level);
    __ProfileStart(name, warm_up, EncodeAndEncrypt(ct1, msg1, level););
    context_->HConj(ct_res, ct1, interface_->GetConjugationKey());
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
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
