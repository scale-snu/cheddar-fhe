# Cheddar: A Swift Fully Homomorphic Encryption (FHE) GPU Library 1.0

Cheddar is a swift C++/CUDA library for the GPU execution of fully homomorphic encryption (FHE).
Cheddar targets the CKKS (Cheon-Kim-Kim-Song) FHE scheme.

## Key Features
Cheddar focuses on maximizing GPU utilization and arithmetic efficiency:
* **High Arithmetic Efficiency**: Supports 32-bit execution with enhanced rational rescaling (based on [1, 2]).
* **Optimized Kernels**: Features highly tuned polynomial operation kernels, including NTT and INTT, designed for GPU architectures.
* **Instruction Optimization**: Implements operational sequence optimizations to minimize memory bottlenecks.
* **Developer Friendly**: Provides a high-level programming interface for easier integration.
* **Open Access**: Fully open-source under the MIT License.

## Security

Suppose a client-server scenario where the client (Alice) wants to offload useful computations on her private data to the server (Bob) but without exposing her private data to anyone including Bob.
Alice can encrypt her data into FHE ciphertexts and send the ciphertexts to Bob.
Bob will perform FHE operations directly on the ciphertexts without decryption and return the resulting FHE ciphertexts back to Alice.
The use of FHE guarantees the IND-CPA security, working with honest-but-curious (i.e., semi-honest) participants (Alice and Bob).
Note that, because Bob only has access to encrypted data from Alice, how Bob performs the computations does not affect the security of the model.

Cheddar only focuses on accelerating server-side (Bob's) FHE operations and does **NOT** provide any security guarantees for the client-side (Alice's) operations (e.g., encryption and decryption).
Especially, the client-side functionalities provided in [include/UserInterface.h](./include/UserInterface.h) or  [include/Random.h](./include/Random.h) should only be used for test purposes.
Also, the security of an FHE scheme relies on an approriate choice of parameters.
Consult security experts if you are unsure of the parameter choice.

This project is provided "as is" without any warranty of any kind, expressed or implied.
The authors shall not be held liable for any damages or issues arising from the use of this code, including but not limited to security vulnerabilities, data loss, or system failures.
Use at your own risk.

## How to Start

### Requirements
> [!IMPORTANT]
>  * NVIDIA server/consumer GPU (Pascal or later) with at least 16GB of DRAM
>  * CUDA 11.8 or later
>  * CMake version 3.24 or greater.

### Install Dependencies (Ubuntu)

You can install required dependencies via `apt`:

```bash
sudo apt update
sudo apt install -y build-essential libgmp-dev
```

> [!NOTE]
> * `libgmp-dev` is required when `USE_GMP=ON`.
> * Ubuntu's default cmake package may be older than 3.24. Install a newer version manually if needed.

### Cheddar Compilation

Configure and build Cheddar:
```bash
cmake -S $PATH_TO_ROOT_DIR -B $PATH_TO_BUILD_DIR
cmake --build $PATH_TO_BUILD_DIR -j
```
You can customize the build by passing options with -D:
```bash
cmake -S $PATH_TO_ROOT_DIR -B $PATH_TO_BUILD_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GMP=ON \
  -DBUILD_UNITTEST=ON \
  -DENABLE_EXTENSION=ON
```

Explanation of each option. The default value for each option is denoted in **bold** under the **Values** column:
| CMake Option       | Values                             | Description                                                 |
| ------------------ | ---------------------------------- | ----------------------------------------------------------- |
| `CMAKE_BUILD_TYPE` | **Release**, Debug, RelWithDebInfo | Select the compilation build type.                          |
| `USE_GMP`          | ON / **OFF**                       | Use GMP for high-precision arithmetic instead of libtommath |
| `BUILD_UNITTEST`   | **ON** / OFF                       | Build unit tests                                            |
| `ENABLE_EXTENSION` | **ON** / OFF                       | Enable extension sources                                    |



## Contact

Please use GitHub issues for any suggestions.
For other inquiries, you can take a look at our [ASPLOS 2026 paper](https://doi.org/10.1145/3760250.3762223) (or [arXiv version](https://arxiv.org/abs/2407.13055)) or contact the authors by e-mail:
* Jongmin Kim (firstname.lastname@snu.ac.kr)
* Wonseok Choi (firstname.lastname@snu.ac.kr)

## License and Citing

See the [License](./LICENSE).
Cheddar (all the files in this repository) is licensed under the MIT License.

Cheddar dynamically links the following third-party libraries:
* NVIDIA CUDA Runtime library (cudart), which is provided under the NVIDIA CUDA Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/index.html
* RMM (licensed under the Apache 2.0 License): https://github.com/rapidsai/rmm
* libtommath (public domain software): https://github.com/libtom/libtommath
* GoogleTest (licensed under the BSD 3-Clause License): https://github.com/google/googletest
* JsonCpp (licensed under the MIT License / public domain software): https://github.com/open-source-parsers/jsoncpp
* GMP (licensed under the LGPL v3): https://gmplib.org/

When using Cheddar (or even Cheddar parameters in the [parameters folder](./parameters)), please cite the following paper:
```
@inproceedings{asplos-2026-cheddar,
  author = {Choi, Wonseok and Kim, Jongmin and Ahn, Jung Ho},
  title = {Cheddar: {A} Swift Fully Homomorphic Encryption Library Designed for {GPU} Architectures},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  year = {2025},
  url = {https://doi.org/10.1145/3760250.3762223},
  doi = {10.1145/3760250.3762223}
}
```

## References

1. Samardzic, Nikola, and Daniel Sanchez. "Bitpacker: Enabling High Arithmetic Efficiency in Fully Homomorphic Encryption Accelerators." ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). 2024. [Link](https://doi.org/10.1145/3620665.3640397)
2. Cheon, Jung Hee, et al. "Grafting: Complementing RNS in CKKS." IACR Cryptology ePrint Archive. 2024. [Link](https://eprint.iacr.org/2024/1014)
