# Cheddar: A Swift Fully Homomorphic Encryption (FHE) GPU Library 0.1

Cheddar is a swift C++/CUDA library for the GPU execution of fully homomorphic encryption (FHE).
Cheddar targets the CKKS (Cheon-Kim-Kim-Song) FHE scheme.
Cheddar significantly improves the performance of CKKS workloads when comparing it to the reported workload performance of state-of-the-art closed-source GPU implementations:


| Implementation (GPU)      | Bootstrapping (ms) | LR training (ms/iteration) | ResNet-20 inference (s) |
|---------------------------|--------------------|----------------------------|-------------------------|
| 100× [1] (V100)           | 328                | 775                        | -                       |
| **Cheddar** (V100)        | 73.8 (4.44×)       | 79.6 (9.74×)               | -                       |
| TensorFHE [2] (A100 40GB) | 250                | 1007                       | 4.94                    |
| **Cheddar** (A100 40GB)   | 42.5 (5.88×)       | 51.4 (19.6×)               | 1.36 (3.63×)            |
| DISCC-GPU [3] (A100 80GB) | 171                | -                          | 8.58                    |
| WarpDrive [4] (A100 80GB) | 121                | 113                        | 5.88                    |
| **Cheddar** (A100 80GB)   | 40.0 (4.28/3.03×)  | 51.9 (2.18×)               | 1.32 (6.50/4.45×)       |
| **Cheddar** (H100 80GB)   | 31.2               | 40.7                       | 1.05                    |
| **Cheddar** (RTX 4090)    | 31.6               | 29.9                       | out-of-memory           |

(Cheddar parameters: [bootparam_40.json](./parameters/bootparam_40.json), PCIe GPU versions used)

Some of the **key features** of Cheddar is:
* 32-bit execution with rational rescaling (enhanced [5, 6]) support
* Optimized polynomial operation (e.g., NTT and INTT) kernels
* Operational sequence optimizations
* High-level programming interface

For the time being, this repository will provide a preview of Cheddar through a compiled library (```libcheddar.so```) and associated C++ header files.
While the entire Cheddar codebase contains more functionalities and various FHE workload implementations, the provided preview provides all the vital CKKS operations and also a highly optimized CKKS bootstrapping implementation, which should allow creating your own FHE CKKS application.

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

**Requirements**
* NVIDIA server/consumer GPU (Pascal or later) with at least 16GB of DRAM
* CUDA 11.8 or later
* Docker and nvidia-container-toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

Please use the provided [Dockerfile](./Dockerfile) (based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) Docker 
image 11.8.0-devel-ubuntu22.04) to test the library.
We recommend looking at the examples in the [unittest](./unittest/) folder and testing the correponding compiled test binaries as a starting point.

Just type the following to test CKKS bootstrapping.
```
sudo docker build -t cheddar-test .
sudo docker run --rm -it --gpus=all cheddar-test
```
You can change the last line in the [Dockerfile](./Dockerfile) to other commands for testing other functionalities.
```Dockerfile
CMD ["/cheddar/unittest/build/boot_test"]
# ---(e.g.)-->
CMD ["/cheddar/unittest/build/basic_test"]
# or
CMD ["/bin/bash"]
```


## Contact

Please use GitHub issues for any suggestions.
For other inquiries, you can take a look at our [arXiv paper](https://arxiv.org/abs/2407.13055) (to be updated) or contact the authors by e-mail:
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

When using Cheddar (or even Cheddar parameters in the [parameters folder](./parameters)), please cite the following [arXiv paper](https://arxiv.org/abs/2407.13055):
```
@article{arxiv-2024-cheddar,
  title = {Cheddar: A Swift Fully Homomorphic Encryption Library for {CUDA} {GPUs}},
  author = {Kim, Jongmin and Choi, Wonseok and Ahn, {Jung Ho}},
  journal = {arXiv preprint},
  year = {2024},
  doi = {10.48550/arXiv.2407.13055}
}
```

## References

1. Jung, Wonkyung, et al. "Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs." IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES). 2021. [Link](https://doi.org/10.46586/tches.v2021.i4.114-148)
2. Fan, Shengyu, et al. "TensorFHE: Achieving Practical Computation on Encrypted Data Using GPGPU." IEEE International Symposium on High-Performance Computer Architecture (HPCA). 2023. [Link](https://doi.org/10.1109/HPCA56546.2023.10071017)
3. Park, Jaiyoung, et al."Toward Practical Privacy-Preserving Convolutional Neural Networks Exploiting Fully Homomorphic Encryption." Workshop on Data Integrity
and Secure Cloud Computing (DISCC). 2023. [Link](https://dtrilla.github.io/discc-workshop-2023/assets/pdfs/DISCC_2023_paper_4.pdf)
4. Fan, Guang, et al. "WarpDrive: GPU-Based Fully Homomorphic Encryption Acceleration Leveraging Tensor and CUDA Cores." IEEE International Symposium on High-Performance Computer Architecture (HPCA). 2025. [Link](https://doi.org/10.1109/HPCA61900.2025.00091)
5. Samardzic, Nikola, and Daniel Sanchez. "Bitpacker: Enabling High Arithmetic Efficiency in Fully Homomorphic Encryption Accelerators." ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). 2024. [Link](https://doi.org/10.1145/3620665.3640397)
6. Cheon, Jung Hee, et al. "Grafting: Complementing RNS in CKKS." IACR Cryptology ePrint Archive. 2024. [Link](https://eprint.iacr.org/2024/1014)