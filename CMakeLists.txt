cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cheddar LANGUAGES CUDA CXX)

# Using C++17 standard
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90)

# Automatically detect the CUDA architecture
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to native")
#  set (CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Auto-detect CUDA architecture")
#endif()

# dependencies
find_package(CUDAToolkit 11.8 REQUIRED)

include(FetchContent)
option(BUILD_TESTS OFF)
FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm
  GIT_TAG        branch-22.12
  GIT_SHALLOW
)
FetchContent_MakeAvailable(rmm)
message(STATUS "RMM source dir: ${rmm_SOURCE_DIR}")

# options
option(ENABLE_EXTENSION "Enable extension sources" ON)
option(BUILD_UNITTEST "Build unit tests" ON)
option(USE_GMP "Use GMP instead of libtommath" OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# source files
set(CKKS_GPU_SOURCES
  # src/core/BigInt.cpp
  src/core/Container.cpp
  src/core/Context.cpp
  src/core/DeviceVector.cpp
  src/core/Encode.cpp
  src/core/EvkMap.cpp
  src/core/EvkRequest.cpp
  src/core/MemoryPool.cpp
  src/core/MultiLevelCiphertext.cpp
  src/core/NPInfo.cpp
  src/core/ElementWise.cu
  src/core/ModSwitch.cu
  src/core/NTT.cu
  src/core/Parameter.cu
  src/UserInterface.cu
)
if(USE_GMP)
  list(APPEND CKKS_GPU_SOURCES src/core/BigInt_gmp.cpp)
else()
  list(APPEND CKKS_GPU_SOURCES src/core/BigInt_tommath.cpp)
endif()

if(ENABLE_EXTENSION)
  add_compile_definitions(ENABLE_EXTENSION)
  list(APPEND CKKS_GPU_SOURCES
    src/extension/BootContext.cpp
    src/extension/BootParameter.cpp
    src/extension/EvalMod.cpp
    src/extension/EvalPoly.cpp
    src/extension/EvalSpecialFFT.cpp
    src/extension/Hoist.cu
    src/extension/LinearTransform.cpp
    src/extension/StripedMatrix.cpp
  )
endif()

message(STATUS "CKKS_GPU_SOURCES: ${CKKS_GPU_SOURCES}")

# build library
add_library(cheddar SHARED ${CKKS_GPU_SOURCES})

if(USE_GMP)
  find_library(GMP gmp REQUIRED)
  set(MATH_LIB gmp)
  target_compile_definitions(cheddar PUBLIC USE_GMP)
else()
  find_library(LIBTOMMATH tommath libtommath REQUIRED)
  set(MATH_LIB ${LIBTOMMATH})
endif()

target_link_libraries(cheddar
  PUBLIC CUDA::cudart rmm ${MATH_LIB}
)

target_include_directories(cheddar
  PUBLIC include
)

target_compile_options(cheddar PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-O0>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:-O3>
)

# subdirectories
if (BUILD_UNITTEST)
  message(STATUS "Building unit tests")
  add_subdirectory(unittest)
endif()
