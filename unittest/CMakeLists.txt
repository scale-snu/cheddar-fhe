cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cheddar_test LANGUAGES CXX)

option(ENABLE_EXTENSION "Enable extension sources" ON)

if(ENABLE_EXTENSION)
  add_compile_definitions(ENABLE_EXTENSION)
endif()
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(BUILD_TESTS OFF)
FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm
  GIT_TAG        branch-22.12
  GIT_SHALLOW
)
FetchContent_MakeAvailable(rmm)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(jsoncpp)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

find_package(CUDAToolkit 11.8 REQUIRED)
find_library(LIBTOMMATH tommath libtommath REQUIRED)

include_directories(/cheddar/include ${CUDAToolkit_INCLUDE_DIRS} 
                    ${LIBTOMMATH_INCLUDE_DIRS})
link_directories(/cheddar/lib)

add_executable(basic_test BasicTest.cpp)
target_link_libraries(basic_test
                      PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)

if (ENABLE_EXTENSION)
  add_executable(boot_test Bootstrapping.cpp)
  target_link_libraries(boot_test
                        PRIVATE cheddar rmm gtest_main nlohmann_json::nlohmann_json)
endif()

configure_file(parameters/bootparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_30.json COPYONLY)
configure_file(parameters/bootparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_35.json COPYONLY)
configure_file(parameters/bootparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40.json COPYONLY)
