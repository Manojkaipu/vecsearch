cmake_minimum_required(VERSION 3.18)
project(vecsearch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VECSEARCH_BUILD_TESTS  "Build GoogleTest suite"     ON)
option(VECSEARCH_BUILD_PYTHON "Build pybind11 module"      OFF)
set(VECSEARCH_ARCH "native" CACHE STRING "-march value: native (fastest, local only), x86-64-v3 (portable AVX2), or empty")
option(VECSEARCH_SANITIZE     "Build with ASan/UBSan"      OFF)
option(VECSEARCH_TSAN         "Build with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)

add_library(vecsearch_core STATIC src/hnsw.cpp src/brute_force.cpp)
target_include_directories(vecsearch_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vecsearch_core PUBLIC Threads::Threads)
set_target_properties(vecsearch_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(vecsearch_core PRIVATE -Wall -Wextra)

if(VECSEARCH_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${VECSEARCH_ARCH}" HAS_MARCH_FLAG)
  if(HAS_MARCH_FLAG)
    target_compile_options(vecsearch_core PUBLIC -march=${VECSEARCH_ARCH})
  else()
    message(WARNING "-march=${VECSEARCH_ARCH} unsupported; building without SIMD")
  endif()
endif()

if(VECSEARCH_SANITIZE)
  target_compile_options(vecsearch_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(vecsearch_core PUBLIC -fsanitize=address,undefined)
endif()

if(VECSEARCH_TSAN)  # catches data races in the parallel graph build
  target_compile_options(vecsearch_core PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(vecsearch_core PUBLIC -fsanitize=thread)
endif()

if(VECSEARCH_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(vecsearch_tests
    tests/cpp/test_distance.cpp
    tests/cpp/test_brute_force.cpp
    tests/cpp/test_hnsw.cpp)
  target_link_libraries(vecsearch_tests PRIVATE vecsearch_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(vecsearch_tests)
endif()

if(VECSEARCH_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_vecsearch python/bindings.cpp)
  target_link_libraries(_vecsearch PRIVATE vecsearch_core)
  install(TARGETS _vecsearch DESTINATION vecsearch)
endif()
