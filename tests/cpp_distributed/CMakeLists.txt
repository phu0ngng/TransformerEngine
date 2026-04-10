# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90)
  endif()
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine_distributed_tests LANGUAGES CUDA CXX)

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "python3 -c 'import transformer_engine as te; print(te.__file__)'"
                    OUTPUT_VARIABLE TE_LIB_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(TE_LIB_PATH ${TE_LIB_FILE} DIRECTORY)
endif()

find_library(TE_LIB NAMES transformer_engine PATHS "${TE_LIB_PATH}/.." ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)

message(STATUS "Found transformer_engine library: ${TE_LIB}")
include_directories(../../transformer_engine/common/include)
include_directories(../../transformer_engine/common)
include_directories(../../transformer_engine)
include_directories(${CMAKE_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)

add_executable(test_comm_gemm
               test_comm_gemm.cu
               ../cpp/test_common.cu)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_library(NCCL_LIB
             NAMES nccl libnccl
             PATH_SUFFIXES lib
             REQUIRED)
target_include_directories(test_comm_gemm PRIVATE ${MPI_CXX_INCLUDE_PATH} $ENV{CUBLASMP_HOME}/include)
target_link_libraries(test_comm_gemm PUBLIC CUDA::cuda_driver CUDA::cudart GTest::gtest ${TE_LIB} CUDA::nvrtc MPI::MPI_CXX ${NCCL_LIB} OpenMP::OpenMP_CXX)

include(GoogleTest)
gtest_discover_tests(test_comm_gemm DISCOVERY_TIMEOUT 600)

# ── EP distributed tests (requires NVTE_WITH_NCCL_EP, HT mode) ──────────────
# No MPI dependency — processes are spawned by run_test_ep.sh with
# --process-id / --num-processes flags.  ncclUniqueId exchange uses a
# shared temp file (see test_ep.cu for details).
option(NVTE_WITH_NCCL_EP "Enable NCCL EP tests" OFF)

if(NVTE_WITH_NCCL_EP)
  find_library(NCCL_EP_LIB
               NAMES nccl_ep libnccl_ep
               HINTS $ENV{NCCL_EP_HOME}/lib $ENV{NCCL_EP_DIR}/lib
               PATH_SUFFIXES lib lib64
               REQUIRED)

  # Primary: use the 3rdparty/nccl_ep submodule for headers.
  # Fallback: NCCL_EP_HOME/NCCL_EP_DIR environment variables.
  set(NCCL_EP_SUBMODULE_INCLUDE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl_ep/contrib/nccl_ep/include")

  if(EXISTS "${NCCL_EP_SUBMODULE_INCLUDE}/nccl_ep.h")
    set(NCCL_EP_INCLUDE_DIR "${NCCL_EP_SUBMODULE_INCLUDE}")
    message(STATUS "EP test: NCCL EP headers from submodule: ${NCCL_EP_INCLUDE_DIR}")
  else()
    find_path(NCCL_EP_INCLUDE_DIR
              NAMES nccl_ep.h
              HINTS $ENV{NCCL_EP_HOME}/include $ENV{NCCL_EP_DIR}/include
              REQUIRED)
    message(STATUS "EP test: NCCL EP headers from external: ${NCCL_EP_INCLUDE_DIR}")
  endif()

  add_executable(test_ep
                 test_ep.cu)

  target_include_directories(test_ep PRIVATE
      ${NCCL_EP_INCLUDE_DIR}
      ../../transformer_engine/common/include
      ../../transformer_engine/common)

  target_link_libraries(test_ep PUBLIC
      CUDA::cuda_driver
      CUDA::cudart
      GTest::gtest
      ${TE_LIB}
      ${NCCL_LIB}
      ${NCCL_EP_LIB})

  # Do NOT use gtest_discover_tests — test_ep requires multi-process
  # launch via run_test_ep.sh, not direct single-process execution.
  message(STATUS "EP distributed test enabled: ${NCCL_EP_LIB}")
endif()
