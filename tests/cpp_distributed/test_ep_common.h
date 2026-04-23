/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Shared TE EP test infrastructure.
 * All declarations are static or inline — safe to include once per translation unit.
 *
 * Process-level state (g_process_id, g_num_processes, …) is populated by
 * ep_bootstrap() which must be called from each test binary's main().
 *
 * Test configuration:
 *   ep_size = num_processes, 4 experts per rank (num_experts = ep_size * 4)
 *   hidden_dim = 256, max_tokens_per_rank = 64
 */
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <transformer_engine/ep.h>
#include <transformer_engine/transformer_engine.h>

// ── Error-checking macros ─────────────────────────────────────────────────────

#define CHECK_NCCL(expr)                                                          \
  do {                                                                            \
    ncclResult_t _err = (expr);                                                   \
    if (_err != ncclSuccess)                                                      \
      FAIL() << "NCCL error " << _err << ": " << ncclGetErrorString(_err);        \
  } while (false)

#define CHECK_CUDA(expr)                                                          \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess)                                                      \
      FAIL() << "CUDA error " << _err << ": " << cudaGetErrorString(_err);        \
  } while (false)

#define ASSERT_CUDA_OK(expr)                                                      \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %d: %s\n", _err, cudaGetErrorString(_err));    \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (false)

#define ASSERT_NCCL_OK(expr)                                                      \
  do {                                                                            \
    ncclResult_t _err = (expr);                                                   \
    if (_err != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL error %d: %s\n", _err, ncclGetErrorString(_err));    \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (false)

// ── Process-level state ───────────────────────────────────────────────────────

static int         g_process_id          = -1;
static int         g_num_processes       = -1;
static std::string g_uid_file;
static bool        g_use_comm            = false;

static int         g_sm_major            = -1;   // set by ep_bootstrap; -1 until then
static int         g_ep_size             = -1;
static int         g_num_experts         = -1;
static int         g_hidden_dim          = 256;
static int         g_max_tokens_per_rank = 64;
static bool        g_ep_initialized      = false;

// ── TensorHandle RAII wrapper ─────────────────────────────────────────────────

struct TensorHandle {
  NVTETensor tensor   = nullptr;
  void*      dev_ptr  = nullptr;

  ~TensorHandle() {
    if (tensor) nvte_destroy_tensor(tensor);
  }

  TensorHandle() = default;
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  TensorHandle(TensorHandle&& o) noexcept : tensor(o.tensor), dev_ptr(o.dev_ptr) {
    o.tensor = nullptr; o.dev_ptr = nullptr;
  }
  TensorHandle& operator=(TensorHandle&& o) noexcept {
    if (this != &o) {
      if (tensor) nvte_destroy_tensor(tensor);
      tensor = o.tensor; dev_ptr = o.dev_ptr;
      o.tensor = nullptr; o.dev_ptr = nullptr;
    }
    return *this;
  }
};

static TensorHandle make_nvte_tensor(void* dev_ptr,
                                     const std::vector<size_t>& shape,
                                     NVTEDType dtype) {
  TensorHandle h;
  h.dev_ptr = dev_ptr;
  h.tensor  = nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING);

  NVTEShape s;
  s.ndim = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) s.data[i] = shape[i];

  NVTEBasicTensor bt;
  bt.data_ptr = dev_ptr;
  bt.dtype    = dtype;
  bt.shape    = s;
  nvte_set_tensor_param_v2(h.tensor, kNVTERowwiseData, &bt, sizeof(bt));

  return h;
}

// ── File-based ncclUniqueId exchange ─────────────────────────────────────────

static void exchange_unique_id(ncclUniqueId* uid) {
  const size_t sz = sizeof(ncclUniqueId);

  if (g_process_id == 0) {
    ASSERT_NCCL_OK(ncclGetUniqueId(uid));
    FILE* f = fopen(g_uid_file.c_str(), "wb");
    if (!f) { fprintf(stderr, "Cannot open uid file: %s\n", g_uid_file.c_str()); exit(EXIT_FAILURE); }
    fwrite(uid, 1, sz, f);
    fclose(f);
  } else {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (true) {
      FILE* f = fopen(g_uid_file.c_str(), "rb");
      if (f) {
        fseek(f, 0, SEEK_END);
        if (static_cast<size_t>(ftell(f)) >= sz) {
          fseek(f, 0, SEEK_SET);
          size_t n = fread(uid, 1, sz, f);
          fclose(f);
          if (n == sz) break;
        } else {
          fclose(f);
        }
      }
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "Process %d: timed out waiting for uid file\n", g_process_id);
        exit(EXIT_FAILURE);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}

// ── CLI parsing ───────────────────────────────────────────────────────────────

static void ep_parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if      (a.rfind("--process-id=",  0) == 0) g_process_id    = std::stoi(a.substr(13));
    else if (a.rfind("--rank=",        0) == 0) g_process_id    = std::stoi(a.substr(7));
    else if (a.rfind("--num-processes=",0)==0)  g_num_processes = std::stoi(a.substr(16));
    else if (a.rfind("--nranks=",      0) == 0) g_num_processes = std::stoi(a.substr(9));
    else if (a.rfind("--uid-file=",    0) == 0) g_uid_file      = a.substr(11);
    else if (a == "--use-comm")                  g_use_comm      = true;
  }

  if (g_process_id < 0 || g_num_processes <= 0) {
    fprintf(stderr,
            "Usage: %s --rank=N --nranks=N [--uid-file=path] [--use-comm] [gtest flags]\n"
            "  Aliases: --process-id=N, --num-processes=N\n",
            argc > 0 ? argv[0] : "test_ep");
    exit(EXIT_FAILURE);
  }

  if (g_uid_file.empty()) {
    const char* t = getenv("TMPDIR"); if (!t) t = "/tmp";
    g_uid_file = std::string(t) + "/te_ep_uid_" + std::to_string(g_process_id);
  }
}

// ── Bootstrap / teardown ──────────────────────────────────────────────────────

// Returns false if the binary should exit without running tests (wrong SM, etc.).
static bool ep_bootstrap(int argc, char* argv[]) {
  ep_parse_args(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaSetDevice(g_process_id % device_count);

  int device, major;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  g_sm_major = major;
  if (major < 9) {
    if (g_process_id == 0)
      printf("SKIP: EP requires SM_90+ (device is SM_%d0)\n", major);
    return false;
  }
  if (g_num_processes < 2) {
    if (g_process_id == 0)
      printf("SKIP: at least 2 processes required\n");
    return false;
  }

  g_ep_size    = g_num_processes;
  g_num_experts = g_ep_size * 4;  // 4 experts per rank

  ncclUniqueId uid{};
  exchange_unique_id(&uid);

  NVTEEpGroupConfig group_config;
  group_config.ep_size             = g_ep_size;
  group_config.num_experts         = g_num_experts;
  group_config.max_tokens_per_rank = g_max_tokens_per_rank;
  group_config.hidden_dim          = g_hidden_dim;

  if (g_use_comm) {
    ncclComm_t world_comm;
    ASSERT_NCCL_OK(ncclCommInitRank(&world_comm, g_num_processes, uid, g_process_id));
    ncclComm_t ep_comm;
    ASSERT_NCCL_OK(ncclCommSplit(world_comm, /*color=*/0, g_process_id, &ep_comm, nullptr));
    ASSERT_NCCL_OK(ncclCommDestroy(world_comm));
    nvte_ep_initialize_with_comm(static_cast<void*>(ep_comm), group_config);
  } else {
    nvte_ep_initialize(reinterpret_cast<const uint8_t*>(&uid),
                       g_num_processes, g_process_id, group_config);
  }

  if (g_process_id == 0) {
    printf("EP initialized via %s: ep_size=%d num_experts=%d "
           "hidden_dim=%d max_tokens_per_rank=%d\n",
           g_use_comm ? "nvte_ep_initialize_with_comm" : "nvte_ep_initialize",
           g_ep_size, g_num_experts, g_hidden_dim, g_max_tokens_per_rank);
  }

  g_ep_initialized = true;
  return true;
}

// No EP teardown API — the singleton lives until process exit.
static void ep_teardown() {
  if (g_process_id == 0) remove(g_uid_file.c_str());
}
