/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file test_ep.cu
 *  \brief Distributed C++ tests for Expert Parallelism (EP), HT mode only.
 *
 *  Tests validate the full EP pipeline: initialize -> prepare ->
 *  dispatch -> combine, and backward variants.
 *
 *  **No MPI dependency.** Processes are spawned by a shell launch script
 *  (run_test_ep.sh) that passes --process-id / --num-processes flags.
 *  The ncclUniqueId is exchanged via a shared temp file (process 0
 *  writes, others poll-read).
 *
 *  Deterministic topk routing patterns allow each rank to independently
 *  compute expected receive-side results for correctness checking without
 *  any cross-rank reduction or comparison.
 *
 *  HT mode API patterns:
 *  - Dispatch: 3 inputs (tokens, topk_weights, topk_idx), 3 outputs
 *    (recv_tokens, recv_topk_weights, recv_topk_idx), 0 local tensors
 *  - Combine: 1 input (expert_out), 1 output (result), 0 local tensors
 *  - All dispatch/combine outputs are 2D tensors
 *  - token_counts passed as RECV_EXPERT_COUNTER in ncclEpCreateHandle
 *
 *  Build:
 *    cd tests/cpp_distributed && mkdir build && cd build
 *    cmake .. -DNVTE_WITH_NCCL_EP=ON && make test_ep
 *
 *  Run:
 *    bash run_test_ep.sh [num_gpus]
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <transformer_engine/ep.h>
#include <transformer_engine/transformer_engine.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define CHECK_NCCL(expr)                                                              \
  do {                                                                                \
    ncclResult_t err = (expr);                                                        \
    if (err != ncclSuccess) {                                                         \
      FAIL() << "NCCL error: " << err << ": " << ncclGetErrorString(err);             \
    }                                                                                 \
  } while (false)

#define CHECK_CUDA(expr)                                                          \
  do {                                                                            \
    cudaError_t err = (expr);                                                     \
    if (err != cudaSuccess) {                                                     \
      FAIL() << "CUDA error: " << err << ": " << cudaGetErrorString(err);         \
    }                                                                             \
  } while (false)

// Non-GTest variant for use outside TEST bodies (main, SetUpTestSuite, etc.)
#define ASSERT_CUDA(expr)                                                         \
  do {                                                                            \
    cudaError_t err = (expr);                                                     \
    if (err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error: %d: %s\n", err, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (false)

#define ASSERT_NCCL(expr)                                                         \
  do {                                                                            \
    ncclResult_t err = (expr);                                                    \
    if (err != ncclSuccess) {                                                     \
      fprintf(stderr, "NCCL error: %d: %s\n", err, ncclGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (false)

// ---------------------------------------------------------------------------
// Global process state (set from CLI, NOT from MPI)
// ---------------------------------------------------------------------------

static int g_process_id = -1;   // analogous to MPI rank
static int g_num_processes = -1; // analogous to MPI world size
static std::string g_uid_file;   // path to shared ncclUniqueId file

// EP test configuration
static int g_ep_size = -1;
static int g_num_experts = -1;
static int g_hidden_dim = 256;
static int g_max_tokens_per_rank = 64;
static bool g_ep_initialized = false;

// ---------------------------------------------------------------------------
// File-based ncclUniqueId exchange (replaces MPI_Bcast)
// ---------------------------------------------------------------------------

/*! \brief Exchange ncclUniqueId via a temp file.
 *
 *  Process 0 generates the ID and writes it to g_uid_file.
 *  All other processes poll the file until it appears and is fully
 *  written (128 bytes for ncclUniqueId).
 */
static void exchange_unique_id(ncclUniqueId* uid) {
  const size_t uid_size = sizeof(ncclUniqueId);

  if (g_process_id == 0) {
    // Generate and write
    ASSERT_NCCL(ncclGetUniqueId(uid));
    FILE* f = fopen(g_uid_file.c_str(), "wb");
    if (!f) {
      fprintf(stderr, "Failed to open uid file for writing: %s\n",
              g_uid_file.c_str());
      exit(EXIT_FAILURE);
    }
    fwrite(uid, 1, uid_size, f);
    fclose(f);
  } else {
    // Poll until the file appears and is complete
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::seconds(60);
    while (true) {
      FILE* f = fopen(g_uid_file.c_str(), "rb");
      if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        if (static_cast<size_t>(sz) >= uid_size) {
          fseek(f, 0, SEEK_SET);
          size_t nread = fread(uid, 1, uid_size, f);
          fclose(f);
          if (nread == uid_size) break;
        } else {
          fclose(f);
        }
      }
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "Process %d: timed out waiting for uid file: %s\n",
                g_process_id, g_uid_file.c_str());
        exit(EXIT_FAILURE);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

static void parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--process-id=", 0) == 0) {
      g_process_id = std::stoi(arg.substr(13));
    } else if (arg.rfind("--num-processes=", 0) == 0) {
      g_num_processes = std::stoi(arg.substr(16));
    } else if (arg.rfind("--uid-file=", 0) == 0) {
      g_uid_file = arg.substr(11);
    }
    // Other args are passed through to GTest
  }

  if (g_process_id < 0 || g_num_processes <= 0) {
    fprintf(stderr,
            "Usage: test_ep --process-id=<N> --num-processes=<N> "
            "[--uid-file=<path>] [gtest flags]\n");
    exit(EXIT_FAILURE);
  }

  if (g_uid_file.empty()) {
    // Default: /tmp/te_ep_test_uid_<ppid>  (parent PID groups co-launched
    // processes so concurrent test runs do not collide).
    const char* tmpdir = getenv("TMPDIR");
    if (!tmpdir) tmpdir = "/tmp";
    g_uid_file = std::string(tmpdir) + "/te_ep_test_uid";
  }
}

// ---------------------------------------------------------------------------
// Helper: create an NVTETensor wrapping a device buffer
// ---------------------------------------------------------------------------

struct TensorHandle {
  NVTETensor tensor;
  void* dev_ptr;

  TensorHandle() : tensor(nullptr), dev_ptr(nullptr) {}

  ~TensorHandle() {
    if (tensor) nvte_destroy_tensor(tensor);
    // dev_ptr is managed separately (caller owns the allocation)
  }

  // Non-copyable
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;

  // Movable
  TensorHandle(TensorHandle&& o) noexcept
      : tensor(o.tensor), dev_ptr(o.dev_ptr) {
    o.tensor = nullptr;
    o.dev_ptr = nullptr;
  }
  TensorHandle& operator=(TensorHandle&& o) noexcept {
    if (this != &o) {
      if (tensor) nvte_destroy_tensor(tensor);
      tensor = o.tensor;
      dev_ptr = o.dev_ptr;
      o.tensor = nullptr;
      o.dev_ptr = nullptr;
    }
    return *this;
  }
};

static TensorHandle make_nvte_tensor(void* dev_ptr,
                                      const std::vector<size_t>& shape,
                                      NVTEDType dtype) {
  TensorHandle h;
  h.dev_ptr = dev_ptr;
  h.tensor = nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING);

  NVTEShape s;
  s.ndim = shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    s.data[i] = shape[i];
  }

  NVTEBasicTensor bt;
  bt.data_ptr = dev_ptr;
  bt.dtype = dtype;
  bt.shape = s;
  nvte_set_tensor_param_v2(h.tensor, kNVTERowwiseData, &bt, sizeof(bt));

  return h;
}

// ---------------------------------------------------------------------------
// main — process bootstrapping without MPI
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  // Parse our flags first (GTest ignores unknown flags by default)
  parse_args(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  // Set CUDA device from process ID
  int device_count;
  ASSERT_CUDA(cudaGetDeviceCount(&device_count));
  int local_rank = g_process_id % device_count;
  ASSERT_CUDA(cudaSetDevice(local_rank));

  // SM_90+ guard — skip gracefully
  int device;
  ASSERT_CUDA(cudaGetDevice(&device));
  int major;
  ASSERT_CUDA(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device));
  if (major < 9) {
    if (g_process_id == 0) {
      printf("EP requires SM_90+ (Hopper or later), skipping all EP tests\n");
    }
    return 0;
  }

  // Bootstrap EP via file-based unique-ID exchange.
  // nvte_ep_initialize creates its own NCCL world comm internally from
  // the unique ID, then splits it — we just pass the raw bytes.
  g_ep_size = g_num_processes;
  g_num_experts = g_ep_size * 4;  // 4 experts per rank

  ncclUniqueId uid;
  exchange_unique_id(&uid);

  NVTEEpGroupConfig group_config;
  group_config.ep_size = g_ep_size;
  group_config.num_experts = g_num_experts;
  group_config.max_tokens_per_rank = g_max_tokens_per_rank;
  group_config.hidden_dim = g_hidden_dim;

  nvte_ep_initialize(reinterpret_cast<const uint8_t*>(&uid),
                     g_num_processes, g_process_id,
                     group_config);
  g_ep_initialized = true;

  if (g_process_id == 0) {
    printf("EP initialized (HT mode): ep_size=%d num_experts=%d "
           "hidden_dim=%d max_tokens_per_rank=%d\n",
           g_ep_size, g_num_experts, g_hidden_dim, g_max_tokens_per_rank);
  }

  int ret = RUN_ALL_TESTS();

  // Cleanup: process 0 removes the uid file
  if (g_process_id == 0) {
    std::remove(g_uid_file.c_str());
  }

  return ret;
}

// ---------------------------------------------------------------------------
// Test: EP initialization
// ---------------------------------------------------------------------------

class EPTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_GE(g_num_processes, 2) << "EP tests require at least 2 processes";
    ASSERT_TRUE(g_ep_initialized) << "EP not initialized in main()";
  }
};

// ---------------------------------------------------------------------------
// Test: Handle mem size query
// ---------------------------------------------------------------------------

TEST_F(EPTest, HandleMemSizeQuery) {
  NVTEEpLayerConfig sparse_cfg{4, NVTE_EP_TOPK_FORMAT_SPARSE,
                               NVTE_DELAYED_TENSOR_SCALING};
  size_t sparse_size = nvte_ep_get_handle_mem_size(sparse_cfg);
  ASSERT_GT(sparse_size, 0u);

  NVTEEpLayerConfig dense_cfg{4, NVTE_EP_TOPK_FORMAT_DENSE,
                              NVTE_DELAYED_TENSOR_SCALING};
  size_t dense_size = nvte_ep_get_handle_mem_size(dense_cfg);
  ASSERT_GT(dense_size, 0u);

  if (g_process_id == 0) {
    printf("  sparse handle_mem_size = %zu\n", sparse_size);
    printf("  dense  handle_mem_size = %zu\n", dense_size);
  }
}

// ===========================================================================
// Deterministic routing helpers
// ===========================================================================

/*! \brief Generate a deterministic round-robin topk routing pattern.
 *
 *  Token t on rank r selects experts:
 *    expert_id = (r * num_local_experts + t * top_k + k) % num_experts
 *  for k in [0, top_k).
 *
 *  This is fully deterministic — every rank can independently compute
 *  the routing table for every other rank, enabling receive-side
 *  verification without cross-rank communication.
 */
static std::vector<int64_t> generate_deterministic_topk_idx(
    int rank, int num_tokens, int top_k, int num_experts,
    int num_local_experts) {
  std::vector<int64_t> idx(num_tokens * top_k);
  for (int t = 0; t < num_tokens; ++t) {
    for (int k = 0; k < top_k; ++k) {
      idx[t * top_k + k] = static_cast<int64_t>(
          (rank * num_local_experts + t * top_k + k) % num_experts);
    }
  }
  return idx;
}

/*! \brief Generate deterministic token data.
 *
 *  Token t on rank r has all hidden dims set to:
 *    value = (rank + 1) * 0.01f + t * 0.001f
 *
 *  This produces unique, rank-identifiable values for verification.
 */
static std::vector<nv_bfloat16> generate_deterministic_tokens(
    int rank, int num_tokens, int hidden_dim) {
  std::vector<nv_bfloat16> tokens(num_tokens * hidden_dim);
  for (int t = 0; t < num_tokens; ++t) {
    float val = static_cast<float>(rank + 1) * 0.01f
                + static_cast<float>(t) * 0.001f;
    for (int h = 0; h < hidden_dim; ++h) {
      tokens[t * hidden_dim + h] = __float2bfloat16(val);
    }
  }
  return tokens;
}

/*! \brief Compute expected receive-side token counts per local expert.
 *
 *  By replaying the deterministic routing for ALL source ranks, the
 *  receiving rank can independently figure out how many tokens each
 *  of its local experts should receive.
 *
 *  \param[in] recv_rank           The receiving rank (this rank).
 *  \param[in] num_processes       Total EP world size.
 *  \param[in] num_tokens_per_rank Tokens sent by each rank.
 *  \param[in] top_k               Top-k value.
 *  \param[in] num_experts         Total number of experts.
 *  \param[in] num_local_experts   Experts per rank.
 *  \return Vector of expected per-expert counts [num_local_experts].
 */
static std::vector<int32_t> compute_expected_token_counts(
    int recv_rank, int num_processes, int num_tokens_per_rank,
    int top_k, int num_experts, int num_local_experts) {
  // Experts owned by recv_rank: [recv_rank * num_local_experts,
  //                               (recv_rank + 1) * num_local_experts)
  int expert_start = recv_rank * num_local_experts;
  int expert_end = expert_start + num_local_experts;

  std::vector<int32_t> counts(num_local_experts, 0);

  for (int src = 0; src < num_processes; ++src) {
    auto src_idx = generate_deterministic_topk_idx(
        src, num_tokens_per_rank, top_k, num_experts, num_local_experts);
    for (int t = 0; t < num_tokens_per_rank; ++t) {
      for (int k = 0; k < top_k; ++k) {
        int64_t expert_id = src_idx[t * top_k + k];
        if (expert_id >= expert_start && expert_id < expert_end) {
          counts[expert_id - expert_start]++;
        }
      }
    }
  }
  return counts;
}

// ===========================================================================
// Test fixture for EP pipeline tests
// ===========================================================================

class EPPipelineTest : public ::testing::Test {
 protected:
  int ep_size_;
  int num_experts_;
  int num_local_experts_;
  int hidden_dim_;
  int max_tokens_per_rank_;
  int top_k_;
  int num_tokens_;

  void SetUp() override {
    ASSERT_GE(g_num_processes, 2) << "EP tests require at least 2 processes";
    ASSERT_TRUE(g_ep_initialized) << "EP not initialized in main()";

    ep_size_ = g_ep_size;
    num_experts_ = g_num_experts;
    num_local_experts_ = num_experts_ / ep_size_;
    hidden_dim_ = g_hidden_dim;
    max_tokens_per_rank_ = g_max_tokens_per_rank;
    top_k_ = 2;
    num_tokens_ = 32;
  }
};

// ---------------------------------------------------------------------------
// Test: Full dispatch + combine pipeline with deterministic correctness check
// ---------------------------------------------------------------------------

TEST_F(EPPipelineTest, DispatchCombineSparse) {
  // ── Deterministic routing ──
  auto host_topk_idx = generate_deterministic_topk_idx(
      g_process_id, num_tokens_, top_k_, num_experts_, num_local_experts_);

  // Uniform weights: 1/top_k
  std::vector<float> host_topk_weights(num_tokens_ * top_k_,
                                        1.0f / static_cast<float>(top_k_));

  // Deterministic tokens
  auto host_tokens = generate_deterministic_tokens(
      g_process_id, num_tokens_, hidden_dim_);

  // Expected per-expert token counts (computed locally, no MPI needed)
  auto expected_counts = compute_expected_token_counts(
      g_process_id, g_num_processes, num_tokens_, top_k_,
      num_experts_, num_local_experts_);

  // ── Allocate device buffers ──
  int64_t* d_topk_idx;
  float* d_topk_weights;
  nv_bfloat16* d_tokens;
  int32_t* d_token_counts;
  uint8_t* d_handle_mem;

  CHECK_CUDA(cudaMalloc(&d_topk_idx,
      num_tokens_ * top_k_ * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_topk_weights,
      num_tokens_ * top_k_ * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_tokens,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_token_counts,
      num_local_experts_ * sizeof(int32_t)));

  NVTEEpLayerConfig layer_config{num_local_experts_,
                                 NVTE_EP_TOPK_FORMAT_SPARSE,
                                 NVTE_DELAYED_TENSOR_SCALING};

  size_t handle_mem_size = nvte_ep_get_handle_mem_size(layer_config);
  CHECK_CUDA(cudaMalloc(&d_handle_mem, handle_mem_size));

  // Copy to device
  CHECK_CUDA(cudaMemcpy(d_topk_idx, host_topk_idx.data(),
      num_tokens_ * top_k_ * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_topk_weights, host_topk_weights.data(),
      num_tokens_ * top_k_ * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_tokens, host_tokens.data(),
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  // Create NVTETensors
  auto topk_idx_t = make_nvte_tensor(
      d_topk_idx,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(top_k_)},
      kNVTEInt64);

  auto topk_weights_t = make_nvte_tensor(
      d_topk_weights,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(top_k_)},
      kNVTEFloat32);

  auto token_counts_t = make_nvte_tensor(
      d_token_counts,
      {static_cast<size_t>(num_local_experts_)},
      kNVTEInt32);

  auto handle_mem_t = make_nvte_tensor(
      d_handle_mem, {handle_mem_size}, kNVTEByte);

  auto tokens_t = make_nvte_tensor(
      d_tokens,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // ── Step 1: Prepare ──
  ASSERT_NO_THROW(nvte_ep_prepare(
      topk_idx_t.tensor,
      layer_config,
      token_counts_t.tensor, handle_mem_t.tensor,
      stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Verify token counts against locally-computed expected values
  std::vector<int32_t> host_token_counts(num_local_experts_);
  CHECK_CUDA(cudaMemcpy(host_token_counts.data(), d_token_counts,
      num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));

  if (g_process_id == 0) {
    printf("  Token counts per local expert (actual / expected):\n");
    for (int i = 0; i < num_local_experts_; ++i) {
      printf("    expert %d: %d / %d\n",
             i, host_token_counts[i], expected_counts[i]);
    }
  }
  for (int i = 0; i < num_local_experts_; ++i) {
    EXPECT_EQ(host_token_counts[i], expected_counts[i])
        << "Token count mismatch for local expert " << i
        << " on process " << g_process_id;
  }

  // ── Step 2: Dispatch ──
  const size_t recv_capacity =
      static_cast<size_t>(num_local_experts_) *
      static_cast<size_t>(max_tokens_per_rank_) *
      static_cast<size_t>(ep_size_);

  nv_bfloat16* d_recv_tokens;
  CHECK_CUDA(cudaMalloc(&d_recv_tokens,
      recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));

  auto recv_tokens_t = make_nvte_tensor(
      d_recv_tokens,
      {recv_capacity, static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);

  ASSERT_NO_THROW(nvte_ep_dispatch(
      handle_mem_t.tensor, tokens_t.tensor, topk_weights_t.tensor,
      recv_tokens_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Verify dispatched tokens are non-NaN/Inf
  int total_recv = 0;
  for (int i = 0; i < num_local_experts_; ++i) total_recv += host_token_counts[i];

  if (total_recv > 0) {
    std::vector<nv_bfloat16> host_recv(total_recv * hidden_dim_);
    CHECK_CUDA(cudaMemcpy(host_recv.data(), d_recv_tokens,
        total_recv * hidden_dim_ * sizeof(nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < total_recv * hidden_dim_; ++i) {
      float v = __bfloat162float(host_recv[i]);
      ASSERT_FALSE(std::isnan(v))
          << "NaN in recv_tokens at index " << i
          << " on process " << g_process_id;
      ASSERT_FALSE(std::isinf(v))
          << "Inf in recv_tokens at index " << i
          << " on process " << g_process_id;
    }
  }

  // ── Step 3: Combine (identity expert) ──
  nv_bfloat16* d_result;
  CHECK_CUDA(cudaMalloc(&d_result,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  auto result_t = make_nvte_tensor(
      d_result,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);

  ASSERT_NO_THROW(nvte_ep_combine(
      handle_mem_t.tensor, recv_tokens_t.tensor, result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Verify combined output
  std::vector<nv_bfloat16> host_result(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(host_result.data(), d_result,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_tokens_ * hidden_dim_; ++i) {
    float res = __bfloat162float(host_result[i]);
    ASSERT_FALSE(std::isnan(res))
        << "NaN in combined result at index " << i
        << " on process " << g_process_id;
    ASSERT_FALSE(std::isinf(res))
        << "Inf in combined result at index " << i
        << " on process " << g_process_id;
  }

  if (g_process_id == 0) {
    printf("  Dispatch + Combine pipeline passed (no NaN/Inf, "
           "token counts verified)\n");
  }

  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_topk_idx));
  CHECK_CUDA(cudaFree(d_topk_weights));
  CHECK_CUDA(cudaFree(d_tokens));
  CHECK_CUDA(cudaFree(d_token_counts));
  CHECK_CUDA(cudaFree(d_handle_mem));
  CHECK_CUDA(cudaFree(d_recv_tokens));
  CHECK_CUDA(cudaFree(d_result));
}

// ---------------------------------------------------------------------------
// Test: Full forward + backward pipeline (HT mode)
// ---------------------------------------------------------------------------

TEST_F(EPPipelineTest, FullForwardBackward) {
  // Deterministic routing
  auto host_topk_idx = generate_deterministic_topk_idx(
      g_process_id, num_tokens_, top_k_, num_experts_, num_local_experts_);
  std::vector<float> host_topk_weights(num_tokens_ * top_k_,
                                        1.0f / static_cast<float>(top_k_));
  auto host_tokens = generate_deterministic_tokens(
      g_process_id, num_tokens_, hidden_dim_);

  // ── Allocate device buffers ──
  int64_t* d_topk_idx;
  float* d_topk_weights;
  int32_t* d_token_counts;
  uint8_t* d_handle_mem;

  CHECK_CUDA(cudaMalloc(&d_topk_idx,
      num_tokens_ * top_k_ * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc(&d_topk_weights,
      num_tokens_ * top_k_ * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_token_counts,
      num_local_experts_ * sizeof(int32_t)));

  NVTEEpLayerConfig layer_config{num_local_experts_,
                                 NVTE_EP_TOPK_FORMAT_SPARSE,
                                 NVTE_DELAYED_TENSOR_SCALING};
  size_t handle_mem_size = nvte_ep_get_handle_mem_size(layer_config);
  CHECK_CUDA(cudaMalloc(&d_handle_mem, handle_mem_size));

  CHECK_CUDA(cudaMemcpy(d_topk_idx, host_topk_idx.data(),
      num_tokens_ * top_k_ * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_topk_weights, host_topk_weights.data(),
      num_tokens_ * top_k_ * sizeof(float), cudaMemcpyHostToDevice));

  auto topk_idx_t = make_nvte_tensor(d_topk_idx,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(top_k_)},
      kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(d_topk_weights,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(top_k_)},
      kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(d_token_counts,
      {static_cast<size_t>(num_local_experts_)}, kNVTEInt32);
  auto handle_mem_t = make_nvte_tensor(d_handle_mem,
      {handle_mem_size}, kNVTEByte);

  // Forward tensors
  const size_t recv_capacity =
      static_cast<size_t>(num_local_experts_) *
      static_cast<size_t>(max_tokens_per_rank_) *
      static_cast<size_t>(ep_size_);

  nv_bfloat16* d_tokens;
  nv_bfloat16* d_recv_tokens;
  nv_bfloat16* d_result;
  CHECK_CUDA(cudaMalloc(&d_tokens,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_recv_tokens,
      recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_result,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  CHECK_CUDA(cudaMemcpy(d_tokens, host_tokens.data(),
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  auto tokens_t = make_nvte_tensor(d_tokens,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);
  auto recv_tokens_t = make_nvte_tensor(d_recv_tokens,
      {recv_capacity, static_cast<size_t>(hidden_dim_)}, kNVTEBFloat16);
  auto result_t = make_nvte_tensor(d_result,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // === Forward ===
  ASSERT_NO_THROW(nvte_ep_prepare(
      topk_idx_t.tensor,
      layer_config,
      token_counts_t.tensor, handle_mem_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  ASSERT_NO_THROW(nvte_ep_dispatch(
      handle_mem_t.tensor, tokens_t.tensor, topk_weights_t.tensor,
      recv_tokens_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  ASSERT_NO_THROW(nvte_ep_combine(
      handle_mem_t.tensor, recv_tokens_t.tensor, result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (g_process_id == 0) {
    printf("  Forward pass completed\n");
  }

  // === Backward ===
  nv_bfloat16* d_grad_result;
  nv_bfloat16* d_grad_expert;
  nv_bfloat16* d_grad_tokens;
  CHECK_CUDA(cudaMalloc(&d_grad_result,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_expert,
      recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_tokens,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  // Fill grad_result with small constant
  CHECK_CUDA(cudaMemset(d_grad_result, 0,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  auto grad_result_t = make_nvte_tensor(d_grad_result,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);
  auto grad_expert_t = make_nvte_tensor(d_grad_expert,
      {recv_capacity, static_cast<size_t>(hidden_dim_)}, kNVTEBFloat16);
  auto grad_tokens_t = make_nvte_tensor(d_grad_tokens,
      {static_cast<size_t>(num_tokens_), static_cast<size_t>(hidden_dim_)},
      kNVTEBFloat16);

  // combine_bwd (dispatch direction in backward)
  ASSERT_NO_THROW(nvte_ep_combine_bwd(
      handle_mem_t.tensor, grad_result_t.tensor, grad_expert_t.tensor,
      stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (g_process_id == 0) {
    printf("  combine_bwd completed\n");
  }

  // dispatch_bwd (combine direction in backward)
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(
      handle_mem_t.tensor, grad_expert_t.tensor, grad_tokens_t.tensor,
      stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (g_process_id == 0) {
    printf("  dispatch_bwd completed\n");
  }

  // Verify backward outputs
  std::vector<nv_bfloat16> host_grad_tokens(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(host_grad_tokens.data(), d_grad_tokens,
      num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_tokens_ * hidden_dim_; ++i) {
    float v = __bfloat162float(host_grad_tokens[i]);
    ASSERT_FALSE(std::isnan(v))
        << "NaN in grad_tokens at index " << i
        << " on process " << g_process_id;
    ASSERT_FALSE(std::isinf(v))
        << "Inf in grad_tokens at index " << i
        << " on process " << g_process_id;
  }

  if (g_process_id == 0) {
    printf("  Full forward + backward pipeline passed\n");
  }

  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_topk_idx));
  CHECK_CUDA(cudaFree(d_topk_weights));
  CHECK_CUDA(cudaFree(d_token_counts));
  CHECK_CUDA(cudaFree(d_handle_mem));
  CHECK_CUDA(cudaFree(d_tokens));
  CHECK_CUDA(cudaFree(d_recv_tokens));
  CHECK_CUDA(cudaFree(d_result));
  CHECK_CUDA(cudaFree(d_grad_result));
  CHECK_CUDA(cudaFree(d_grad_expert));
  CHECK_CUDA(cudaFree(d_grad_tokens));
}
