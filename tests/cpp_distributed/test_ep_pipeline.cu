/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * EP pipeline tests — individual operations and integrated scenarios.
 *
 * Tests are structured from smallest to largest scope so that failures
 * pinpoint the broken operation before bigger tests run.
 *
 * Individual operation tests (smallest to largest scope):
 *   EPDispatchTest/PrepareAndDispatch    — prepare + dispatch; token counts + exact recv values
 *   EPCombineTest/Combine                — combine (dispatch as setup); round-trip value equality
 *   EPCombineBwdTest/CombineBwdCheck     — fwd, then combine_bwd; exact grad_expert values
 *   EPDispatchBwdTest/DispatchBwdCheck   — fwd + combine_bwd, then dispatch_bwd; exact grad_tokens
 *
 * Integrated tests:
 *   EPPipelineTest/FullForwardBackward   — fwd + bwd end-to-end NaN/Inf check
 *
 * Routing: token t on rank r → expert (r * num_local_experts + t * top_k + k) % num_experts
 * Token values: rank r, token t → all hidden dims = (r+1)*0.01 + t*0.001
 *
 * With this setup the following exact results are derivable:
 *   dispatch recv:      multiset of source-token values routed to this rank's experts
 *   combine result:     result[t] == top_k * tokens[t]  (forward combine sums
 *                       expert outputs without applying weights — caller applies them)
 *   combine_bwd:        grad_expert[slot] == d_result[t] == 0.1  (combine_bwd is a
 *                       forward dispatch of d_result; weights are NOT applied)
 *   dispatch_bwd:       grad_tokens[t] == top_k * 0.1 == 0.2  (sum over top_k of
 *                       grad_expert without weights)
 */

#include "test_ep_common.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

// ── Deterministic routing helpers ─────────────────────────────────────────────

static std::vector<int64_t> generate_topk_idx(
    int rank, int num_tokens, int top_k, int num_experts, int num_local_experts) {
  std::vector<int64_t> idx(num_tokens * top_k);
  for (int t = 0; t < num_tokens; ++t)
    for (int k = 0; k < top_k; ++k)
      idx[t * top_k + k] = (rank * num_local_experts + t * top_k + k) % num_experts;
  return idx;
}

static std::vector<nv_bfloat16> generate_tokens(int rank, int num_tokens, int hidden_dim) {
  std::vector<nv_bfloat16> v(num_tokens * hidden_dim);
  for (int t = 0; t < num_tokens; ++t) {
    float val = (rank + 1) * 0.01f + t * 0.001f;
    for (int h = 0; h < hidden_dim; ++h)
      v[t * hidden_dim + h] = __float2bfloat16(val);
  }
  return v;
}

// Replays routing for all source ranks to compute expected counts at recv_rank.
static std::vector<int32_t> expected_token_counts(
    int recv_rank, int num_processes, int num_tokens, int top_k,
    int num_experts, int num_local_experts) {
  int base = recv_rank * num_local_experts;
  std::vector<int32_t> cnt(num_local_experts, 0);
  for (int src = 0; src < num_processes; ++src) {
    auto idx = generate_topk_idx(src, num_tokens, top_k, num_experts, num_local_experts);
    for (int t = 0; t < num_tokens; ++t)
      for (int k = 0; k < top_k; ++k) {
        int64_t e = idx[t * top_k + k];
        if (e >= base && e < base + num_local_experts) ++cnt[e - base];
      }
  }
  return cnt;
}

// Returns the sorted list of BF16-rounded token values expected in the recv buffer.
// One entry per received token-slot (same token may appear for top_k>1).
static std::vector<float> expected_recv_values_sorted(
    int recv_rank, int num_processes, int num_tokens, int top_k,
    int num_experts, int num_local_experts) {
  int base = recv_rank * num_local_experts;
  std::vector<float> vals;
  for (int src = 0; src < num_processes; ++src) {
    auto idx = generate_topk_idx(src, num_tokens, top_k, num_experts, num_local_experts);
    for (int t = 0; t < num_tokens; ++t) {
      for (int k = 0; k < top_k; ++k) {
        int64_t e = idx[t * top_k + k];
        if (e >= base && e < base + num_local_experts) {
          float raw = (src + 1) * 0.01f + t * 0.001f;
          vals.push_back(__bfloat162float(__float2bfloat16(raw)));
        }
      }
    }
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// ── NaN/Inf check ─────────────────────────────────────────────────────────────

static bool check_no_nan_inf(const nv_bfloat16* dev, int count, const char* name) {
  std::vector<nv_bfloat16> h(count);
  cudaMemcpy(h.data(), dev, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
  for (int i = 0; i < count; ++i) {
    float v = __bfloat162float(h[i]);
    if (std::isnan(v)) {
      fprintf(stderr, "Rank %d: NaN in %s[%d]\n", g_process_id, name, i);
      return false;
    }
    if (std::isinf(v)) {
      fprintf(stderr, "Rank %d: Inf in %s[%d]\n", g_process_id, name, i);
      return false;
    }
  }
  return true;
}

// ── Shared forward-buffer allocation ─────────────────────────────────────────

struct EPForwardBuffers {
  int64_t*     d_topk_idx          = nullptr;
  float*       d_topk_weights      = nullptr;
  nv_bfloat16* d_tokens            = nullptr;
  int32_t*     d_token_counts      = nullptr;
  uint8_t*     d_handle_mem        = nullptr;
  nv_bfloat16* d_recv_tokens       = nullptr;
  float*       d_recv_topk_weights = nullptr;
  nv_bfloat16* d_result            = nullptr;

  size_t handle_mem_size = 0;
  size_t recv_capacity   = 0;
  int    top_k_          = 0;

  void alloc(int num_tokens, int top_k, int hidden_dim,
             int num_local_experts, int ep_size, int max_tokens_per_rank) {
    // Match the slot budget configured in test_ep_common.h (ep_size * max_tokens_per_rank * 2).
    recv_capacity = static_cast<size_t>(ep_size) * max_tokens_per_rank * 2;
    top_k_ = top_k;

    cudaMalloc(&d_topk_idx,          num_tokens * top_k         * sizeof(int64_t));
    cudaMalloc(&d_topk_weights,      num_tokens * top_k         * sizeof(float));
    cudaMalloc(&d_tokens,            num_tokens * hidden_dim    * sizeof(nv_bfloat16));
    cudaMalloc(&d_token_counts,      num_local_experts          * sizeof(int32_t));
    cudaMalloc(&d_recv_tokens,       recv_capacity * hidden_dim * sizeof(nv_bfloat16));
    cudaMalloc(&d_recv_topk_weights, recv_capacity              * sizeof(float));
    cudaMalloc(&d_result,            num_tokens * hidden_dim    * sizeof(nv_bfloat16));

    NVTEEpLayerConfig cfg{num_local_experts, top_k, /*alignment=*/0};
    handle_mem_size = nvte_ep_get_handle_mem_size(cfg);
    cudaMalloc(&d_handle_mem, handle_mem_size);
  }

  void free_all() {
    cudaFree(d_topk_idx);          d_topk_idx          = nullptr;
    cudaFree(d_topk_weights);      d_topk_weights      = nullptr;
    cudaFree(d_tokens);            d_tokens            = nullptr;
    cudaFree(d_token_counts);      d_token_counts      = nullptr;
    cudaFree(d_handle_mem);        d_handle_mem        = nullptr;
    cudaFree(d_recv_tokens);       d_recv_tokens       = nullptr;
    cudaFree(d_recv_topk_weights); d_recv_topk_weights = nullptr;
    cudaFree(d_result);            d_result            = nullptr;
  }
};

// ── Shared fixture base ───────────────────────────────────────────────────────

class EpOpTestBase : public ::testing::Test {
 protected:
  int ep_size_, num_experts_, num_local_experts_, hidden_dim_;
  int max_tokens_per_rank_, top_k_, num_tokens_;

  void SetUp() override {
    if (g_sm_major < 9)
      GTEST_SKIP() << "EP requires SM_90+ (device is SM_" << g_sm_major << "0)";
    ASSERT_GE(g_num_processes, 2);
    ASSERT_TRUE(g_ep_initialized);

    ep_size_             = g_ep_size;
    num_experts_         = g_num_experts;
    num_local_experts_   = num_experts_ / ep_size_;
    hidden_dim_          = g_hidden_dim;
    max_tokens_per_rank_ = g_max_tokens_per_rank;
    top_k_               = 2;
    num_tokens_          = 32;
  }

  void upload_inputs(const EPForwardBuffers& buf) {
    auto h_idx = generate_topk_idx(g_process_id, num_tokens_, top_k_,
                                   num_experts_, num_local_experts_);
    std::vector<float> h_w(num_tokens_ * top_k_, 1.0f / top_k_);
    auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);

    CHECK_CUDA(cudaMemcpy(buf.d_topk_idx, h_idx.data(),
                          num_tokens_ * top_k_ * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(buf.d_topk_weights, h_w.data(),
                          num_tokens_ * top_k_ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(buf.d_tokens, h_tok.data(),
                          num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  }

  NVTEEpLayerConfig layer_config() const {
    return NVTEEpLayerConfig{num_local_experts_, top_k_, /*alignment=*/0};
  }

  // Run prepare + dispatch, return total tokens received.
  int run_fwd_dispatch(const EPForwardBuffers& buf, cudaStream_t stream) {
    auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                              {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
    auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                              {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
    auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                              {(size_t)num_local_experts_}, kNVTEInt32);
    auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                              {buf.handle_mem_size}, kNVTEByte);
    auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                              {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
    auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                              {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
    auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                              {buf.recv_capacity}, kNVTEFloat32);

    (void)layer_config();
    EXPECT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                    token_counts_t.tensor, handle_mem_t.tensor,
                                    /*dispatch_output_per_expert_alignment=*/0, stream));
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                     topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<int32_t> cnt(num_local_experts_);
    cudaMemcpy(cnt.data(), buf.d_token_counts,
               num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int c : cnt) total += c;
    return total;
  }
};

// =============================================================================
// Individual operation: dispatch
// =============================================================================

/*
 * Tests prepare (InitHandle + UpdateHandle) and dispatch.
 * Verifies:
 *   1. Per-expert token counts match deterministic expectation.
 *   2. The sorted multiset of first-element values in the recv buffer exactly
 *      matches the set of source token values routed to this rank's experts.
 */
class EPDispatchTest : public EpOpTestBase {};

TEST_F(EPDispatchTest, PrepareAndDispatch) {
  EPForwardBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                            {buf.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                            {buf.recv_capacity}, kNVTEFloat32);

  // Zero recv buffer so unfilled slots are distinguishable.
  CHECK_CUDA(cudaMemset(buf.d_recv_tokens, 0,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));

  (void)layer_config();
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                  token_counts_t.tensor, handle_mem_t.tensor,
                                  /*dispatch_output_per_expert_alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 1. Verify token counts.
  std::vector<int32_t> got_counts(num_local_experts_);
  CHECK_CUDA(cudaMemcpy(got_counts.data(), buf.d_token_counts,
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  auto exp_counts = expected_token_counts(g_process_id, g_num_processes, num_tokens_, top_k_,
                                          num_experts_, num_local_experts_);
  int total_recv = 0;
  for (int i = 0; i < num_local_experts_; ++i) {
    EXPECT_EQ(got_counts[i], exp_counts[i]) << "local expert " << i;
    total_recv += exp_counts[i];
  }

  ASSERT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                   topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 2. Verify recv buffer contains exactly the expected token values (as a sorted multiset).
  // Each token slot has uniform hidden values — compare the first element of each non-zero slot.
  std::vector<nv_bfloat16> h_recv(buf.recv_capacity * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_recv.data(), buf.d_recv_tokens,
                        h_recv.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  std::vector<float> got_vals;
  for (size_t s = 0; s < buf.recv_capacity; ++s) {
    float v = __bfloat162float(h_recv[s * hidden_dim_]);
    if (v != 0.0f) got_vals.push_back(v);
  }
  std::sort(got_vals.begin(), got_vals.end());

  auto exp_vals = expected_recv_values_sorted(g_process_id, g_num_processes, num_tokens_,
                                              top_k_, num_experts_, num_local_experts_);

  ASSERT_EQ(got_vals.size(), exp_vals.size())
      << "recv slot count mismatch: got " << got_vals.size()
      << " non-zero, expected " << exp_vals.size();
  for (size_t i = 0; i < exp_vals.size(); ++i)
    EXPECT_NEAR(got_vals[i], exp_vals[i], 0.002f)
        << "recv value mismatch at sorted index " << i;

  if (g_process_id == 0)
    printf("  PrepareAndDispatch: passed (recv=%d tokens, values exact)\n", total_recv);

  CHECK_CUDA(cudaStreamDestroy(stream));
  buf.free_all();
}

// =============================================================================
// Individual operation: combine
// =============================================================================

/*
 * Tests combine in isolation (dispatch is prerequisite setup, not under test).
 * Forward combine sums expert outputs without applying topk_weights —
 * the caller is responsible for weighting expert_out beforehand. With identity
 * expert (expert_out == recv_tokens, each = source token value) and top_k copies:
 *   result[t] = sum_k expert_out[slot_k(t)] = top_k * tokens[t]
 */
class EPCombineTest : public EpOpTestBase {};

TEST_F(EPCombineTest, Combine) {
  EPForwardBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                            {buf.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                            {buf.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(buf.d_result,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);

  (void)layer_config();
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                  token_counts_t.tensor, handle_mem_t.tensor,
                                  /*dispatch_output_per_expert_alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                   topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Identity expert: pass dispatch output directly to combine.
  ASSERT_NO_THROW(nvte_ep_combine(handle_mem_t.tensor, recv_tokens_t.tensor,
                                  result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // result[t] must equal top_k * tokens[t] (forward combine is unweighted sum).
  std::vector<nv_bfloat16> h_result(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_result.data(), buf.d_result,
                        h_result.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);
  for (int t = 0; t < num_tokens_; ++t) {
    float got = __bfloat162float(h_result[t * hidden_dim_]);
    float exp = __bfloat162float(h_tok[t * hidden_dim_]) * static_cast<float>(top_k_);
    EXPECT_NEAR(got, exp, 0.005f) << "token " << t << " rank " << g_process_id;
  }

  if (g_process_id == 0)
    printf("  Combine: passed (result == top_k * tokens for all t)\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
  buf.free_all();
}

// =============================================================================
// Individual operation: combine_bwd
// =============================================================================

/*
 * Runs a full forward pass, then combine_bwd with d_result = 0.1 for all elements.
 *
 * combine_bwd is a forward dispatch of d_result (no weighting), so
 * each filled slot in grad_expert receives d_result[t] = 0.1 directly.
 * Unfilled slots must remain 0 (grad_expert is zeroed before the call).
 */
class EPCombineBwdTest : public EpOpTestBase {};

TEST_F(EPCombineBwdTest, CombineBwdCheck) {
  EPForwardBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                            {buf.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                            {buf.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(buf.d_result,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);

  (void)layer_config();
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Forward.
  ASSERT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                  token_counts_t.tensor, handle_mem_t.tensor,
                                  /*dispatch_output_per_expert_alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                   topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_combine(handle_mem_t.tensor, recv_tokens_t.tensor,
                                  result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Total received tokens (= number of filled slots in grad_expert).
  std::vector<int32_t> cnt(num_local_experts_);
  CHECK_CUDA(cudaMemcpy(cnt.data(), buf.d_token_counts,
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  int total_recv = 0;
  for (int c : cnt) total_recv += c;

  // Backward: combine_bwd with uniform d_result = 0.1.
  nv_bfloat16 *d_grad_result, *d_grad_expert;
  CHECK_CUDA(cudaMalloc(&d_grad_result,
                        (size_t)num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_expert,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));

  std::vector<nv_bfloat16> h_grad_r(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpy(d_grad_result, h_grad_r.data(),
                        h_grad_r.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  // Zero grad_expert so unfilled slots are distinguishable.
  CHECK_CUDA(cudaMemset(d_grad_expert, 0,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));

  auto grad_result_t = make_nvte_tensor(d_grad_result,
                           {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_expert_t = make_nvte_tensor(d_grad_expert,
                           {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);

  ASSERT_NO_THROW(nvte_ep_combine_bwd(handle_mem_t.tensor, grad_result_t.tensor,
                                      grad_expert_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // combine_bwd is unweighted: filled slots receive the raw d_result (0.1).
  // Unfilled slots must remain 0.
  std::vector<nv_bfloat16> h_ge(buf.recv_capacity * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_ge.data(), d_grad_expert,
                        h_ge.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  const float kExpGrad = 0.1f;
  const float kTol     = 0.002f;
  int filled = 0;
  for (size_t s = 0; s < buf.recv_capacity; ++s) {
    float v = __bfloat162float(h_ge[s * hidden_dim_]);
    if (v != 0.0f) {
      ++filled;
      EXPECT_NEAR(v, kExpGrad, kTol)
          << "grad_expert slot " << s << ": expected " << kExpGrad << ", got " << v;
    }
  }
  EXPECT_EQ(filled, total_recv)
      << "filled slots in grad_expert (" << filled
      << ") != total received tokens (" << total_recv << ")";

  if (g_process_id == 0)
    printf("  CombineBwdCheck: passed (grad_expert[filled] == 0.1, filled=%d)\n", filled);

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_grad_result));
  CHECK_CUDA(cudaFree(d_grad_expert));
  buf.free_all();
}

// =============================================================================
// Individual operation: dispatch_bwd
// =============================================================================

/*
 * Runs full forward + combine_bwd, then dispatch_bwd.
 *
 * Unweighted semantics:
 *   combine_bwd:    d_expert_out[slot] = d_result[t] = 0.1
 *   dispatch_bwd:   d_tokens[t] = sum_k d_expert_in[slot_k(t)] = top_k * 0.1 = 0.2
 *
 * Expected: grad_tokens[t] == top_k * 0.1 for all t.
 */
class EPDispatchBwdTest : public EpOpTestBase {};

TEST_F(EPDispatchBwdTest, DispatchBwdCheck) {
  EPForwardBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                            {buf.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                            {buf.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(buf.d_result,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);

  (void)layer_config();
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Forward.
  ASSERT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                  token_counts_t.tensor, handle_mem_t.tensor,
                                  /*dispatch_output_per_expert_alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                   topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_combine(handle_mem_t.tensor, recv_tokens_t.tensor,
                                  result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Backward.
  nv_bfloat16 *d_grad_result, *d_grad_expert, *d_grad_tokens;
  CHECK_CUDA(cudaMalloc(&d_grad_result,
                        (size_t)num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_expert,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_tokens,
                        (size_t)num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  std::vector<nv_bfloat16> h_grad_r(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpy(d_grad_result, h_grad_r.data(),
                        h_grad_r.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_grad_expert, 0,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));

  auto grad_result_t = make_nvte_tensor(d_grad_result,
                           {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_expert_t = make_nvte_tensor(d_grad_expert,
                           {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_tokens_t = make_nvte_tensor(d_grad_tokens,
                           {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);

  ASSERT_NO_THROW(nvte_ep_combine_bwd(handle_mem_t.tensor, grad_result_t.tensor,
                                      grad_expert_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(handle_mem_t.tensor, grad_expert_t.tensor,
                                       grad_tokens_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // grad_tokens[t] == top_k * 0.1 for all t (unweighted: sum_k 0.1).
  std::vector<nv_bfloat16> h_gt(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_gt.data(), d_grad_tokens,
                        h_gt.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const float kExpGrad = static_cast<float>(top_k_) * 0.1f;
  const float kTol     = 0.005f;
  for (int t = 0; t < num_tokens_; ++t)
    EXPECT_NEAR(__bfloat162float(h_gt[t * hidden_dim_]), kExpGrad, kTol)
        << "grad_tokens token " << t << " rank " << g_process_id;

  if (g_process_id == 0)
    printf("  DispatchBwdCheck: passed (grad_tokens == %.2f for all %d tokens)\n",
           kExpGrad, num_tokens_);

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_grad_result));
  CHECK_CUDA(cudaFree(d_grad_expert));
  CHECK_CUDA(cudaFree(d_grad_tokens));
  buf.free_all();
}

// =============================================================================
// Integrated tests
// =============================================================================

class EPPipelineTest : public EpOpTestBase {};

// ---------------------------------------------------------------------------
// Full forward + backward end-to-end NaN/Inf check.
// Combines all four operations without detailed per-stage value assertions
// (those are covered by the individual tests above).
// ---------------------------------------------------------------------------

TEST_F(EPPipelineTest, FullForwardBackward) {
  EPForwardBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  nv_bfloat16 *d_grad_result, *d_grad_expert, *d_grad_tokens;
  CHECK_CUDA(cudaMalloc(&d_grad_result,
                        (size_t)num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_expert,
                        buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_grad_tokens,
                        (size_t)num_tokens_ * hidden_dim_ * sizeof(nv_bfloat16)));

  auto topk_idx_t     = make_nvte_tensor(buf.d_topk_idx,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(buf.d_topk_weights,
                            {(size_t)num_tokens_, (size_t)top_k_}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(buf.d_token_counts,
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(buf.d_handle_mem,
                            {buf.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(buf.d_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(buf.d_recv_tokens,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_topk_weights_t = make_nvte_tensor(buf.d_recv_topk_weights,
                            {buf.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(buf.d_result,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_result_t  = make_nvte_tensor(d_grad_result,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_expert_t  = make_nvte_tensor(d_grad_expert,
                            {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto grad_tokens_t  = make_nvte_tensor(d_grad_tokens,
                            {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);

  (void)layer_config();
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Forward.
  ASSERT_NO_THROW(nvte_ep_prepare(topk_idx_t.tensor,
                                  token_counts_t.tensor, handle_mem_t.tensor,
                                  /*dispatch_output_per_expert_alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(handle_mem_t.tensor, topk_idx_t.tensor, tokens_t.tensor,
                                   topk_weights_t.tensor, recv_tokens_t.tensor, recv_topk_weights_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_combine(handle_mem_t.tensor, recv_tokens_t.tensor,
                                  result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_TRUE(check_no_nan_inf(buf.d_result, num_tokens_ * hidden_dim_, "fwd result"));

  if (g_process_id == 0) printf("  Forward: OK\n");

  // Backward.
  std::vector<nv_bfloat16> h_grad(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpy(d_grad_result, h_grad.data(),
                        h_grad.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(handle_mem_t.tensor, grad_result_t.tensor,
                                      grad_expert_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(handle_mem_t.tensor, grad_expert_t.tensor,
                                       grad_tokens_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_TRUE(check_no_nan_inf(d_grad_tokens, num_tokens_ * hidden_dim_, "grad_tokens"));

  if (g_process_id == 0) printf("  Backward: OK\n  FullForwardBackward: passed\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
  buf.free_all();
  CHECK_CUDA(cudaFree(d_grad_result));
  CHECK_CUDA(cudaFree(d_grad_expert));
  CHECK_CUDA(cudaFree(d_grad_tokens));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (!ep_bootstrap(argc, argv)) return 0;
  int ret = RUN_ALL_TESTS();
  ep_teardown();
  return ret;
}
