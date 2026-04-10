/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nccl_ep_backend.cpp
 *  \brief NCCLEPBackend implementation (HT mode only).
 *
 *  Wraps ncclEpCreateGroup, ncclEpCreateHandle, ncclEpDispatch,
 *  ncclEpCombine, and their destroy counterparts. Each per-step op
 *  creates ephemeral ncclNDTensor_t handles around user-provided
 *  buffers — no allocations, negligible overhead.
 *
 *  HT mode API patterns (different from LL mode):
 *  - ncclEpCreateHandle: accepts optional RECV_EXPERT_COUNTER local tensor
 *  - ncclEpDispatch (forward): 3 inputs (tokens, topk_weights, topk_idx),
 *    3 outputs (recv_tokens, recv_topk_weights, recv_topk_idx), 0 local tensors
 *  - ncclEpDispatch (backward/combine_bwd): 1 input (grad), 1 output (result),
 *    0 local tensors (no topk_weights in backward direction)
 *  - ncclEpCombine: 1 input (expert_out), 1 output (result), 0 local tensors
 *  - All dispatch/combine outputs are 2D tensors
 */

#include "nccl_ep_backend.h"  // includes ep_backend.h -> ep.h -> transformer_engine.h
#include "../util/logging.h"

namespace transformer_engine {
namespace ep {

// ---------------------------------------------------------------------------
// handle_mem header layout
// ---------------------------------------------------------------------------

/*! \brief Structured header stored at the start of handle_mem.
 *
 *  This header caches everything that dispatch/combine/bwd ops need
 *  to call the NCCL EP API without re-extracting from the original
 *  NVTETensor arguments.
 *
 *  topk_idx data pointer and shape are cached here because dispatch
 *  needs them as an input tensor.  topk_weights are passed directly
 *  to dispatch() as an NVTETensor (not cached in prepare).
 */
struct HandleMemHeader {
  ncclEpHandle_t handle;          ///< NCCL EP handle (created in prepare)
  NVTEEpLayerConfig layer_config; ///< Per-layer config
  void* topk_idx_data;            ///< Device pointer to topk_idx [T, top_k] int64
  void* token_counts_data;        ///< Device pointer to token_counts [num_local_experts] int32
  unsigned int num_tokens;        ///< Number of tokens (T)
  unsigned int top_k;             ///< Top-k value
  unsigned int num_local_experts; ///< Number of local experts
  unsigned int num_recv_tokens;   ///< Number of tokens received after dispatch
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ncclDataType_t NCCLEPBackend::nvte_dtype_to_nccl(NVTEDType dtype) {
  switch (dtype) {
    case kNVTEFloat32:   return ncclFloat32;
    case kNVTEFloat16:   return ncclFloat16;
    case kNVTEBFloat16:  return ncclBfloat16;
    case kNVTEInt32:     return ncclInt32;
    case kNVTEInt64:     return ncclInt64;
    case kNVTEByte:      return ncclUint8;
    case kNVTEFloat8E4M3: return ncclFloat8e4m3;
    case kNVTEFloat8E5M2: return ncclFloat8e5m2;
    default:
      NVTE_ERROR("Unsupported NVTEDType for NCCL EP conversion: ",
                 static_cast<int>(dtype));
  }
  return ncclFloat32;  // unreachable
}

ncclNDTensor_t NCCLEPBackend::make_tensor(void* data, unsigned int ndim,
                                           ncclDataType_t datatype,
                                           ncclEpTensorTag_t tag,
                                           unsigned int size0,
                                           unsigned int size1,
                                           unsigned int size2,
                                           unsigned int size3,
                                           unsigned int size4) {
  NVTE_CHECK(ep_group_ != nullptr, "NCCLEPBackend not initialized");
  ncclNDTensor_t tensor;
  NVTE_CHECK_NCCL(ncclEpTensorCreate(
      ep_group_, &tensor, ndim, datatype, tag,
      data, size0, size1, size2, size3, size4));
  return tensor;
}

void NCCLEPBackend::destroy_tensor(ncclNDTensor_t tensor) {
  if (tensor != nullptr && ep_group_ != nullptr) {
    NVTE_CHECK_NCCL(ncclEpTensorDestroy(ep_group_, tensor));
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

NCCLEPBackend::~NCCLEPBackend() {
  if (ep_group_ != nullptr) {
    // Best-effort cleanup. Use default stream for teardown.
    ncclEpGroupDestroy(ep_group_, 0);
    ep_group_ = nullptr;
  }
}

void NCCLEPBackend::init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(!initialized_, "NCCLEPBackend already initialized");

  group_config_ = group_config;

  // Configure the NCCL EP group — HT mode only
  ncclEpGroupConfig_t cfg;
  cfg.version = 1;
  cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
  cfg.num_experts = static_cast<unsigned int>(group_config.num_experts);
  cfg.max_tokens_per_rank = static_cast<unsigned int>(group_config.max_tokens_per_rank);
  cfg.token_size_bytes = static_cast<unsigned int>(
      group_config.hidden_dim * sizeof(nv_bfloat16));
  // NCCL_EP_AUTO lets NCCL EP choose the RDMA buffer size internally.
  // For HT mode, there is no get_low_latency_rdma_size_hint equivalent;
  // NCCL_EP_AUTO is the standard approach.
  cfg.rdma_buffer_size = NCCL_EP_AUTO;
  cfg.num_qp_per_rank = NCCL_EP_AUTO;
  cfg.num_channels = NCCL_EP_AUTO;

  // Create the EP group — this is a collective call
  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));
  NVTE_CHECK_NCCL(ncclEpCreateGroup(&ep_group_, ep_comm, &cfg, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));

  // Destroy the EP sub-comm — the EP group has its own copy.
  // The EP sub-comm was split from world_comm by EPManager; we no
  // longer need it after the group is created.
  NVTE_CHECK_NCCL(ncclCommDestroy(ep_comm));

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Handle mem size query
// ---------------------------------------------------------------------------

size_t NCCLEPBackend::get_handle_mem_size(NVTEEpLayerConfig layer_config) {
  NVTE_CHECK(initialized_, "NCCLEPBackend not initialized");

  // handle_mem stores:
  //   - HandleMemHeader struct (handle pointer, cached metadata)
  //   - No additional routing buffer space needed — NCCL EP HT mode
  //     manages its own internal buffers. The handle_mem is primarily
  //     a carrier for the HandleMemHeader.
  //
  // We add generous padding for future extensibility and alignment.

  const size_t header_size = sizeof(HandleMemHeader);

  // Round up to 128-byte alignment
  size_t total = ((header_size + 127) / 128) * 128;

  return total;
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

void NCCLEPBackend::prepare(void* handle_mem,
                             const NVTETensor topk_idx,
                             NVTETensor token_counts,
                             NVTEEpLayerConfig layer_config,
                             cudaStream_t stream) {
  NVTE_CHECK(initialized_, "NCCLEPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  hdr->layer_config = layer_config;

  // Extract topk_idx shape and data
  NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
  void* idx_data = nvte_tensor_data(topk_idx);
  NVTE_CHECK(idx_data != nullptr, "topk_idx data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(idx_shape.data[0]);
  unsigned int top_k = 1;
  if (idx_shape.ndim > 1) {
    top_k = static_cast<unsigned int>(idx_shape.data[1]);
  }

  hdr->num_tokens = num_tokens;
  hdr->top_k = top_k;
  hdr->num_local_experts = static_cast<unsigned int>(layer_config.num_local_experts);

  // Cache topk_idx data pointer for dispatch to use as an input.
  // topk_weights are passed directly to dispatch() — not cached here.
  hdr->topk_idx_data = idx_data;
  hdr->token_counts_data = (token_counts != nullptr) ?
      nvte_tensor_data(token_counts) : nullptr;

  // Create NCCL EP tensor for topk_idx
  // NCCL EP requires topk_idx to be ncclInt64.
  ncclNDTensor_t nccl_topk_idx;
  if (layer_config.topk_format == NVTE_EP_TOPK_FORMAT_SPARSE) {
    nccl_topk_idx = make_tensor(idx_data, 2, ncclInt64,
                                NCCL_EP_TENSOR_TAG_TOPK_IDX,
                                num_tokens, top_k);
  } else {
    // DENSE: [T, E] float32
    unsigned int num_experts_dim = static_cast<unsigned int>(idx_shape.data[1]);
    nccl_topk_idx = make_tensor(idx_data, 2, ncclFloat32,
                                NCCL_EP_TENSOR_TAG_TOPK_IDX,
                                num_tokens, num_experts_dim);
  }

  // In HT mode, ncclEpCreateHandle accepts an optional RECV_EXPERT_COUNTER
  // local tensor. We pass token_counts so NCCL EP populates it during
  // the metadata exchange.
  ncclNDTensor_t handle_local_tensors[1] = {nullptr};
  unsigned int handle_num_local_tensors = 0;

  if (hdr->token_counts_data != nullptr) {
    handle_local_tensors[0] = make_tensor(
        hdr->token_counts_data, 1, ncclInt32,
        NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
        hdr->num_local_experts);
    handle_num_local_tensors = 1;
  }

  // Create ncclEpHandle_t — performs metadata exchange in HT mode.
  ncclEpHandle_t handle;

  bool use_fp8 = (layer_config.scaling_mode != NVTE_DELAYED_TENSOR_SCALING);

  NVTE_CHECK_NCCL(ncclEpCreateHandle(
      &handle, ep_group_, nccl_topk_idx,
      handle_local_tensors, handle_num_local_tensors,
      nullptr,            // config — reserved, must be NULL
      stream,
      use_fp8));

  // Query the actual number of received tokens for HT mode.
  // This is needed to correctly size dispatch output tensors.
  unsigned int num_recv_tokens = 0;
  NVTE_CHECK_NCCL(ncclEpHandleGetNumRecvTokens(handle, &num_recv_tokens));
  hdr->num_recv_tokens = num_recv_tokens;

  // Store the handle pointer in handle_mem header
  hdr->handle = handle;

  // Cleanup tensor handles (handle-only, not the underlying data)
  destroy_tensor(nccl_topk_idx);
  if (handle_num_local_tensors > 0) {
    destroy_tensor(handle_local_tensors[0]);
  }
}

void NCCLEPBackend::dispatch(void* handle_mem,
                              const NVTETensor tokens,
                              const NVTETensor topk_weights,
                              NVTETensor recv_tokens,
                              cudaStream_t stream) {
  NVTE_CHECK(initialized_, "NCCLEPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  ncclEpHandle_t handle = hdr->handle;
  NVTE_CHECK(handle != nullptr, "EP handle is null — was prepare() called?");

  // Extract token tensor metadata
  NVTEShape tok_shape = nvte_tensor_shape(tokens);
  void* tok_data = nvte_tensor_data(tokens);
  NVTEDType tok_dtype = nvte_tensor_type(tokens);
  NVTE_CHECK(tok_data != nullptr, "tokens data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(tok_shape.data[0]);
  const unsigned int hidden_dim = static_cast<unsigned int>(tok_shape.data[1]);

  // Build input tensor list.
  // Forward dispatch: 3 inputs (tokens, topk_weights, topk_idx).
  // Backward (combine_bwd): 1 input (grad tokens only, no weights/idx).
  ncclNDTensor_t nccl_tokens_in = make_tensor(
      tok_data, 2, nvte_dtype_to_nccl(tok_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      num_tokens, hidden_dim);

  // Check whether topk_weights was provided (forward) or not (backward).
  void* weights_data = (topk_weights != nullptr) ?
      nvte_tensor_data(topk_weights) : nullptr;
  const bool is_forward = (weights_data != nullptr);

  ncclNDTensor_t nccl_topk_weights_in = nullptr;
  ncclNDTensor_t nccl_topk_idx_in = nullptr;
  unsigned int num_inputs = 1;

  if (is_forward) {
    nccl_topk_weights_in = make_tensor(
        weights_data, 2, ncclFloat32,
        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
        hdr->num_tokens, hdr->top_k);

    nccl_topk_idx_in = make_tensor(
        hdr->topk_idx_data, 2, ncclInt64,
        NCCL_EP_TENSOR_TAG_TOPK_IDX,
        hdr->num_tokens, hdr->top_k);
    num_inputs = 3;
  }

  const ncclNDTensor_t inputs[] = {nccl_tokens_in, nccl_topk_weights_in, nccl_topk_idx_in};

  // Build output tensor list.
  NVTEShape recv_shape = nvte_tensor_shape(recv_tokens);
  void* recv_data = nvte_tensor_data(recv_tokens);
  NVTEDType recv_dtype = nvte_tensor_type(recv_tokens);
  NVTE_CHECK(recv_data != nullptr, "recv_tokens data must not be null");

  const unsigned int num_recv_tokens = hdr->num_recv_tokens;

  ncclNDTensor_t nccl_tokens_out = make_tensor(
      recv_data, 2, nvte_dtype_to_nccl(recv_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(recv_shape.data[0]),
      static_cast<unsigned int>(recv_shape.data[1]));

  // Forward: 3 outputs (recv_tokens, recv_topk_weights, recv_topk_idx).
  // Backward: 1 output (recv grad tokens only).
  ncclNDTensor_t nccl_recv_topk_weights = nullptr;
  ncclNDTensor_t nccl_recv_topk_idx = nullptr;
  unsigned int num_outputs = 1;

  if (is_forward) {
    // NCCL EP-owned output tensors for recv_topk_weights and recv_topk_idx.
    // These are needed by NCCL EP internally for the combine phase.
    // We let NCCL EP allocate the memory (data=nullptr).
    nccl_recv_topk_weights = make_tensor(
        nullptr, 2, ncclFloat32,
        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
        num_recv_tokens, hdr->top_k);

    nccl_recv_topk_idx = make_tensor(
        nullptr, 2, ncclInt64,
        NCCL_EP_TENSOR_TAG_TOPK_IDX,
        num_recv_tokens, hdr->top_k);
    num_outputs = 3;
  }

  const ncclNDTensor_t outputs[] = {nccl_tokens_out, nccl_recv_topk_weights, nccl_recv_topk_idx};

  // Dispatch configuration
  ncclEpDispatchConfig_t dispatch_cfg;
  dispatch_cfg.round_scales = 0;

  // HT mode: no local tensors for dispatch
  NVTE_CHECK_NCCL(ncclEpDispatch(
      handle,
      inputs, num_inputs,
      outputs, num_outputs,
      nullptr, 0,        // no local tensors in HT mode
      0,                  // send_only = false
      &dispatch_cfg,
      stream));

  // Complete the dispatch operation.
  NVTE_CHECK_NCCL(ncclEpComplete(handle, nullptr, stream));

  destroy_tensor(nccl_tokens_in);
  if (nccl_topk_weights_in) destroy_tensor(nccl_topk_weights_in);
  if (nccl_topk_idx_in) destroy_tensor(nccl_topk_idx_in);
  destroy_tensor(nccl_tokens_out);
  if (nccl_recv_topk_weights) destroy_tensor(nccl_recv_topk_weights);
  if (nccl_recv_topk_idx) destroy_tensor(nccl_recv_topk_idx);
}

void NCCLEPBackend::combine(void* handle_mem,
                              const NVTETensor expert_out,
                              NVTETensor result,
                              cudaStream_t stream) {
  NVTE_CHECK(initialized_, "NCCLEPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  ncclEpHandle_t handle = hdr->handle;
  NVTE_CHECK(handle != nullptr, "EP handle is null — was prepare() called?");

  // Expert output tensor — 2D in HT mode [num_recv_tokens, hidden_dim]
  NVTEShape exp_shape = nvte_tensor_shape(expert_out);
  void* exp_data = nvte_tensor_data(expert_out);
  NVTEDType exp_dtype = nvte_tensor_type(expert_out);
  NVTE_CHECK(exp_data != nullptr, "expert_out data must not be null");

  ncclNDTensor_t nccl_expert_in = make_tensor(
      exp_data, 2, nvte_dtype_to_nccl(exp_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(exp_shape.data[0]),
      static_cast<unsigned int>(exp_shape.data[1]));

  // Result tensor [T, hidden_dim]
  NVTEShape res_shape = nvte_tensor_shape(result);
  void* res_data = nvte_tensor_data(result);
  NVTEDType res_dtype = nvte_tensor_type(result);
  NVTE_CHECK(res_data != nullptr, "result data must not be null");

  ncclNDTensor_t nccl_result_out = make_tensor(
      res_data, 2, nvte_dtype_to_nccl(res_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(res_shape.data[0]),
      static_cast<unsigned int>(res_shape.data[1]));

  const ncclNDTensor_t inputs[] = {nccl_expert_in};
  const ncclNDTensor_t outputs[] = {nccl_result_out};

  // HT mode: no local tensors for combine.
  // Topk_weights were sent alongside tokens during dispatch; NCCL EP
  // uses the received copy for weighted accumulation.
  NVTE_CHECK_NCCL(ncclEpCombine(
      handle,
      inputs, 1,
      outputs, 1,
      nullptr, 0,        // no local tensors in HT mode
      0,                  // send_only = false
      nullptr,            // config — reserved
      stream));

  // Complete the combine operation.
  NVTE_CHECK_NCCL(ncclEpComplete(handle, nullptr, stream));

  destroy_tensor(nccl_expert_in);
  destroy_tensor(nccl_result_out);
}

void NCCLEPBackend::dispatch_bwd(void* handle_mem,
                                   const NVTETensor grad,
                                   NVTETensor result,
                                   cudaStream_t stream) {
  // Backward of dispatch is a combine operation (reverse direction).
  // We reuse the same handle_mem with a combine call.
  combine(handle_mem, grad, result, stream);
}

void NCCLEPBackend::combine_bwd(void* handle_mem,
                                  const NVTETensor grad,
                                  NVTETensor result,
                                  cudaStream_t stream) {
  // Backward of combine is a dispatch operation (reverse direction).
  // Pass nullptr for topk_weights — only forward dispatch sends weights
  // alongside tokens. dispatch() detects the null and uses 1 input/output
  // instead of 3.
  dispatch(handle_mem, grad, /*topk_weights=*/nullptr, result, stream);
}

}  // namespace ep
}  // namespace transformer_engine
