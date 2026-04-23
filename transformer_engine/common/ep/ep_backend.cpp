/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.cpp
 *  \brief EPBackend implementation (HT mode only).
 *
 *  Wraps ncclEpCreateGroup, ncclEpInitHandle, ncclEpUpdateHandle,
 *  ncclEpDispatch, ncclEpCombine, and their destroy counterparts.
 *  Each per-step op creates ephemeral ncclNDTensor_t handles around
 *  user-provided buffers — no allocations, negligible overhead.
 *
 *  HT mode API patterns:
 *  - ncclEpInitHandle: maps routing buffers in handle_mem (no collective)
 *  - ncclEpUpdateHandle: AllGather + metadata preprocessing (collective)
 *  - ncclEpDispatch (forward): 3 inputs (tokens, topk_weights, topk_idx),
 *    3 outputs (recv_tokens, recv_topk_weights, recv_topk_idx), 0 local tensors
 *  - ncclEpDispatch (backward/combine_bwd): 1 input (grad), 1 output (result),
 *    0 local tensors (no topk_weights in backward direction)
 *  - ncclEpCombine: 1 input (expert_out), 1 output (result), 0 local tensors
 *  - All dispatch/combine outputs are 2D tensors
 */

#include "ep_backend.h"
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
 *
 *  handle_mem layout:
 *    [0 .. kHeaderAlignedSize-1]  : HandleMemHeader
 *    [kHeaderAlignedSize ..]      : NCCL EP routing buffers (size = routing_buf_size_)
 */
struct HandleMemHeader {
  ncclEpHandle_t handle;              ///< NCCL EP handle; set by reinit_handle, destroyed in destroy_handle
  ncclNDTensor_t routing_tensor;      ///< Tensor wrapping routing buffer region; destroyed with handle
  void* topk_idx_data;                ///< Device pointer to topk_idx [T, top_k] int64
  void* token_counts_data;            ///< Device pointer to token_counts [num_local_experts] int32
  unsigned int num_tokens;            ///< Number of tokens (T)
  unsigned int top_k;                 ///< Top-k value
  unsigned int num_local_experts;     ///< Number of local experts
  unsigned int num_recv_tokens;       ///< Number of tokens received after dispatch
};

// Routing buffers are placed after the header, aligned to 128 bytes.
static constexpr size_t kHeaderAlignedSize = ((sizeof(HandleMemHeader) + 127) / 128) * 128;

// ---------------------------------------------------------------------------
// Singleton + bootstrap
// ---------------------------------------------------------------------------

EPBackend& EPBackend::instance() {
  static EPBackend inst;
  return inst;
}

EPBackend& EPBackend::get() {
  EPBackend& inst = instance();
  NVTE_CHECK(inst.initialized_,
             "EPBackend not initialized. Call nvte_ep_initialize() first.");
  return inst;
}

void EPBackend::validate_config(const NVTEEpGroupConfig& config) {
  NVTE_CHECK(config.ep_size > 0,
             "ep_size must be positive, got ", config.ep_size);
  NVTE_CHECK(config.num_experts > 0,
             "num_experts must be positive, got ", config.num_experts);
  NVTE_CHECK(config.max_tokens_per_rank > 0,
             "max_tokens_per_rank must be positive, got ", config.max_tokens_per_rank);
  NVTE_CHECK(config.hidden_dim > 0,
             "hidden_dim must be positive, got ", config.hidden_dim);
  NVTE_CHECK(config.num_experts % config.ep_size == 0,
             "num_experts (", config.num_experts,
             ") must be divisible by ep_size (", config.ep_size, ")");

  int device, major;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(major >= 9, "NCCL EP requires SM_90+ (Hopper or later), "
             "but current device has compute capability ", major, ".x");
}

void EPBackend::initialize(const ncclUniqueId& uid, int world_size, int rank,
                            NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_,
             "EP already initialized. Call initialize only once per process.");

  NVTE_CHECK(world_size > 0, "world_size must be positive, got ", world_size);
  NVTE_CHECK(rank >= 0 && rank < world_size,
             "rank must be in [0, world_size), got rank=", rank,
             " world_size=", world_size);
  NVTE_CHECK(world_size >= config.ep_size,
             "world_size (", world_size, ") must be >= ep_size (", config.ep_size, ")");
  validate_config(config);

  ncclComm_t world_comm;
  NVTE_CHECK_NCCL(ncclCommInitRank(&world_comm, world_size, uid, rank));

  ncclComm_t ep_comm;
  NVTE_CHECK_NCCL(ncclCommSplit(world_comm, rank / config.ep_size, rank, &ep_comm, nullptr));
  NVTE_CHECK_NCCL(ncclCommDestroy(world_comm));

  inst.init(ep_comm, config);
}

void EPBackend::initialize_with_comm(ncclComm_t ep_comm, NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_,
             "EP already initialized. Call initialize only once per process.");
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  validate_config(config);

  int comm_size = 0;
  NVTE_CHECK_NCCL(ncclCommCount(ep_comm, &comm_size));
  NVTE_CHECK(comm_size == config.ep_size,
             "ep_comm size (", comm_size, ") must equal ep_size (", config.ep_size,
             "). Pass the EP sub-communicator, not the world comm.");

  inst.init(ep_comm, config);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ncclDataType_t EPBackend::nvte_dtype_to_nccl(NVTEDType dtype) {
  switch (dtype) {
    case kNVTEFloat32:    return ncclFloat32;
    case kNVTEFloat16:    return ncclFloat16;
    case kNVTEBFloat16:   return ncclBfloat16;
    case kNVTEInt32:      return ncclInt32;
    case kNVTEInt64:      return ncclInt64;
    case kNVTEByte:       return ncclUint8;
    case kNVTEFloat8E4M3: return ncclFloat8e4m3;
    case kNVTEFloat8E5M2: return ncclFloat8e5m2;
    default:
      NVTE_ERROR("Unsupported NVTEDType for NCCL EP conversion: ",
                 static_cast<int>(dtype));
  }
  return ncclFloat32;  // unreachable
}

ncclNDTensor_t EPBackend::make_tensor(void* data, unsigned int ndim,
                                      ncclDataType_t datatype,
                                      ncclEpTensorTag_t tag,
                                      unsigned int size0,
                                      unsigned int size1,
                                      unsigned int size2,
                                      unsigned int size3,
                                      unsigned int size4) {
  NVTE_CHECK(ep_group_ != nullptr, "EPBackend not initialized");
  ncclNDTensor_t tensor;
  NVTE_CHECK_NCCL(ncclEpTensorCreate(
      ep_group_, &tensor, ndim, datatype, tag,
      data, size0, size1, size2, size3, size4));
  return tensor;
}

void EPBackend::destroy_tensor(ncclNDTensor_t tensor) {
  if (tensor != nullptr && ep_group_ != nullptr) {
    NVTE_CHECK_NCCL(ncclEpTensorDestroy(ep_group_, tensor));
  }
}

void EPBackend::reinit_handle(HandleMemHeader* hdr, void* handle_mem) {
  auto* routing_buf = static_cast<uint8_t*>(handle_mem) + kHeaderAlignedSize;
  const bool use_fp8 = false;
  hdr->routing_tensor = make_tensor(routing_buf, 1, ncclUint8,
                                    NCCL_EP_TENSOR_TAG_NONE,
                                    static_cast<unsigned int>(routing_buf_size_));
  NVTE_CHECK_NCCL(ncclEpInitHandle(&hdr->handle, ep_group_, nullptr,
                                    hdr->routing_tensor, use_fp8));
}

void EPBackend::destroy_handle(HandleMemHeader* hdr) {
  NVTE_CHECK_NCCL(ncclEpHandleDestroy(hdr->handle));
  hdr->handle = nullptr;
  destroy_tensor(hdr->routing_tensor);
  hdr->routing_tensor = nullptr;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

EPBackend::~EPBackend() {
  if (ep_group_ != nullptr) {
    ncclEpGroupDestroy(ep_group_, 0);
    ep_group_ = nullptr;
  }
}

void EPBackend::init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(!initialized_, "EPBackend already initialized");

  group_config_ = group_config;

  ncclEpGroupConfig_t cfg;
  cfg.version           = 1;
  cfg.algorithm         = NCCL_EP_ALGO_HIGH_THROUGHPUT;
  cfg.num_experts       = static_cast<unsigned int>(group_config.num_experts);
  cfg.max_tokens_per_rank = static_cast<unsigned int>(group_config.max_tokens_per_rank);
  cfg.token_size_bytes  = static_cast<unsigned int>(group_config.hidden_dim * sizeof(nv_bfloat16));
  cfg.rdma_buffer_size  = NCCL_EP_AUTO;
  cfg.num_qp_per_rank   = NCCL_EP_AUTO;
  cfg.num_channels      = NCCL_EP_AUTO;

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));
  NVTE_CHECK_NCCL(ncclEpCreateGroup(&ep_group_, ep_comm, &cfg, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));

  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, nullptr, &routing_buf_size_));

  // ep_comm is no longer needed — the EP group has its own copy
  NVTE_CHECK_NCCL(ncclCommDestroy(ep_comm));

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Handle mem size query
// ---------------------------------------------------------------------------

size_t EPBackend::get_handle_mem_size(NVTEEpLayerConfig /*layer_config*/) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  return kHeaderAlignedSize + routing_buf_size_;
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

void EPBackend::prepare(void* handle_mem,
                        const NVTETensor topk_idx,
                        NVTETensor token_counts,
                        NVTEEpLayerConfig layer_config,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);

  NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
  void* idx_data = nvte_tensor_data(topk_idx);
  NVTE_CHECK(idx_data != nullptr, "topk_idx data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(idx_shape.data[0]);
  unsigned int top_k = 1;
  if (idx_shape.ndim > 1) {
    top_k = static_cast<unsigned int>(idx_shape.data[1]);
  }

  hdr->num_tokens        = num_tokens;
  hdr->top_k             = top_k;
  hdr->num_local_experts = static_cast<unsigned int>(layer_config.num_local_experts);
  hdr->topk_idx_data     = idx_data;
  hdr->token_counts_data = (token_counts != nullptr) ? nvte_tensor_data(token_counts) : nullptr;

  ncclNDTensor_t nccl_topk_idx = make_tensor(idx_data, 2, ncclInt64,
                                              NCCL_EP_TENSOR_TAG_TOPK_IDX, num_tokens, top_k);

  // Optionally pass token_counts as RECV_EXPERT_COUNTER so ncclEpUpdateHandle
  // populates it during the metadata exchange.
  ncclNDTensor_t handle_local_tensors[1] = {nullptr};
  unsigned int handle_num_local_tensors = 0;
  if (hdr->token_counts_data != nullptr) {
    handle_local_tensors[0] = make_tensor(
        hdr->token_counts_data, 1, ncclInt32,
        NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE, hdr->num_local_experts);
    handle_num_local_tensors = 1;
  }

  // Init handle: map routing buffers in handle_mem (no collective, no kernels)
  reinit_handle(hdr, handle_mem);

  // Update handle: AllGather + metadata preprocessing (collective)
  NVTE_CHECK_NCCL(ncclEpUpdateHandle(hdr->handle, nccl_topk_idx,
                                      handle_local_tensors, handle_num_local_tensors, stream));

  unsigned int num_recv_tokens = 0;
  NVTE_CHECK_NCCL(ncclEpHandleGetNumRecvTokens(hdr->handle, &num_recv_tokens));
  hdr->num_recv_tokens = num_recv_tokens;

  destroy_tensor(nccl_topk_idx);
  if (handle_num_local_tensors > 0) {
    destroy_tensor(handle_local_tensors[0]);
  }
}

void EPBackend::dispatch(void* handle_mem,
                         const NVTETensor tokens,
                         const NVTETensor topk_weights,
                         NVTETensor recv_tokens,
                         cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  ncclEpHandle_t handle = hdr->handle;
  NVTE_CHECK(handle != nullptr, "EP handle is null — was prepare() called?");

  NVTEShape tok_shape = nvte_tensor_shape(tokens);
  void* tok_data      = nvte_tensor_data(tokens);
  NVTEDType tok_dtype = nvte_tensor_type(tokens);
  NVTE_CHECK(tok_data != nullptr, "tokens data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(tok_shape.data[0]);
  const unsigned int hidden_dim = static_cast<unsigned int>(tok_shape.data[1]);

  ncclNDTensor_t nccl_tokens_in = make_tensor(
      tok_data, 2, nvte_dtype_to_nccl(tok_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS, num_tokens, hidden_dim);

  void* weights_data    = (topk_weights != nullptr) ? nvte_tensor_data(topk_weights) : nullptr;
  const bool is_forward = (weights_data != nullptr);

  ncclNDTensor_t nccl_topk_weights_in = nullptr;
  ncclNDTensor_t nccl_topk_idx_in     = nullptr;
  unsigned int num_inputs = 1;

  if (is_forward) {
    nccl_topk_weights_in = make_tensor(
        weights_data, 2, ncclFloat32,
        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, hdr->num_tokens, hdr->top_k);
    nccl_topk_idx_in = make_tensor(
        hdr->topk_idx_data, 2, ncclInt64,
        NCCL_EP_TENSOR_TAG_TOPK_IDX, hdr->num_tokens, hdr->top_k);
    num_inputs = 3;
  }

  const ncclNDTensor_t inputs[] = {nccl_tokens_in, nccl_topk_weights_in, nccl_topk_idx_in};

  NVTEShape recv_shape  = nvte_tensor_shape(recv_tokens);
  void* recv_data       = nvte_tensor_data(recv_tokens);
  NVTEDType recv_dtype  = nvte_tensor_type(recv_tokens);
  NVTE_CHECK(recv_data != nullptr, "recv_tokens data must not be null");

  const unsigned int num_recv_tokens = hdr->num_recv_tokens;

  ncclNDTensor_t nccl_tokens_out = make_tensor(
      recv_data, 2, nvte_dtype_to_nccl(recv_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(recv_shape.data[0]),
      static_cast<unsigned int>(recv_shape.data[1]));

  ncclNDTensor_t nccl_recv_topk_weights = nullptr;
  ncclNDTensor_t nccl_recv_topk_idx     = nullptr;
  unsigned int num_outputs = 1;

  if (is_forward) {
    nccl_recv_topk_weights = make_tensor(
        nullptr, 2, ncclFloat32,
        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, num_recv_tokens, hdr->top_k);
    nccl_recv_topk_idx = make_tensor(
        nullptr, 2, ncclInt64,
        NCCL_EP_TENSOR_TAG_TOPK_IDX, num_recv_tokens, hdr->top_k);
    num_outputs = 3;
  }

  const ncclNDTensor_t outputs[] = {nccl_tokens_out, nccl_recv_topk_weights, nccl_recv_topk_idx};

  ncclEpDispatchConfig_t dispatch_cfg;
  dispatch_cfg.round_scales = 0;

  NVTE_CHECK_NCCL(ncclEpDispatch(
      handle,
      inputs, num_inputs,
      outputs, num_outputs,
      nullptr, 0,
      0,
      &dispatch_cfg,
      stream));

  destroy_tensor(nccl_tokens_in);
  if (nccl_topk_weights_in) destroy_tensor(nccl_topk_weights_in);
  if (nccl_topk_idx_in)     destroy_tensor(nccl_topk_idx_in);
  destroy_tensor(nccl_tokens_out);
  if (nccl_recv_topk_weights) destroy_tensor(nccl_recv_topk_weights);
  if (nccl_recv_topk_idx)     destroy_tensor(nccl_recv_topk_idx);
}

void EPBackend::combine(void* handle_mem,
                        const NVTETensor expert_out,
                        NVTETensor result,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  ncclEpHandle_t handle = hdr->handle;
  NVTE_CHECK(handle != nullptr, "EP handle is null — was prepare() called?");

  NVTEShape exp_shape  = nvte_tensor_shape(expert_out);
  void* exp_data       = nvte_tensor_data(expert_out);
  NVTEDType exp_dtype  = nvte_tensor_type(expert_out);
  NVTE_CHECK(exp_data != nullptr, "expert_out data must not be null");

  ncclNDTensor_t nccl_expert_in = make_tensor(
      exp_data, 2, nvte_dtype_to_nccl(exp_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(exp_shape.data[0]),
      static_cast<unsigned int>(exp_shape.data[1]));

  NVTEShape res_shape  = nvte_tensor_shape(result);
  void* res_data       = nvte_tensor_data(result);
  NVTEDType res_dtype  = nvte_tensor_type(result);
  NVTE_CHECK(res_data != nullptr, "result data must not be null");

  ncclNDTensor_t nccl_result_out = make_tensor(
      res_data, 2, nvte_dtype_to_nccl(res_dtype),
      NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(res_shape.data[0]),
      static_cast<unsigned int>(res_shape.data[1]));

  const ncclNDTensor_t inputs[]  = {nccl_expert_in};
  const ncclNDTensor_t outputs[] = {nccl_result_out};

  NVTE_CHECK_NCCL(ncclEpCombine(
      handle,
      inputs, 1,
      outputs, 1,
      nullptr, 0,
      0,
      nullptr,
      stream));

  // Forward pass done — destroy handle (routing buffers remain in handle_mem for backward)
  destroy_handle(hdr);

  destroy_tensor(nccl_expert_in);
  destroy_tensor(nccl_result_out);
}

void EPBackend::dispatch_bwd(void* handle_mem,
                              const NVTETensor grad,
                              NVTETensor result,
                              cudaStream_t stream) {
  combine(handle_mem, grad, result, stream);
}

void EPBackend::combine_bwd(void* handle_mem,
                             const NVTETensor grad,
                             NVTETensor result,
                             cudaStream_t stream) {
  // Re-map routing buffers already in handle_mem (no collective, no alloc)
  auto* hdr = reinterpret_cast<HandleMemHeader*>(handle_mem);
  reinit_handle(hdr, handle_mem);
  dispatch(handle_mem, grad, /*topk_weights=*/nullptr, result, stream);
}

}  // namespace ep
}  // namespace transformer_engine
