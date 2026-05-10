/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.cpp
 *  \brief EPBackend implementation.
 *
 *  Wraps ncclEpCreateGroup, ncclEpInitHandle, ncclEpUpdateHandle,
 *  ncclEpDispatch, ncclEpCombine, and their destroy counterparts.
 *  Each per-step op creates ephemeral ncclNDTensor_t handles around
 *  user-provided buffers — no allocations, negligible overhead.
 *
 *  API patterns:
 *  - ncclEpInitHandle: maps routing buffers in handle_mem (no collective)
 *  - ncclEpUpdateHandle: AllGather + metadata preprocessing (collective)
 *  - ncclEpDispatch (forward): 3 inputs (tokens, topk_weights, topk_idx),
 *    2 outputs (recv_tokens, recv_topk_weights), 0 local tensors
 *  - ncclEpDispatch (backward/combine_bwd): 1 input (grad), 1 output (result),
 *    0 local tensors (no topk_weights in backward direction)
 *  - ncclEpCombine: 1 input (expert_out), 1 output (result), 0 local tensors
 *  - All dispatch/combine outputs are 2D tensors
 */

#include "ep_backend.h"

#include "../util/cuda_runtime.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace ep {

// ---------------------------------------------------------------------------
// Singleton + bootstrap
// ---------------------------------------------------------------------------

EPBackend& EPBackend::instance() {
  static EPBackend inst;
  return inst;
}

EPBackend& EPBackend::get() {
  EPBackend& inst = instance();
  NVTE_CHECK(inst.initialized_, "EPBackend not initialized. Call nvte_ep_initialize() first.");
  return inst;
}

void EPBackend::validate_config(const NVTEEpGroupConfig& config) {
  NVTE_CHECK(config.ep_size > 0, "ep_size must be positive, got ", config.ep_size);
  NVTE_CHECK(config.num_experts > 0, "num_experts must be positive, got ", config.num_experts);
  NVTE_CHECK(config.max_tokens_per_rank > 0, "max_tokens_per_rank must be positive, got ",
             config.max_tokens_per_rank);
  NVTE_CHECK(config.hidden_dim > 0, "hidden_dim must be positive, got ", config.hidden_dim);
  NVTE_CHECK(config.num_experts % config.ep_size == 0, "num_experts (", config.num_experts,
             ") must be divisible by ep_size (", config.ep_size, ")");

  int device, major;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(major >= 9,
             "NCCL EP requires SM_90+ (Hopper or later), "
             "but current device has compute capability ",
             major, ".x");

  // NCCL EP requires CUDA multicast (NVLS) support — init hangs forever on
  // hardware where CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED == 0 (e.g. H200 NVL
  // with NVLink Bridge topology). Detect at init and fail fast.
  NVTE_CHECK(cuda::supports_multicast(device),
             "NCCL EP requires CUDA multicast (NVLS) support on device ", device,
             " but CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED reports 0. "
             "This typically means the system has a partial NVLink topology "
             "(e.g. H200 NVL with NVLink Bridge). NCCL EP requires a full "
             "NVLink mesh fabric (e.g. GH200 NVL72 / DGX H200 / DGX H100).");
}

void EPBackend::initialize(const ncclUniqueId& uid, int world_size, int rank,
                           NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_, "EP already initialized. Call initialize only once per process.");

  NVTE_CHECK(world_size > 0, "world_size must be positive, got ", world_size);
  NVTE_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size), got rank=", rank,
             " world_size=", world_size);
  NVTE_CHECK(world_size >= config.ep_size, "world_size (", world_size, ") must be >= ep_size (",
             config.ep_size, ")");
  validate_config(config);

  ncclComm_t world_comm;
  NVTE_CHECK_NCCL(ncclCommInitRank(&world_comm, world_size, uid, rank));

  ncclComm_t ep_comm;
  NVTE_CHECK_NCCL(ncclCommSplit(world_comm, rank / config.ep_size, rank, &ep_comm, nullptr));
  NVTE_CHECK_NCCL(ncclCommDestroy(world_comm));

  inst.init(ep_comm, config, /*owns_comm=*/true);
}

void EPBackend::initialize_with_comm(ncclComm_t ep_comm, NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_, "EP already initialized. Call initialize only once per process.");
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  validate_config(config);

  int comm_size = 0;
  NVTE_CHECK_NCCL(ncclCommCount(ep_comm, &comm_size));
  NVTE_CHECK(comm_size == config.ep_size, "ep_comm size (", comm_size, ") must equal ep_size (",
             config.ep_size, "). Pass the EP sub-communicator, not the world comm.");

  inst.init(ep_comm, config, /*owns_comm=*/false);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ncclDataType_t EPBackend::nvte_dtype_to_nccl(NVTEDType dtype) {
  switch (dtype) {
    case kNVTEFloat32:
      return ncclFloat32;
    case kNVTEFloat16:
      return ncclFloat16;
    case kNVTEBFloat16:
      return ncclBfloat16;
    case kNVTEInt32:
      return ncclInt32;
    case kNVTEInt64:
      return ncclInt64;
    case kNVTEByte:
      return ncclUint8;
    case kNVTEFloat8E4M3:
      return ncclFloat8e4m3;
    case kNVTEFloat8E5M2:
      return ncclFloat8e5m2;
    default:
      NVTE_ERROR("Unsupported NVTEDType for NCCL EP conversion: ", static_cast<int>(dtype));
  }
  return ncclFloat32;  // unreachable
}

ncclNDTensor_t EPBackend::make_tensor(void* data, unsigned int ndim, ncclDataType_t datatype,
                                      ncclEpTensorTag_t tag, unsigned int size0, unsigned int size1,
                                      unsigned int size2, unsigned int size3, unsigned int size4) {
  NVTE_CHECK(ep_group_ != nullptr, "EPBackend not initialized");
  ncclNDTensor_t tensor;
  NVTE_CHECK_NCCL(
      ncclEpTensorCreate(&tensor, ndim, datatype, tag, data, size0, size1, size2, size3, size4));
  return tensor;
}

void EPBackend::destroy_tensor(ncclNDTensor_t tensor) {
  if (tensor != nullptr) {
    NVTE_CHECK_NCCL(ncclEpTensorDestroy(tensor));
  }
}

// Build a transient ncclEpHandle that views the existing handle_mem buffer.
// ncclEpInitHandle is pure host-side pointer arithmetic: it stores
// handle_mem->data as a raw pointer in the host-side struct. The routing-tensor
// wrapper can be freed immediately. Calling Init after a prior UpdateHandle
// preserves the AllGather results already in the device buffer.
ncclEpHandle_t EPBackend::open_handle(void* handle_mem, int num_topk) {
  ncclNDTensor_t routing_tensor = make_tensor(handle_mem, 1, ncclUint8, NCCL_EP_TENSOR_TAG_NONE,
                                              static_cast<unsigned int>(routing_buf_size_));
  ncclEpHandle_t handle;
  NVTE_CHECK_NCCL(ncclEpInitHandle(&handle, ep_group_, nullptr, num_topk,
                                   /*use_fp8=*/false, routing_tensor));
  destroy_tensor(routing_tensor);
  return handle;
}

void EPBackend::close_handle(ncclEpHandle_t handle) {
  if (handle != nullptr) {
    NVTE_CHECK_NCCL(ncclEpHandleDestroy(handle));
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

EPBackend::~EPBackend() {
  // Guard against concurrent dispatch/combine calls during teardown.
  std::lock_guard<std::mutex> lock(mutex_);
  if (cur_handle_ != nullptr) {
    ncclEpHandleDestroy(cur_handle_);
    cur_handle_ = nullptr;
  }
  // Order matters: ncclEpGroupDestroy reads from ep_comm_, so destroy the
  // group first, then the underlying communicator.
  if (ep_group_ != nullptr) {
    ncclEpGroupDestroy(ep_group_);
    ep_group_ = nullptr;
  }
  if (ep_comm_ != nullptr) {
    if (owns_ep_comm_) {
      ncclCommDestroy(ep_comm_);
    }
    ep_comm_ = nullptr;
  }
}

void EPBackend::init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config, bool owns_comm) {
  NVTE_CHECK(!initialized_, "EPBackend already initialized");

  group_config_ = group_config;
  owns_ep_comm_ = owns_comm;

  ncclEpGroupConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.version = 1;
  cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
  cfg.layout = NCCL_EP_LAYOUT_EXPERT_MAJOR;
  cfg.num_experts = static_cast<unsigned int>(group_config.num_experts);
  cfg.max_send_tokens_per_rank = static_cast<unsigned int>(group_config.max_tokens_per_rank);
  cfg.token_size_bytes = static_cast<unsigned int>(group_config.hidden_dim * sizeof(nv_bfloat16));
  cfg.rdma_buffer_size = NCCL_EP_AUTO;
  cfg.num_qp_per_rank = NCCL_EP_AUTO;
  cfg.num_channels = NCCL_EP_AUTO;
  // Must be > 0; NCCL EP errors out on 0.
  cfg.max_recv_token_slots_per_rank =
      static_cast<unsigned int>(group_config.max_recv_tokens_per_rank);

  NVTE_CHECK_NCCL(
      ncclEpCreateGroup(&ep_group_, ep_comm, &cfg, /*alloc=*/nullptr, /*free=*/nullptr));

  // handle_mem size depends on top_k — sized per call in get_handle_mem_size()
  // / open_handle().

  // Keep ep_comm alive for the EP group's lifetime — ncclEpGroupDestroy
  // depends on it at teardown.
  ep_comm_ = ep_comm;

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Handle mem size query
// ---------------------------------------------------------------------------

size_t EPBackend::get_handle_mem_size(NVTEEpLayerConfig layer_config) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(layer_config.top_k > 0, "NVTEEpLayerConfig.top_k must be > 0");
  size_t bytes = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, nullptr, &bytes, layer_config.top_k));
  routing_buf_size_ = bytes;
  return bytes;
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

void EPBackend::prepare(const NVTETensor topk_idx, NVTETensor token_counts, void* handle_mem,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
  void* idx_data = nvte_tensor_data(topk_idx);
  NVTE_CHECK(idx_data != nullptr, "topk_idx data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(idx_shape.data[0]);
  const unsigned int top_k = idx_shape.ndim > 1 ? static_cast<unsigned int>(idx_shape.data[1]) : 1;
  const unsigned int num_local_experts =
      static_cast<unsigned int>(group_config_.num_experts / group_config_.ep_size);

  ncclNDTensor_t nccl_topk_idx =
      make_tensor(idx_data, 2, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, num_tokens, top_k);

  // Optionally surface token_counts as RECV_EXPERT_COUNTER so UpdateHandle
  // populates it during the metadata exchange.
  ncclNDTensor_t handle_local_tensors[1] = {nullptr};
  unsigned int handle_num_local_tensors = 0;
  void* token_counts_data = (token_counts != nullptr) ? nvte_tensor_data(token_counts) : nullptr;
  if (token_counts_data != nullptr) {
    handle_local_tensors[0] =
        make_tensor(token_counts_data, 1, ncclInt32, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                    num_local_experts);
    handle_num_local_tensors = 1;
  }

  // Reuse the persistent handle if it already views this handle_mem buffer;
  // otherwise close any prior handle and open a new one. UpdateHandle then
  // populates num_tokens / routing state on the live handle.
  if (cur_handle_ != nullptr && cur_handle_mem_ != handle_mem) {
    close_handle(cur_handle_);
    cur_handle_ = nullptr;
  }
  if (cur_handle_ == nullptr) {
    cur_handle_ = open_handle(handle_mem, static_cast<int>(top_k));
    cur_handle_mem_ = handle_mem;
  }
  NVTE_CHECK_NCCL(ncclEpUpdateHandle(cur_handle_, nccl_topk_idx, handle_local_tensors,
                                     handle_num_local_tensors, stream));

  destroy_tensor(nccl_topk_idx);
  if (handle_num_local_tensors > 0) destroy_tensor(handle_local_tensors[0]);
}

void EPBackend::dispatch(void* handle_mem, const NVTETensor topk_idx, const NVTETensor tokens,
                         const NVTETensor topk_weights, NVTETensor recv_tokens,
                         NVTETensor recv_topk_weights, cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape tok_shape = nvte_tensor_shape(tokens);
  void* tok_data = nvte_tensor_data(tokens);
  NVTEDType tok_dtype = nvte_tensor_type(tokens);
  NVTE_CHECK(tok_data != nullptr, "tokens data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(tok_shape.data[0]);
  const unsigned int hidden_dim = static_cast<unsigned int>(tok_shape.data[1]);

  ncclNDTensor_t nccl_tokens_in = make_tensor(tok_data, 2, nvte_dtype_to_nccl(tok_dtype),
                                              NCCL_EP_TENSOR_TAG_TOKENS, num_tokens, hidden_dim);

  void* weights_data = (topk_weights != nullptr) ? nvte_tensor_data(topk_weights) : nullptr;
  const bool is_forward = (weights_data != nullptr);

  ncclNDTensor_t nccl_topk_weights_in = nullptr;
  ncclNDTensor_t nccl_topk_idx_in = nullptr;
  unsigned int num_inputs = 1;

  if (is_forward) {
    NVTE_CHECK(topk_idx != nullptr, "topk_idx required in forward dispatch");
    NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
    void* idx_data = nvte_tensor_data(topk_idx);
    const unsigned int top_k =
        idx_shape.ndim > 1 ? static_cast<unsigned int>(idx_shape.data[1]) : 1;
    nccl_topk_weights_in = make_tensor(weights_data, 2, ncclFloat32,
                                       NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, num_tokens, top_k);
    nccl_topk_idx_in =
        make_tensor(idx_data, 2, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, num_tokens, top_k);
    num_inputs = 3;
  }

  const ncclNDTensor_t inputs[] = {nccl_tokens_in, nccl_topk_weights_in, nccl_topk_idx_in};

  NVTEShape recv_shape = nvte_tensor_shape(recv_tokens);
  void* recv_data = nvte_tensor_data(recv_tokens);
  NVTEDType recv_dtype = nvte_tensor_type(recv_tokens);
  NVTE_CHECK(recv_data != nullptr, "recv_tokens data must not be null");
  const unsigned int recv_capacity = static_cast<unsigned int>(recv_shape.data[0]);

  ncclNDTensor_t nccl_tokens_out =
      make_tensor(recv_data, 2, nvte_dtype_to_nccl(recv_dtype), NCCL_EP_TENSOR_TAG_TOKENS,
                  recv_capacity, static_cast<unsigned int>(recv_shape.data[1]));

  // Forward: 2 outputs (tokens, topk_weights). recv_topk_idx not required.
  ncclNDTensor_t nccl_topk_weights_out = nullptr;
  unsigned int num_outputs = 1;

  if (is_forward) {
    NVTE_CHECK(recv_topk_weights != nullptr,
               "recv_topk_weights must not be null in forward dispatch");
    void* recv_w_data = nvte_tensor_data(recv_topk_weights);
    NVTEShape recv_w_shape = nvte_tensor_shape(recv_topk_weights);
    NVTE_CHECK(recv_w_data != nullptr, "recv_topk_weights data must not be null");
    NVTE_CHECK(recv_w_shape.ndim == 1,
               "recv_topk_weights must be 1D [recv_capacity]");
    nccl_topk_weights_out = make_tensor(recv_w_data, 1, ncclFloat32,
                                        NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, recv_capacity);
    num_outputs = 2;
  }

  // outputs[] contains only the non-null tensors (no trailing nullptr slot).
  ncclNDTensor_t outputs_buf[2] = {nccl_tokens_out, nullptr};
  if (is_forward) outputs_buf[1] = nccl_topk_weights_out;
  const ncclNDTensor_t* outputs = outputs_buf;

  ncclEpDispatchConfig_t dispatch_cfg;
  memset(&dispatch_cfg, 0, sizeof(dispatch_cfg));

  // Use the persistent handle populated by prepare(); fall back to opening
  // a fresh one (backward path: dispatch may run without a prior prepare).
  ncclEpHandle_t handle = cur_handle_;
  bool owns_handle = false;
  if (handle == nullptr || cur_handle_mem_ != handle_mem) {
    int handle_num_topk = -1;
    if (is_forward) {
      NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
      handle_num_topk = idx_shape.ndim > 1 ? static_cast<int>(idx_shape.data[1]) : 1;
    }
    handle = open_handle(handle_mem, handle_num_topk);
    owns_handle = true;
  }
  NVTE_CHECK_NCCL(ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs, nullptr, 0, 0,
                                 &dispatch_cfg, stream));
  if (owns_handle) close_handle(handle);

  destroy_tensor(nccl_tokens_in);
  if (nccl_topk_weights_in) destroy_tensor(nccl_topk_weights_in);
  if (nccl_topk_idx_in) destroy_tensor(nccl_topk_idx_in);
  destroy_tensor(nccl_tokens_out);
  if (nccl_topk_weights_out) destroy_tensor(nccl_topk_weights_out);
}

void EPBackend::combine(void* handle_mem, const NVTETensor expert_out, NVTETensor result,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape exp_shape = nvte_tensor_shape(expert_out);
  void* exp_data = nvte_tensor_data(expert_out);
  NVTEDType exp_dtype = nvte_tensor_type(expert_out);
  NVTE_CHECK(exp_data != nullptr, "expert_out data must not be null");

  ncclNDTensor_t nccl_expert_in = make_tensor(
      exp_data, 2, nvte_dtype_to_nccl(exp_dtype), NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(exp_shape.data[0]), static_cast<unsigned int>(exp_shape.data[1]));

  NVTEShape res_shape = nvte_tensor_shape(result);
  void* res_data = nvte_tensor_data(result);
  NVTEDType res_dtype = nvte_tensor_type(result);
  NVTE_CHECK(res_data != nullptr, "result data must not be null");

  ncclNDTensor_t nccl_result_out = make_tensor(
      res_data, 2, nvte_dtype_to_nccl(res_dtype), NCCL_EP_TENSOR_TAG_TOKENS,
      static_cast<unsigned int>(res_shape.data[0]), static_cast<unsigned int>(res_shape.data[1]));

  const ncclNDTensor_t inputs[] = {nccl_expert_in};
  const ncclNDTensor_t outputs[] = {nccl_result_out};

  // Combine requires the persistent handle from prepare() — combine reads
  // handle->num_tokens which is only populated by ncclEpUpdateHandle.
  NVTE_CHECK(cur_handle_ != nullptr && cur_handle_mem_ == handle_mem,
             "ep_combine requires a prior ep_prepare on the same handle_mem");
  NVTE_CHECK_NCCL(
      ncclEpCombine(cur_handle_, inputs, 1, outputs, 1, nullptr, 0, 0, nullptr, stream));

  destroy_tensor(nccl_expert_in);
  destroy_tensor(nccl_result_out);
}

void EPBackend::dispatch_bwd(void* handle_mem, const NVTETensor grad, NVTETensor grad_tokens,
                             cudaStream_t stream) {
  // Backward of dispatch is a combine in disguise.
  combine(handle_mem, grad, grad_tokens, stream);
}

void EPBackend::combine_bwd(void* handle_mem, const NVTETensor grad, NVTETensor grad_expert_out,
                            cudaStream_t stream) {
  // Backward of combine is a dispatch in the backward direction (no weights, 1 input).
  dispatch(handle_mem, /*topk_idx=*/nullptr, grad, /*topk_weights=*/nullptr, grad_expert_out,
           /*recv_topk_weights=*/nullptr, stream);
}

}  // namespace ep
}  // namespace transformer_engine
