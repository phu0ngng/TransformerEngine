/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.cpp
 *  \brief EPBackend implementation. See ep_backend.h for the op flow.
 */

#include "ep_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace ep {

namespace {

// RAII guard for ncclNDTensor_t — destroys on scope exit, leak-free on throw.
class ScopedNdTensor {
 public:
  ScopedNdTensor() = default;
  ScopedNdTensor(ncclNDTensor_t t) : t_(t) {}
  ~ScopedNdTensor() {
    if (t_ != nullptr) ncclEpTensorDestroy(t_);
  }
  ScopedNdTensor(const ScopedNdTensor&) = delete;
  ScopedNdTensor& operator=(const ScopedNdTensor&) = delete;
  ScopedNdTensor(ScopedNdTensor&& other) noexcept : t_(other.t_) { other.t_ = nullptr; }
  ScopedNdTensor& operator=(ScopedNdTensor&& other) noexcept {
    if (this != &other) {
      if (t_ != nullptr) ncclEpTensorDestroy(t_);
      t_ = other.t_;
      other.t_ = nullptr;
    }
    return *this;
  }
  operator ncclNDTensor_t() const { return t_; }
  ncclNDTensor_t get() const { return t_; }

 private:
  ncclNDTensor_t t_ = nullptr;
};

// RAII guard for ncclEpHandle_t — destroys on scope exit, leak-free on throw.
class ScopedEpHandle {
 public:
  ScopedEpHandle() = default;
  ScopedEpHandle(ncclEpHandle_t h) : h_(h) {}
  ~ScopedEpHandle() {
    if (h_ != nullptr) ncclEpHandleDestroy(h_);
  }
  ScopedEpHandle(const ScopedEpHandle&) = delete;
  ScopedEpHandle& operator=(const ScopedEpHandle&) = delete;
  ScopedEpHandle(ScopedEpHandle&& other) noexcept : h_(other.h_) { other.h_ = nullptr; }
  ScopedEpHandle& operator=(ScopedEpHandle&& other) noexcept {
    if (this != &other) {
      if (h_ != nullptr) ncclEpHandleDestroy(h_);
      h_ = other.h_;
      other.h_ = nullptr;
    }
    return *this;
  }
  operator ncclEpHandle_t() const { return h_; }
  ncclEpHandle_t get() const { return h_; }

 private:
  ncclEpHandle_t h_ = nullptr;
};

}  // namespace

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
  NVTE_CHECK(config.hidden_dim * sizeof(nv_bfloat16) >= 16,
             "hidden_dim * 2 must be >= 16 (NCCL EP 16B row alignment); got hidden_dim=",
             config.hidden_dim);
  NVTE_CHECK(config.num_experts % config.ep_size == 0, "num_experts (", config.num_experts,
             ") must be divisible by ep_size (", config.ep_size, ")");
  NVTE_CHECK(config.max_num_sms >= 0, "max_num_sms must be >= 0 (0 = auto), got ",
             config.max_num_sms);

  int device, major;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(major >= 9,
             "NCCL EP requires SM_90+ (Hopper or later), "
             "but current device has compute capability ",
             major, ".x");

  // NCCL EP needs CUDA multicast (NVLS); init hangs without it.
  NVTE_CHECK(cuda::supports_multicast(device),
             "NCCL EP requires CUDA multicast (NVLS) support on device ", device,
             " but CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED reports 0.");
}

void EPBackend::initialize(ncclComm_t ep_comm, NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_, "EP already initialized. Call initialize only once per process.");
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  validate_config(config);

  int comm_size = 0;
  NVTE_CHECK_NCCL(ncclCommCount(ep_comm, &comm_size));
  NVTE_CHECK(comm_size == config.ep_size, "ep_comm size (", comm_size, ") must equal ep_size (",
             config.ep_size, "). Pass the EP sub-communicator, not the world comm.");

  inst.init(ep_comm, config);
}

void EPBackend::shutdown() {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  if (!inst.initialized_) return;
  inst.handles_.clear();
  // ncclEpGroupDestroy reads from ep_comm_; destroy group while comm is still alive.
  if (inst.ep_group_ != nullptr) {
    ncclEpGroupDestroy(inst.ep_group_);
    inst.ep_group_ = nullptr;
  }
  inst.ep_comm_ = nullptr;  // borrowed — caller destroys
  inst.initialized_ = false;
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
                                      unsigned int size0, unsigned int size1, unsigned int size2,
                                      unsigned int size3, unsigned int size4) {
  NVTE_CHECK(ep_group_ != nullptr, "EPBackend not initialized");
  NVTE_CHECK(ndim >= 1 && ndim <= 5, "make_tensor: ndim must be in [1, 5], got ", ndim);
  const size_t sizes[5] = {size0, size1, size2, size3, size4};
  ncclNDTensor_t tensor;
  NVTE_CHECK_NCCL(ncclEpTensorCreate(&tensor, ndim, datatype, data, sizes));
  return tensor;
}

// NCCL-window peer handle accessor; returns nullptr/0 if not attached.
static inline void* tensor_nccl_window(const NVTETensor t) {
  const auto* tensor = convertNVTETensor(t);
  return (tensor != nullptr && tensor->peer_handle_kind == NVTE_PEER_HANDLE_NCCL_WINDOW)
             ? tensor->peer_handle_data
             : nullptr;
}
static inline uint64_t tensor_nccl_window_offset(const NVTETensor t) {
  const auto* tensor = convertNVTETensor(t);
  return (tensor != nullptr && tensor->peer_handle_kind == NVTE_PEER_HANDLE_NCCL_WINDOW)
             ? tensor->peer_handle_offset
             : 0;
}

ncclNDTensor_t EPBackend::make_payload_tensor(const NVTETensor t, unsigned int ndim,
                                              ncclDataType_t datatype, unsigned int size0,
                                              unsigned int size1, unsigned int size2,
                                              unsigned int size3, unsigned int size4) {
  NVTE_CHECK(ep_group_ != nullptr, "EPBackend not initialized");
  NVTE_CHECK(ndim >= 1 && ndim <= 5, "make_payload_tensor: ndim must be in [1, 5], got ", ndim);
  void* win = tensor_nccl_window(t);
  const size_t sizes[5] = {size0, size1, size2, size3, size4};
  ncclNDTensor_t tensor;
  if (win != nullptr) {
    const uint64_t off = tensor_nccl_window_offset(t);
    NVTE_CHECK_NCCL(ncclEpTensorCreateFromWindow(&tensor, ndim, datatype,
                                                 static_cast<ncclWindow_t>(win), off, sizes));
  } else {
    void* data = nvte_tensor_data(t);
    NVTE_CHECK(data != nullptr, "make_payload_tensor: tensor data must not be null");
    NVTE_CHECK_NCCL(ncclEpTensorCreate(&tensor, ndim, datatype, data, sizes));
  }
  return tensor;
}

// Open a transient ncclEpHandle over handle_mem. Caller owns the result;
// wrap in ScopedEpHandle for leak-safe teardown.
ncclEpHandle_t EPBackend::open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                                      size_t dispatch_output_per_expert_alignment) {
  ScopedNdTensor routing_tensor(
      make_tensor(handle_mem, 1, ncclUint8, static_cast<unsigned int>(handle_mem_size)));
  ncclEpHandleConfig_t hcfg{};
  hcfg.size = static_cast<unsigned int>(sizeof(hcfg));
  hcfg.dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment;
  ncclEpHandle_t handle;
  NVTE_CHECK_NCCL(ncclEpInitHandle(&handle, ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, num_topk,
                                   routing_tensor));
  return handle;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Last-ditch teardown at static-dtor time: skip NCCL calls (CUDA context /
// borrowed ep_comm_ may already be gone) and just release in-memory state.
EPBackend::~EPBackend() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) return;
  handles_.clear();
  ep_group_ = nullptr;
  ep_comm_ = nullptr;
  initialized_ = false;
}

void EPBackend::init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(!initialized_, "EPBackend already initialized");

  group_config_ = group_config;

  ncclEpGroupConfig_t cfg{};
  cfg.size = static_cast<unsigned int>(sizeof(cfg));
  cfg.version = NCCL_EP_API_VERSION;
  cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
  cfg.num_experts = static_cast<unsigned int>(group_config.num_experts);
  cfg.max_dispatch_tokens_per_rank = static_cast<unsigned int>(group_config.max_tokens_per_rank);
  cfg.max_token_bytes = static_cast<unsigned int>(group_config.hidden_dim * sizeof(nv_bfloat16));
  cfg.rdma_buffer_size = NCCL_EP_AUTO;
  cfg.num_qp_per_rank = NCCL_EP_AUTO;
  cfg.num_channels = NCCL_EP_AUTO;
  cfg.max_num_sms = group_config.max_num_sms > 0
                        ? static_cast<unsigned int>(group_config.max_num_sms)
                        : NCCL_EP_AUTO;
  // Must be > 0; NCCL EP errors out on 0.
  cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(group_config.max_recv_tokens_per_rank);

  NVTE_CHECK_NCCL(ncclEpCreateGroup(&ep_group_, ep_comm, &cfg));

  ep_comm_ = ep_comm;

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Handle mem size query
// ---------------------------------------------------------------------------

size_t EPBackend::get_handle_mem_size(NVTEEpLayerConfig layer_config) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(layer_config.top_k > 0, "NVTEEpLayerConfig.top_k must be > 0");
  ncclEpHandleConfig_t hcfg{};
  hcfg.size = static_cast<unsigned int>(sizeof(hcfg));
  hcfg.dispatch_output_per_expert_alignment = layer_config.dispatch_output_per_expert_alignment;
  size_t bytes = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, &bytes,
                                      layer_config.top_k));
  return bytes;
}

// ---------------------------------------------------------------------------
// Per-handle_id config cache
// ---------------------------------------------------------------------------

uint64_t EPBackend::insert_new_entry(size_t handle_mem_size, int top_k, size_t alignment) {
  if (handle_cache_cap_ == 0) {
    // Default 8192 (~1 MB) covers PP-MoE worst case.
    const char* cap_env = std::getenv("NVTE_EP_HANDLE_CACHE_SIZE");
    handle_cache_cap_ = (cap_env != nullptr) ? std::max<size_t>(1, std::atoi(cap_env)) : 8192;
  }
  NVTE_CHECK(handles_.size() < handle_cache_cap_, "EP handle cache full (", handle_cache_cap_,
             " entries). Raise the cap via NVTE_EP_HANDLE_CACHE_SIZE if your"
             " pipeline-parallel layer/micro-batch count legitimately exceeds it.");
  uint64_t id = next_handle_id_.fetch_add(1, std::memory_order_relaxed);
  handles_.emplace(id, HandleEntry{handle_mem_size, alignment, top_k});
  return id;
}

EPBackend::HandleEntry& EPBackend::lookup_config(uint64_t handle_id) {
  auto it = handles_.find(handle_id);
  NVTE_CHECK(it != handles_.end(), "ep op on handle_id=", handle_id,
             " with no cached config — call ep_prepare first.");
  return it->second;
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

uint64_t EPBackend::allocate_handle_id(NVTEEpLayerConfig layer_config) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(layer_config.top_k > 0, "NVTEEpLayerConfig.top_k must be > 0");
  ncclEpHandleConfig_t hcfg{};
  hcfg.size = static_cast<unsigned int>(sizeof(hcfg));
  hcfg.dispatch_output_per_expert_alignment = layer_config.dispatch_output_per_expert_alignment;
  size_t hm_size = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, &hm_size,
                                      layer_config.top_k));
  std::lock_guard<std::mutex> lock(mutex_);
  return insert_new_entry(hm_size, layer_config.top_k,
                          layer_config.dispatch_output_per_expert_alignment);
}

void EPBackend::prepare(uint64_t handle_id, const NVTETensor topk_idx, NVTETensor token_counts,
                        void* handle_mem, size_t dispatch_output_per_expert_alignment,
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

  ScopedNdTensor nccl_topk_idx(make_tensor(idx_data, 2, ncclInt64, num_tokens, top_k));

  // ncclEpUpdateHandle writes per-expert counts via expert_counters.
  ScopedNdTensor token_counts_tensor;
  void* token_counts_data = (token_counts != nullptr) ? nvte_tensor_data(token_counts) : nullptr;
  if (token_counts_data != nullptr) {
    token_counts_tensor =
        ScopedNdTensor(make_tensor(token_counts_data, 1, ncclInt32, num_local_experts));
  }
  ncclEpLayoutInfo_t layout_info{};
  layout_info.size = static_cast<unsigned int>(sizeof(layout_info));
  layout_info.expert_counters = token_counts_tensor;

  ScopedEpHandle transient;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    HandleEntry& cfg = lookup_config(handle_id);
    NVTE_CHECK(cfg.alignment == dispatch_output_per_expert_alignment,
               "ep_prepare: alignment mismatch for handle_id=", handle_id,
               " (cached=", cfg.alignment, ", got=", dispatch_output_per_expert_alignment, ")");
    transient =
        ScopedEpHandle(open_handle(handle_mem, cfg.handle_mem_size, cfg.top_k, cfg.alignment));
  }
  NVTE_CHECK_NCCL(ncclEpUpdateHandle(transient, nccl_topk_idx, &layout_info, stream));
}

void EPBackend::dispatch(uint64_t handle_id, void* handle_mem, const NVTETensor topk_idx,
                         const NVTETensor tokens, const NVTETensor topk_weights,
                         NVTETensor recv_tokens, NVTETensor recv_topk_weights,
                         cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape tok_shape = nvte_tensor_shape(tokens);
  void* tok_data = nvte_tensor_data(tokens);
  NVTEDType tok_dtype = nvte_tensor_type(tokens);
  NVTE_CHECK(tok_data != nullptr, "tokens data must not be null");

  const unsigned int num_tokens = static_cast<unsigned int>(tok_shape.data[0]);
  const unsigned int hidden_dim = static_cast<unsigned int>(tok_shape.data[1]);

  ScopedNdTensor nccl_tokens_in(
      make_payload_tensor(tokens, 2, nvte_dtype_to_nccl(tok_dtype), num_tokens, hidden_dim));

  void* weights_data = (topk_weights != nullptr) ? nvte_tensor_data(topk_weights) : nullptr;
  const bool is_forward = (weights_data != nullptr);

  // Routing is already cached in handle_mem by ep_prepare; dispatch only
  // needs topk_weights to reconstruct the sparse-to-dense prob map.
  ScopedNdTensor nccl_topk_weights_in;
  if (is_forward) {
    NVTE_CHECK(topk_idx != nullptr, "topk_idx required in forward dispatch");
    NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
    const unsigned int top_k =
        idx_shape.ndim > 1 ? static_cast<unsigned int>(idx_shape.data[1]) : 1;
    nccl_topk_weights_in =
        ScopedNdTensor(make_tensor(weights_data, 2, ncclFloat32, num_tokens, top_k));
  }

  NVTEShape recv_shape = nvte_tensor_shape(recv_tokens);
  void* recv_data = nvte_tensor_data(recv_tokens);
  NVTEDType recv_dtype = nvte_tensor_type(recv_tokens);
  NVTE_CHECK(recv_data != nullptr, "recv_tokens data must not be null");
  const unsigned int recv_capacity = static_cast<unsigned int>(recv_shape.data[0]);

  ScopedNdTensor nccl_tokens_out(
      make_payload_tensor(recv_tokens, 2, nvte_dtype_to_nccl(recv_dtype), recv_capacity,
                          static_cast<unsigned int>(recv_shape.data[1])));

  ScopedNdTensor nccl_topk_weights_out;
  if (is_forward) {
    NVTE_CHECK(recv_topk_weights != nullptr,
               "recv_topk_weights must not be null in forward dispatch");
    void* recv_w_data = nvte_tensor_data(recv_topk_weights);
    NVTEShape recv_w_shape = nvte_tensor_shape(recv_topk_weights);
    NVTE_CHECK(recv_w_data != nullptr, "recv_topk_weights data must not be null");
    NVTE_CHECK(recv_w_shape.ndim == 1, "recv_topk_weights must be 1D [recv_capacity]");
    nccl_topk_weights_out = ScopedNdTensor(make_tensor(recv_w_data, 1, ncclFloat32, recv_capacity));
  }

  ncclEpDispatchInputs_t in_struct{};
  in_struct.size = static_cast<unsigned int>(sizeof(in_struct));
  in_struct.tokens = nccl_tokens_in;
  in_struct.topk_weights = nccl_topk_weights_in;

  ncclEpDispatchOutputs_t out_struct{};
  out_struct.size = static_cast<unsigned int>(sizeof(out_struct));
  out_struct.tokens = nccl_tokens_out;
  out_struct.topk_weights = nccl_topk_weights_out;

  ncclEpDispatchConfig_t dispatch_cfg{};
  dispatch_cfg.size = static_cast<unsigned int>(sizeof(dispatch_cfg));

  ScopedEpHandle transient;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    HandleEntry& cfg = lookup_config(handle_id);
    transient =
        ScopedEpHandle(open_handle(handle_mem, cfg.handle_mem_size, cfg.top_k, cfg.alignment));
  }
  NVTE_CHECK_NCCL(ncclEpDispatch(transient, &in_struct, &out_struct,
                                 /*layout_info=*/nullptr, &dispatch_cfg, stream));
}

void EPBackend::combine(uint64_t handle_id, void* handle_mem, const NVTETensor expert_out,
                        NVTETensor result, cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape exp_shape = nvte_tensor_shape(expert_out);
  void* exp_data = nvte_tensor_data(expert_out);
  NVTEDType exp_dtype = nvte_tensor_type(expert_out);
  NVTE_CHECK(exp_data != nullptr, "expert_out data must not be null");

  ScopedNdTensor nccl_expert_in(make_payload_tensor(expert_out, 2, nvte_dtype_to_nccl(exp_dtype),
                                                    static_cast<unsigned int>(exp_shape.data[0]),
                                                    static_cast<unsigned int>(exp_shape.data[1])));

  NVTEShape res_shape = nvte_tensor_shape(result);
  void* res_data = nvte_tensor_data(result);
  NVTEDType res_dtype = nvte_tensor_type(result);
  NVTE_CHECK(res_data != nullptr, "result data must not be null");

  ScopedNdTensor nccl_result_out(make_tensor(res_data, 2, nvte_dtype_to_nccl(res_dtype),
                                             static_cast<unsigned int>(res_shape.data[0]),
                                             static_cast<unsigned int>(res_shape.data[1])));

  ncclEpCombineInputs_t in_struct{};
  in_struct.size = static_cast<unsigned int>(sizeof(in_struct));
  in_struct.tokens = nccl_expert_in;

  ncclEpCombineOutputs_t out_struct{};
  out_struct.size = static_cast<unsigned int>(sizeof(out_struct));
  out_struct.tokens = nccl_result_out;

  ScopedEpHandle transient;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    HandleEntry& cfg = lookup_config(handle_id);
    transient =
        ScopedEpHandle(open_handle(handle_mem, cfg.handle_mem_size, cfg.top_k, cfg.alignment));
  }
  NVTE_CHECK_NCCL(ncclEpCombine(transient, &in_struct, &out_struct, /*config=*/nullptr, stream));
}

void EPBackend::dispatch_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                             const NVTETensor g_recv_topk_weights, NVTETensor grad_tokens,
                             NVTETensor grad_topk_weights, cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape g_shape = nvte_tensor_shape(grad);
  void* g_data = nvte_tensor_data(grad);
  NVTEDType g_dtype = nvte_tensor_type(grad);
  NVTE_CHECK(g_data != nullptr, "grad data must not be null");
  ScopedNdTensor nccl_tok_in(make_payload_tensor(grad, 2, nvte_dtype_to_nccl(g_dtype),
                                                 static_cast<unsigned int>(g_shape.data[0]),
                                                 static_cast<unsigned int>(g_shape.data[1])));

  // g_recv_topk_weights must be 1D [recv_capacity] — caller flattens.
  NVTEShape gw_shape = nvte_tensor_shape(g_recv_topk_weights);
  void* gw_data = nvte_tensor_data(g_recv_topk_weights);
  NVTE_CHECK(gw_data != nullptr, "g_recv_topk_weights data must not be null");
  NVTE_CHECK(gw_shape.ndim == 1,
             "g_recv_topk_weights must be 1D [recv_capacity]; caller must flatten leading dims");
  const unsigned int recv_capacity = static_cast<unsigned int>(gw_shape.data[0]);
  ScopedNdTensor nccl_w_in(make_tensor(gw_data, 1, ncclFloat32, recv_capacity));

  NVTEShape gt_shape = nvte_tensor_shape(grad_tokens);
  void* gt_data = nvte_tensor_data(grad_tokens);
  NVTE_CHECK(gt_data != nullptr, "grad_tokens data must not be null");
  ScopedNdTensor nccl_tok_out(make_tensor(gt_data, 2, nvte_dtype_to_nccl(g_dtype),
                                          static_cast<unsigned int>(gt_shape.data[0]),
                                          static_cast<unsigned int>(gt_shape.data[1])));

  NVTEShape gtw_shape = nvte_tensor_shape(grad_topk_weights);
  void* gtw_data = nvte_tensor_data(grad_topk_weights);
  NVTE_CHECK(gtw_data != nullptr, "grad_topk_weights data must not be null");
  NVTE_CHECK(gtw_shape.ndim == 2, "grad_topk_weights must be 2D [T, top_k]");
  ScopedNdTensor nccl_w_out(make_tensor(gtw_data, 2, ncclFloat32,
                                        static_cast<unsigned int>(gtw_shape.data[0]),
                                        static_cast<unsigned int>(gtw_shape.data[1])));

  ncclEpCombineInputs_t in_struct{};
  in_struct.size = static_cast<unsigned int>(sizeof(in_struct));
  in_struct.tokens = nccl_tok_in;
  in_struct.topk_weights = nccl_w_in;

  ncclEpCombineOutputs_t out_struct{};
  out_struct.size = static_cast<unsigned int>(sizeof(out_struct));
  out_struct.tokens = nccl_tok_out;
  out_struct.topk_weights = nccl_w_out;

  ScopedEpHandle transient;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    HandleEntry& cfg = lookup_config(handle_id);
    transient =
        ScopedEpHandle(open_handle(handle_mem, cfg.handle_mem_size, cfg.top_k, cfg.alignment));
  }
  NVTE_CHECK_NCCL(ncclEpCombine(transient, &in_struct, &out_struct, /*config=*/nullptr, stream));
}

void EPBackend::combine_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                            NVTETensor grad_expert_out, cudaStream_t stream) {
  // Backward of combine = reverse-direction dispatch.
  dispatch(handle_id, handle_mem, /*topk_idx=*/nullptr, grad, /*topk_weights=*/nullptr,
           grad_expert_out, /*recv_topk_weights=*/nullptr, stream);
}

}  // namespace ep
}  // namespace transformer_engine
