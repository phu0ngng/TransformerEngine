/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.h
 *  \brief EP backend — process-lifetime singleton wrapping the NCCL EP library.
 *
 *  Internal to transformer_engine/common/ep/. Not part of the public API.
 *
 *  Handle lifecycle: NCCL EP keeps device-side routing state in handle_mem
 *  (caller-owned uint8 buffer of size ncclEpHandleMemSize). The host-side
 *  ncclEpHandle_t is opened by prepare() against a specific handle_mem buffer
 *  and CACHED in `handles_` keyed by a uint64_t handle_id allocated by an
 *  internal atomic counter at prepare() time. The id flows through @jax.jit
 *  as an int64[1] device tensor that the FFI reads on each subsequent op —
 *  this decouples the cache key from the handle_mem device pointer, which
 *  XLA may relocate between primitive boundaries.
 *  combine() reads host-side fields populated by ncclEpUpdateHandle
 *  (notably handle->num_tokens). Per-layer / pipeline-parallel friendly: each
 *  prepare gets a unique id and therefore an independent entry.
 *  - prepare():     allocate new id, open cache entry + ncclEpUpdateHandle
 *                   (collective; AllGather routing → handle_mem). Returns id.
 *  - dispatch():    lookup entry by handle_id + ncclEpDispatch. Reopens the
 *                   NCCL handle if handle_mem moved since the cached open.
 *  - combine():     lookup entry by handle_id + ncclEpCombine.
 *  - dispatch_bwd → combine();   combine_bwd → dispatch() with no weights.
 *  Cache eviction: LRU with cap NVTE_EP_HANDLE_CACHE_SIZE (default 64).
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_ep.h>
#include <transformer_engine/ep.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>

namespace transformer_engine {
namespace ep {

/*! \brief EP backend singleton — owns the NCCL EP group for the process lifetime. */
class EPBackend {
 public:
  /*! \brief Access the singleton. Aborts if not yet initialized. */
  static EPBackend& get();

  /*! \brief JAX path: bootstrap a per-DP-group EP comm directly via ncclCommInitRank.
   *
   *  Caller (JAX bootstrap) computes a distinct ncclUniqueId per DP color and
   *  passes the rank within the EP group (i.e. world_rank % ep_size). No
   *  ncclCommSplit — each EP group lives on its own root comm so that two
   *  EP groups colocated on one physical node remain fully independent.
   */
  static void initialize(const ncclUniqueId& uid, int ep_size, int rank_within_group,
                         NVTEEpGroupConfig config);

  /*! \brief PyTorch path: bootstrap from an existing EP sub-communicator.
   *
   *  Borrows ep_comm — caller (PyTorch) retains ownership and must outlive the
   *  EP backend. Not destroyed by EPBackend. ep_comm must span exactly
   *  config.ep_size ranks.
   */
  static void initialize_with_comm(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  size_t get_handle_mem_size(NVTEEpLayerConfig layer_config);

  // prepare allocates a fresh uint64_t handle_id from an atomic counter, opens
  // (or reopens) the NCCL handle against handle_mem, and caches it under that
  // id. Returns the new id; the FFI writes it into a 1-element int64 device
  // buffer that flows alongside handle_mem through @jax.jit to subsequent ops.
  uint64_t prepare(const NVTETensor topk_idx, NVTETensor token_counts, void* handle_mem,
                   size_t dispatch_output_per_expert_alignment, cudaStream_t stream);

  void dispatch(uint64_t handle_id, void* handle_mem, const NVTETensor topk_idx,
                const NVTETensor tokens, const NVTETensor topk_weights, NVTETensor recv_tokens,
                NVTETensor recv_topk_weights, cudaStream_t stream);

  void combine(uint64_t handle_id, void* handle_mem, const NVTETensor expert_out, NVTETensor result,
               cudaStream_t stream);

  void dispatch_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                    NVTETensor grad_tokens, cudaStream_t stream);

  void combine_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                   NVTETensor grad_expert_out, cudaStream_t stream);

 private:
  EPBackend() = default;
  ~EPBackend();
  EPBackend(const EPBackend&) = delete;
  EPBackend& operator=(const EPBackend&) = delete;

  // Shared finalizer called by both initialize paths after bootstrap.
  // owns_comm=true means EPBackend will ncclCommDestroy(ep_comm) at teardown
  // (JAX path: TE created the comm via CommSplit). false means caller owns
  // (PyTorch path: comm comes from torch.distributed).
  void init(ncclComm_t ep_comm, NVTEEpGroupConfig config, bool owns_comm);

  static EPBackend& instance();  // Meyers singleton accessor
  static void validate_config(const NVTEEpGroupConfig& config);

  ncclNDTensor_t make_tensor(void* data, unsigned int ndim, ncclDataType_t datatype,
                             ncclEpTensorTag_t tag, unsigned int size0, unsigned int size1 = 1,
                             unsigned int size2 = 1, unsigned int size3 = 1,
                             unsigned int size4 = 1);

  void destroy_tensor(ncclNDTensor_t tensor);
  static ncclDataType_t nvte_dtype_to_nccl(NVTEDType dtype);
  // Build a transient ncclEpHandle from the device handle_mem buffer.
  // ncclEpInitHandle is pure host-side pointer arithmetic over an existing
  // handle_mem block — it does not touch the device data populated by a
  // prior ncclEpUpdateHandle, so we can rebuild the host view per op.
  // num_topk must match what was used during prepare()'s ncclEpUpdateHandle so
  // ncclEpDispatch's internal assertion (topk_weights.sizes[1] == handle->num_topk)
  // passes. Pass -1 for paths that don't carry per-token weights (combine, bwd dispatch).
  ncclEpHandle_t open_handle(void* handle_mem, int num_topk,
                             size_t dispatch_output_per_expert_alignment);
  void close_handle(ncclEpHandle_t handle);

  ncclEpGroup_t ep_group_{nullptr};
  // Underlying NCCL communicator the EP group was built from. Kept alive
  // for the EP group's lifetime — ncclEpGroupDestroy depends on it.
  // Destroyed by EPBackend only when owns_ep_comm_ (JAX path).
  ncclComm_t ep_comm_{nullptr};
  bool owns_ep_comm_{false};
  NVTEEpGroupConfig group_config_{};
  bool initialized_{false};
  size_t routing_buf_size_{0};
  std::mutex mutex_;
  // Per-handle_id cache (LRU). Key is the uint64_t id assigned at prepare time
  // (NOT the handle_mem device pointer — under @jax.jit XLA may relocate
  // buffers between primitive boundaries, so the device pointer is not stable).
  // Entry holds the opened NCCL handle, the handle_mem pointer it was opened
  // against, and the (alignment, top_k) config. Multiple layers / PP stages
  // each get a unique id from next_handle_id_ and therefore independent entries.
  struct HandleEntry {
    ncclEpHandle_t handle;
    void* handle_mem;
    size_t alignment;
    int top_k;
  };
  std::list<uint64_t> lru_;  // front = most recently used
  std::unordered_map<uint64_t, std::pair<HandleEntry, std::list<uint64_t>::iterator>> handles_;
  std::atomic<uint64_t> next_handle_id_{1};  // 0 reserved as "no id"
  size_t handle_cache_cap_{0};               // set lazily from NVTE_EP_HANDLE_CACHE_SIZE

  // Allocate a fresh uint64_t handle_id, open the NCCL handle against
  // handle_mem, and insert under the new id. Caller must hold mutex_.
  uint64_t insert_new_entry(void* handle_mem, int top_k, size_t alignment);
  // Lookup a cache entry by handle_id; reopens the NCCL handle if handle_mem
  // moved since the cached open. Aborts if id not present. Holds mutex_.
  HandleEntry& lookup_entry(uint64_t handle_id, void* handle_mem);
  // Bound the cache to handle_cache_cap_ entries; closes the LRU tail.
  void evict_if_full();
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
