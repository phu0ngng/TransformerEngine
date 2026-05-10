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
 *  Handle lifecycle (HT mode): the persistent state lives in handle_mem
 *  (caller-owned device buffer of size ncclEpHandleMemSize). The host-side
 *  ncclEpHandle_t is built on the fly per op via ncclEpInitHandle (pure pointer
 *  arithmetic over handle_mem) and torn down with ncclEpHandleDestroy at the
 *  end of each op. This keeps EPBackend stateless across ops.
 *  - prepare():     open + ncclEpUpdateHandle (collective; AllGather routing → handle_mem) + close
 *  - dispatch():    open + ncclEpDispatch + close
 *  - combine():     open + ncclEpCombine + close
 *  - dispatch_bwd → combine();   combine_bwd → dispatch() with no weights.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_ep.h>
#include <transformer_engine/ep.h>

#include <cstddef>
#include <mutex>

namespace transformer_engine {
namespace ep {

/*! \brief EP backend singleton — owns the NCCL EP group for the process lifetime. */
class EPBackend {
 public:
  /*! \brief Access the singleton. Aborts if not yet initialized. */
  static EPBackend& get();

  /*! \brief JAX path: bootstrap from uid via ncclCommInitRank + ncclCommSplit. */
  static void initialize(const ncclUniqueId& uid, int world_size, int rank,
                         NVTEEpGroupConfig config);

  /*! \brief PyTorch path: bootstrap from an existing EP sub-communicator.
   *
   *  Borrows ep_comm — caller (PyTorch) retains ownership and must outlive the
   *  EP backend. Not destroyed by EPBackend. ep_comm must span exactly
   *  config.ep_size ranks.
   */
  static void initialize_with_comm(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  size_t get_handle_mem_size(NVTEEpLayerConfig layer_config);

  void prepare(const NVTETensor topk_idx, NVTETensor token_counts, void* handle_mem,
               cudaStream_t stream);

  void dispatch(void* handle_mem, const NVTETensor topk_idx, const NVTETensor tokens,
                const NVTETensor topk_weights, NVTETensor recv_tokens, NVTETensor recv_topk_weights,
                cudaStream_t stream);

  void combine(void* handle_mem, const NVTETensor expert_out, NVTETensor result,
               cudaStream_t stream);

  void dispatch_bwd(void* handle_mem, const NVTETensor grad, NVTETensor grad_tokens,
                    cudaStream_t stream);

  void combine_bwd(void* handle_mem, const NVTETensor grad, NVTETensor grad_expert_out,
                   cudaStream_t stream);

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
  ncclEpHandle_t open_handle(void* handle_mem, int num_topk);
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
  // Persistent handle from the most recent prepare(). NCCL stores num_tokens
  // / num_topk on the host-side handle struct (set during ncclEpUpdateHandle),
  // and HT combine asserts on it — so we keep the handle alive across the
  // prepare → dispatch → combine cycle instead of reopening per op.
  ncclEpHandle_t cur_handle_{nullptr};
  void* cur_handle_mem_{nullptr};
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
