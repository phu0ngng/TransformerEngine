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
 *  HT mode handle lifecycle (per step):
 *  - prepare():     ncclEpInitHandle (maps routing bufs) + ncclEpUpdateHandle (collective)
 *  - dispatch():    ncclEpDispatch using handle from prepare()
 *  - combine():     ncclEpCombine + ncclEpHandleDestroy (forward handle freed)
 *  - combine_bwd(): ncclEpInitHandle (remap, no collective) + ncclEpDispatch
 *  - dispatch_bwd() → combine(): ncclEpCombine + ncclEpHandleDestroy (backward handle freed)
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_

#include <cstddef>
#include <mutex>
#include <cuda_runtime_api.h>

#include <transformer_engine/ep.h>
#include <nccl.h>
#include <nccl_ep.h>

namespace transformer_engine {
namespace ep {

// Defined in ep_backend.cpp — opaque to callers
struct HandleMemHeader;

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
   *  Takes ownership of ep_comm (destroys it after ncclEpCreateGroup).
   *  ep_comm must span exactly config.ep_size ranks.
   */
  static void initialize_with_comm(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  size_t get_handle_mem_size(NVTEEpLayerConfig layer_config);

  void prepare(void* handle_mem,
               const NVTETensor topk_idx,
               NVTETensor token_counts,
               NVTEEpLayerConfig layer_config,
               cudaStream_t stream);

  void dispatch(void* handle_mem,
                const NVTETensor tokens,
                const NVTETensor topk_weights,
                NVTETensor recv_tokens,
                cudaStream_t stream);

  void combine(void* handle_mem,
               const NVTETensor expert_out,
               NVTETensor result,
               cudaStream_t stream);

  void dispatch_bwd(void* handle_mem,
                    const NVTETensor grad,
                    NVTETensor result,
                    cudaStream_t stream);

  void combine_bwd(void* handle_mem,
                   const NVTETensor grad,
                   NVTETensor result,
                   cudaStream_t stream);

 private:
  EPBackend() = default;
  ~EPBackend();
  EPBackend(const EPBackend&) = delete;
  EPBackend& operator=(const EPBackend&) = delete;

  // Shared finalizer called by both initialize paths after bootstrap.
  void init(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  static EPBackend& instance();   // Meyers singleton accessor
  static void validate_config(const NVTEEpGroupConfig& config);

  ncclNDTensor_t make_tensor(void* data, unsigned int ndim,
                             ncclDataType_t datatype,
                             ncclEpTensorTag_t tag,
                             unsigned int size0,
                             unsigned int size1 = 1,
                             unsigned int size2 = 1,
                             unsigned int size3 = 1,
                             unsigned int size4 = 1);

  void destroy_tensor(ncclNDTensor_t tensor);
  static ncclDataType_t nvte_dtype_to_nccl(NVTEDType dtype);
  void reinit_handle(HandleMemHeader* hdr, void* handle_mem);
  void destroy_handle(HandleMemHeader* hdr);

  ncclEpGroup_t    ep_group_{nullptr};
  NVTEEpGroupConfig group_config_{};
  bool             initialized_{false};
  size_t           routing_buf_size_{0};
  std::mutex       mutex_;
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
