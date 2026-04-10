/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nccl_ep_backend.h
 *  \brief NCCL EP backend: delegates all kernel + transport work to NCCL EP.
 *
 *  Internal to transformer_engine/common/ep/.
 *  Uses High Throughput (HT) mode only.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_BACKEND_H_

#include "ep_backend.h"

// nccl_ep.h provides ncclEpGroup_t, ncclEpHandle_t, ncclNDTensor_t, etc.
// The build system must ensure the NCCL EP include directory is in the
// include path (either from the submodule or an external install).
#include <nccl_ep.h>

namespace transformer_engine {
namespace ep {

/*! \brief EP backend wrapping NCCL EP (HT mode only).
 *
 *  Owns the ncclEpGroup_t for its lifetime. All per-step ops create
 *  ephemeral ncclEpHandle_t + ncclNDTensor_t wrappers around user-provided
 *  buffers — no allocations, negligible overhead.
 *
 *  HT mode differences from LL mode:
 *  - ncclEpCreateHandle: accepts optional RECV_EXPERT_COUNTER local tensor
 *  - ncclEpDispatch: 3 inputs (tokens + topk_weights + topk_idx), no local tensors
 *  - ncclEpCombine: 1 input (expert_out), no local tensors
 *  - Output tensors are 2D [recv_T, dim], not 3D [experts, recv_T, dim]
 */
class NCCLEPBackend : public EPBackend {
 public:
  NCCLEPBackend() = default;
  ~NCCLEPBackend() override;

  void init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) override;
  size_t get_handle_mem_size(NVTEEpLayerConfig layer_config) override;
  void prepare(void* handle_mem,
               const NVTETensor topk_idx,
               NVTETensor token_counts,
               NVTEEpLayerConfig layer_config,
               cudaStream_t stream) override;
  void dispatch(void* handle_mem,
                const NVTETensor tokens,
                const NVTETensor topk_weights,
                NVTETensor recv_tokens,
                cudaStream_t stream) override;
  void combine(void* handle_mem,
               const NVTETensor expert_out,
               NVTETensor result,
               cudaStream_t stream) override;
  void dispatch_bwd(void* handle_mem,
                    const NVTETensor grad,
                    NVTETensor result,
                    cudaStream_t stream) override;
  void combine_bwd(void* handle_mem,
                   const NVTETensor grad,
                   NVTETensor result,
                   cudaStream_t stream) override;

 private:
  /*! \brief Create an ncclNDTensor_t wrapping a user-provided buffer.
   *
   *  The tensor does NOT own the data — ncclEpTensorDestroy only frees
   *  the handle, not the buffer.
   *
   *  \param[in] data     Device pointer to the buffer.
   *  \param[in] ndim     Number of dimensions.
   *  \param[in] datatype NCCL data type.
   *  \param[in] tag      Tensor tag identifying its role.
   *  \param[in] sizes    Dimension sizes (up to 5).
   *  \return ncclNDTensor_t handle.
   */
  ncclNDTensor_t make_tensor(void* data, unsigned int ndim,
                             ncclDataType_t datatype,
                             ncclEpTensorTag_t tag,
                             unsigned int size0,
                             unsigned int size1 = 1,
                             unsigned int size2 = 1,
                             unsigned int size3 = 1,
                             unsigned int size4 = 1);

  /*! \brief Destroy an ncclNDTensor_t created by make_tensor(). */
  void destroy_tensor(ncclNDTensor_t tensor);

  /*! \brief Map NVTETensor dtype to ncclDataType_t. */
  static ncclDataType_t nvte_dtype_to_nccl(NVTEDType dtype);

  ncclEpGroup_t ep_group_{nullptr};
  NVTEEpGroupConfig group_config_{};
  bool initialized_{false};
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_BACKEND_H_
