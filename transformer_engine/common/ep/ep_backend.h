/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.h
 *  \brief Internal abstract backend interface for Expert Parallelism.
 *
 *  NOT part of the public API. Internal to transformer_engine/common/ep/.
 *  The interface may change as EP kernels are brought in-house.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_

#include <cstddef>
#include <cuda_runtime_api.h>

// ep.h provides the public NVTEEpTopkFormat enum and NVTEScalingMode
// (via transformer_engine.h).
#include <transformer_engine/ep.h>

// Forward-declare ncclComm (the struct behind ncclComm_t) so that this
// internal header does not require <nccl.h> on the include path.
// Only files that actually call NCCL APIs (ep_manager.cpp,
// nccl_ep_backend.cpp) include <nccl.h> directly.
struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace transformer_engine {
namespace ep {

// NVTEEpGroupConfig and NVTEEpLayerConfig are defined in ep.h (included above).
// They are the single source of truth for EP configuration — no internal
// duplicates needed.

/*! \brief Abstract EP backend interface.
 *
 *  Currently: NCCLEPBackend delegates all work to NCCL EP (HT mode only).
 *  Future: PluggableEPBackend will host kernels in-house with pluggable transport.
 */
class EPBackend {
 public:
  virtual ~EPBackend() = default;

  /*! \brief Bootstrap — called once at program start with the EP sub-comm.
   *
   *  The backend allocates persistent device buffers (RDMA staging, AllGather
   *  workspace) and registers memory.
   *
   *  \param[in] ep_comm       NCCL communicator for the EP sub-group.
   *  \param[in] group_config  Group-level configuration.
   */
  virtual void init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) = 0;

  /*! \brief Query handle_mem size in bytes for a given per-layer config.
   *
   *  Called once per layer at construction; result is static for fixed config.
   *
   *  \param[in] layer_config  Per-layer configuration.
   *  \return Size in bytes of the handle_mem buffer needed.
   */
  virtual size_t get_handle_mem_size(NVTEEpLayerConfig layer_config) = 0;

  /*! \brief Per-step routing preparation (allocation-free, graph-capturable).
   *
   *  Creates an ncclEpHandle_t via ncclEpCreateHandle, which performs
   *  the metadata exchange (AllGather of routing maps) in HT mode.
   *  Stores the handle and metadata in handle_mem for dispatch/combine.
   *
   *  topk_weights are NOT passed here — they travel alongside tokens
   *  during dispatch (see dispatch()).
   *
   *  \param[in,out] handle_mem    Pre-allocated buffer, sized by get_handle_mem_size().
   *  \param[in]     topk_idx      Routing indices: SPARSE [T,k] int64, or DENSE [T,E] float32.
   *  \param[out]    token_counts  Per-local-expert token counts [num_local_experts] int32.
   *  \param[in]     layer_config  Per-layer configuration.
   *  \param[in]     stream        CUDA stream.
   */
  virtual void prepare(void* handle_mem,
                       const NVTETensor topk_idx,
                       NVTETensor token_counts,
                       NVTEEpLayerConfig layer_config,
                       cudaStream_t stream) = 0;

  /*! \brief Dispatch tokens (and routing weights) to expert ranks
   *         (allocation-free, graph-capturable).
   *
   *  In HT mode, topk_weights are sent alongside tokens as a second
   *  input tensor; the expert rank receives both and stores the weights
   *  in handle_mem for use by combine().
   *
   *  \param[in]  handle_mem    Handle memory from prepare().
   *  \param[in]  tokens        Input tokens [T, hidden_dim].
   *  \param[in]  topk_weights  SPARSE: [T, top_k] float32; DENSE: pass null NVTETensor.
   *  \param[out] recv_tokens   Received tokens after dispatch.
   *  \param[in]  stream        CUDA stream.
   */
  virtual void dispatch(void* handle_mem,
                        const NVTETensor tokens,
                        const NVTETensor topk_weights,
                        NVTETensor recv_tokens,
                        cudaStream_t stream) = 0;

  /*! \brief Combine expert outputs back to original token order (allocation-free).
   *
   *  \param[in]  handle_mem   Handle memory from prepare().
   *  \param[in]  expert_out   Expert outputs.
   *  \param[out] result       Combined output [T, hidden_dim].
   *  \param[in]  stream       CUDA stream.
   */
  virtual void combine(void* handle_mem,
                       const NVTETensor expert_out,
                       NVTETensor result,
                       cudaStream_t stream) = 0;

  /*! \brief Backward of dispatch (combine direction in backward pass). */
  virtual void dispatch_bwd(void* handle_mem,
                            const NVTETensor grad,
                            NVTETensor result,
                            cudaStream_t stream) = 0;

  /*! \brief Backward of combine (dispatch direction in backward pass). */
  virtual void combine_bwd(void* handle_mem,
                           const NVTETensor grad,
                           NVTETensor result,
                           cudaStream_t stream) = 0;
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
