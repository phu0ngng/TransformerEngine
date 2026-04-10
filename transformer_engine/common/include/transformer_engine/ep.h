/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep.h
 *  \brief Stable public C API for Expert Parallelism (EP).
 *
 *  This is the only public surface for EP functionality. All framework
 *  extensions (TE/PyTorch, TE/JAX) call these functions. Internal
 *  classes (EPManager, EPBackend, NCCLEPBackend) are never exposed.
 *
 *  All per-step ops (prepare, dispatch, combine, and their backward
 *  variants) are allocation-free and CUDA graph-capturable.
 *
 *  High Throughput (HT) mode only.
 */

#ifndef TRANSFORMER_ENGINE_EP_H_
#define TRANSFORMER_ENGINE_EP_H_

// transformer_engine.h defines NVTETensor, NVTEDType, NVTEScalingMode, etc.
#include <transformer_engine/transformer_engine.h>

#include <cuda_runtime_api.h>
#include <stddef.h>   // size_t
#include <stdint.h>   // uint8_t

#ifdef __cplusplus
extern "C" {
#endif

/* ── Enums ──────────────────────────────────────────────────────────────── */

/*! \brief Topk routing format for EP dispatch/combine. */
typedef enum {
  NVTE_EP_TOPK_FORMAT_SPARSE = 0,  /*!< [T, top_k] int64 indices + [T, top_k] float32 weights */
  NVTE_EP_TOPK_FORMAT_DENSE  = 1,  /*!< [T, E] float32 masked probability map */
} NVTEEpTopkFormat;

/* ── Config structs ─────────────────────────────────────────────────────── */

/*! \brief Group-level EP configuration (fixed for the lifetime of the EP group).
 *
 *  Passed to nvte_ep_initialize. All fields are static for the process lifetime.
 */
typedef struct {
  int ep_size;              /*!< EP world size (number of GPUs in EP domain). */
  int num_experts;          /*!< Total experts across all ranks. */
  int max_tokens_per_rank;  /*!< Static upper bound on tokens per rank; fixed for CUDA graph. */
  int hidden_dim;           /*!< Token hidden dimension. */
} NVTEEpGroupConfig;

/*! \brief Per-layer EP configuration (fixed per layer at construction).
 *
 *  Passed to nvte_ep_get_handle_mem_size and nvte_ep_prepare. Groups all
 *  non-tensor, non-stream per-layer attributes.
 */
typedef struct {
  int num_local_experts;        /*!< Experts hosted on this rank. */
  NVTEEpTopkFormat topk_format; /*!< Routing format: SPARSE or DENSE. */
  NVTEScalingMode scaling_mode; /*!< Scaling mode. Currently NVTE_DELAYED_TENSOR_SCALING (no quant).
                                     FP8 block scaling modes may be added in future. */
} NVTEEpLayerConfig;

/* ── Bootstrap ──────────────────────────────────────────────────────────── */

/*! \brief Bootstrap — called once per process. Returns nothing.
 *
 *  TE/Common creates its own world-sized NCCL communicator internally:
 *    ncclCommInitRank(&world_comm, world_size, uid, rank)
 *  then splits it into the EP sub-communicator:
 *    ncclCommSplit(world_comm, color = rank / ep_size, ...)
 *  caches the EP sub-comm in EPManager, then destroys the temporary world comm.
 *
 *  unique_id_bytes must be a 128-byte ncclUniqueId, identical on all ranks.
 *  The caller generates it on rank 0 and broadcasts it via the framework's
 *  own distributed primitives before calling this function.
 *
 *  Rationale: neither PyTorch nor JAX exposes its internal ncclComm_t to
 *  external libraries. TE/Common therefore bootstraps its own communicator
 *  from the unique_id, making both frameworks follow the same code path.
 *
 *  Requires SM_90+ (Hopper or later). Uses High Throughput (HT) mode.
 *
 *  \param[in] unique_id_bytes  Pointer to 128-byte ncclUniqueId (broadcast by caller).
 *  \param[in] world_size       Total number of ranks.
 *  \param[in] rank             This process's rank.
 *  \param[in] group_config     Group-level EP configuration.
 */
void nvte_ep_initialize(const uint8_t* unique_id_bytes,
                        int world_size,
                        int rank,
                        NVTEEpGroupConfig group_config);

/* ── Per-layer size query ────────────────────────────────────────────────── */

/*! \brief Query handle_mem size for a per-layer config.
 *
 *  Called once per layer at construction; result is static for fixed config.
 *
 *  \param[in] layer_config  Per-layer EP configuration.
 *  \return Size in bytes of the handle_mem buffer.
 */
size_t nvte_ep_get_handle_mem_size(NVTEEpLayerConfig layer_config);

/* ── Per-step ops (all allocation-free, CUDA graph-capturable) ──────────── */

/*! \brief Routing preparation: AllGather routing map, compute token counts.
 *
 *  Exchanges topk_idx across all ranks (AllGather) so each rank knows how
 *  many tokens it will receive per local expert. Stores routing metadata
 *  in handle_mem for use by dispatch and combine.
 *
 *  topk_weights are NOT passed here — they travel alongside tokens during
 *  dispatch (see nvte_ep_dispatch).
 *
 *  \param[in]     topk_idx      SPARSE: [T, top_k] int64; DENSE: [T, E] float32.
 *  \param[in]     layer_config  Per-layer EP configuration.
 *  \param[out]    token_counts  Per-local-expert token counts [num_local_experts] int32.
 *  \param[in,out] handle_mem    Pre-allocated uint8 buffer (sized by nvte_ep_get_handle_mem_size).
 *  \param[in]     stream        CUDA stream.
 */
void nvte_ep_prepare(NVTETensor topk_idx,
                     NVTEEpLayerConfig layer_config,
                     NVTETensor token_counts,
                     NVTETensor handle_mem,
                     cudaStream_t stream);

/*! \brief Dispatch tokens (and routing weights) to expert ranks.
 *
 *  Sends tokens and topk_weights to the destination expert ranks according
 *  to the routing computed in nvte_ep_prepare. In HT mode, weights are sent
 *  alongside tokens as a second tensor; the expert rank receives both and
 *  stores the weights in handle_mem for use by nvte_ep_combine.
 *
 *  \param[in]     handle_mem    Handle memory from nvte_ep_prepare.
 *  \param[in]     tokens        Input tokens [T, hidden_dim].
 *  \param[in]     topk_weights  SPARSE: [T, top_k] float32; DENSE: pass null NVTETensor.
 *  \param[out]    recv_tokens   Received tokens [recv_T, hidden_dim].
 *  \param[in]     stream        CUDA stream.
 */
void nvte_ep_dispatch(NVTETensor handle_mem,
                      NVTETensor tokens,
                      NVTETensor topk_weights,
                      NVTETensor recv_tokens,
                      cudaStream_t stream);

/*! \brief Combine expert outputs with weighted accumulation.
 *
 *  Gathers expert outputs back to the originating ranks and applies
 *  weighted accumulation using the weights received during nvte_ep_dispatch.
 *
 *  \param[in]  handle_mem  Handle memory from nvte_ep_prepare.
 *  \param[in]  expert_out  Expert outputs [recv_T, hidden_dim].
 *  \param[out] result      Combined output [T, hidden_dim].
 *  \param[in]  stream      CUDA stream.
 */
void nvte_ep_combine(NVTETensor handle_mem,
                     NVTETensor expert_out,
                     NVTETensor result,
                     cudaStream_t stream);

/*! \brief Backward of dispatch (combine direction in backward pass).
 *
 *  \param[in]  handle_mem  Handle memory from nvte_ep_prepare.
 *  \param[in]  grad        Gradient w.r.t. recv_tokens.
 *  \param[out] result      Gradient w.r.t. tokens.
 *  \param[in]  stream      CUDA stream.
 */
void nvte_ep_dispatch_bwd(NVTETensor handle_mem,
                          NVTETensor grad,
                          NVTETensor result,
                          cudaStream_t stream);

/*! \brief Backward of combine (dispatch direction in backward pass).
 *
 *  \param[in]  handle_mem  Handle memory from nvte_ep_prepare.
 *  \param[in]  grad        Gradient w.r.t. result.
 *  \param[out] result      Gradient w.r.t. expert_out.
 *  \param[in]  stream      CUDA stream.
 */
void nvte_ep_combine_bwd(NVTETensor handle_mem,
                         NVTETensor grad,
                         NVTETensor result,
                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_H_
