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
 *  The EPBackend singleton is never exposed.
 *
 *  All per-step ops (prepare, dispatch, combine, and their backward
 *  variants) are allocation-free and CUDA graph-capturable.
 */

#ifndef TRANSFORMER_ENGINE_EP_H_
#define TRANSFORMER_ENGINE_EP_H_

// transformer_engine.h defines NVTETensor, NVTEDType, etc.
#include <cuda_runtime_api.h>
#include <stddef.h>  // size_t
#include <stdint.h>  // uint8_t
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Config structs ─────────────────────────────────────────────────────── */

/*! \brief Group-level EP configuration (fixed for the lifetime of the EP group).
 *
 *  Passed to nvte_ep_initialize. All fields are static for the process lifetime.
 */
typedef struct {
  int ep_size;                  /*!< EP world size (number of GPUs in EP domain). */
  int num_experts;              /*!< Total experts across all ranks. */
  int max_tokens_per_rank;      /*!< Static upper bound on tokens this rank will SEND
                            *   per dispatch. Fixed for CUDA graph. */
  int max_recv_tokens_per_rank; /*!< Static upper bound on tokens this rank will RECEIVE
                                 *   per dispatch (= recv_capacity_max). Must be > 0
                                 *   (NCCL EP has no auto-default). Size for worst-case
                                 *   top_k fan-out (e.g. ep_size * max_tokens_per_rank * top_k). */
  int hidden_dim;               /*!< Token hidden dimension. */
} NVTEEpGroupConfig;

/*! \brief Per-layer EP configuration (fixed per layer at construction).
 *
 *  Passed to nvte_ep_get_handle_mem_size and nvte_ep_prepare. Groups all
 *  non-tensor, non-stream per-layer attributes.
 */
typedef struct {
  int num_local_experts; /*!< Reserved; not consumed by the backend today
                              (derived from group_config.num_experts / ep_size at
                              nvte_ep_initialize time). Kept for ABI stability. */
  int top_k;             /*!< Per-token expert fan-out. Required (handle_mem size scales
                              with top_k). */
  /* topk_format   — reserved; only sparse int64 routing supported today */
  /* scaling_mode  — reserved; FP8 block-scaling support planned */
} NVTEEpLayerConfig;

/* ── Bootstrap ──────────────────────────────────────────────────────────── */

/*! \brief Bootstrap from a serialized ncclUniqueId — JAX path.
 *
 *  TE creates a world-sized NCCL comm from the uid internally:
 *    ncclCommInitRank(&world_comm, world_size, uid, rank)
 *  then splits it into the EP sub-communicator and initializes the backend.
 *  The world comm is destroyed immediately after the split.
 *
 *  Use this path for JAX, which does not expose its internal ncclComm_t.
 *  For PyTorch use nvte_ep_initialize_with_comm() instead.
 *
 *  Requires SM_90+ (Hopper or later).
 *
 *  \param[in] unique_id_bytes  Pointer to 128-byte ncclUniqueId (broadcast by caller).
 *  \param[in] world_size       Total number of ranks.
 *  \param[in] rank             This process's rank.
 *  \param[in] group_config     Group-level EP configuration.
 */
void nvte_ep_initialize(const uint8_t* unique_id_bytes, int world_size, int rank,
                        NVTEEpGroupConfig group_config);

/*! \brief Bootstrap from an existing NCCL EP sub-communicator — PyTorch path.
 *
 *  For frameworks that already hold an ncclComm_t for the EP process group
 *  (e.g. PyTorch via torch.distributed), pass it here instead of going
 *  through the uid bootstrap. TE takes ownership and destroys it after
 *  the backend is initialized.
 *
 *  ep_comm must span exactly group_config.ep_size ranks. Passing the world
 *  communicator is an error and will be caught at runtime.
 *
 *  Requires SM_90+ (Hopper or later).
 *
 *  \param[in] ep_comm      Opaque ncclComm_t for the EP sub-group (cast to void*).
 *  \param[in] group_config Group-level EP configuration.
 */
void nvte_ep_initialize_with_comm(void* ep_comm, NVTEEpGroupConfig group_config);

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
 *  num_local_experts is NOT passed here — it is derived from the cached
 *  group config (num_experts / ep_size) at nvte_ep_initialize time.
 *
 *  \param[in]     topk_idx      [T, top_k] int64 sparse routing indices.
 *  \param[out]    token_counts  Per-local-expert token counts [num_local_experts] int32.
 *  \param[in,out] handle_mem    uint8 device buffer sized by nvte_ep_get_handle_mem_size().
 *                               Holds NCCL EP routing tensors only — no host-side header.
 *                               No zero-init required.
 *  \param[in]     stream        CUDA stream.
 */
void nvte_ep_prepare(NVTETensor topk_idx, NVTETensor token_counts, NVTETensor handle_mem,
                     cudaStream_t stream);

/*! \brief Dispatch tokens (and routing weights) to expert ranks.
 *
 *  Sends tokens and topk_weights to the destination expert ranks according
 *  to the routing computed in nvte_ep_prepare. NCCL EP routes 3 inputs
 *  (tokens, topk_weights, topk_idx) and writes back 2 outputs
 *  (recv_tokens, recv_topk_weights).
 *
 *  topk_idx is consumed directly here (not cached by prepare) so the backend
 *  can be stateless across per-step ops — each call rebuilds a transient
 *  ncclEpHandle as a host-side view over the device handle_mem buffer.
 *
 *  \param[in]     handle_mem         Handle memory from nvte_ep_prepare.
 *  \param[in]     topk_idx           [T, top_k] int64 sparse routing indices.
 *  \param[in]     tokens             Input tokens [T, hidden_dim].
 *  \param[in]     topk_weights       SPARSE: [T, top_k] float32; DENSE: pass null NVTETensor.
 *  \param[out]    recv_tokens        Received tokens [recv_T, hidden_dim].
 *  \param[out]    recv_topk_weights  Received per-slot weights [recv_T] float32
 *                                    (1 weight per slot). Pass null NVTETensor
 *                                    in backward (no weights to scatter back).
 *  \param[in]     stream             CUDA stream.
 */
void nvte_ep_dispatch(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor tokens,
                      NVTETensor topk_weights, NVTETensor recv_tokens, NVTETensor recv_topk_weights,
                      cudaStream_t stream);

/*! \brief Combine expert outputs back to originating ranks (unweighted sum).
 *
 *  NCCL EP forward combine performs an UNWEIGHTED sum of the top_k expert
 *  contributions per token. The caller is responsible for pre-multiplying
 *  expert_out by recv_topk_weights (from nvte_ep_dispatch's 2nd output)
 *  before calling this if weighted reduction is desired. topk_weights are
 *  only consumed by combine in the backward path (nvte_ep_combine_bwd).
 *
 *  \param[in]  handle_mem  Handle memory from nvte_ep_prepare.
 *  \param[in]  expert_out  Post-hadamard, masked expert outputs
 *                          [recv_T, hidden_dim] — caller has already applied
 *                          `expert_out * recv_topk_weights * mask`. NCCL EP
 *                          combine is an UNWEIGHTED scatter-sum, so caller
 *                          weighting must happen before this call.
 *  \param[out] result      Combined output [T, hidden_dim].
 *  \param[in]  stream      CUDA stream.
 */
void nvte_ep_combine(NVTETensor handle_mem, NVTETensor expert_out, NVTETensor result,
                     cudaStream_t stream);

/*! \brief Backward of dispatch (combine direction in backward pass).
 *
 *  \param[in]  handle_mem   Handle memory from nvte_ep_prepare.
 *  \param[in]  grad         Gradient w.r.t. recv_tokens [recv_capacity, hidden_dim].
 *  \param[out] grad_tokens  Gradient w.r.t. tokens [T, hidden_dim].
 *  \param[in]  stream       CUDA stream.
 */
void nvte_ep_dispatch_bwd(NVTETensor handle_mem, NVTETensor grad, NVTETensor grad_tokens,
                          cudaStream_t stream);

/*! \brief Backward of combine (dispatch direction in backward pass).
 *
 *  Equivalent to a forward dispatch of `grad` with no topk_weights — the
 *  output is the gradient of `expert_out` from the forward call. Padded
 *  slots in `grad_expert_out` receive zero from NCCL EP.
 *
 *  \param[in]  handle_mem       Handle memory from nvte_ep_prepare.
 *  \param[in]  grad             Gradient w.r.t. result [T, hidden_dim].
 *  \param[out] grad_expert_out  Gradient w.r.t. expert_out [recv_capacity, hidden_dim].
 *  \param[in]  stream           CUDA stream.
 */
void nvte_ep_combine_bwd(NVTETensor handle_mem, NVTETensor grad, NVTETensor grad_expert_out,
                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_H_
