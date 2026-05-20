/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep.h
 *  \brief Public C API for Expert Parallelism. Per-step ops are allocation-free
 *         and CUDA graph-capturable.
 */

#ifndef TRANSFORMER_ENGINE_EP_H_
#define TRANSFORMER_ENGINE_EP_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Config structs ─────────────────────────────────────────────────────── */

/*! \brief Group-level EP configuration (fixed for the EP group lifetime). */
typedef struct {
  int ep_size;                  /*!< EP world size. */
  int num_experts;              /*!< Total experts across all ranks. */
  int max_tokens_per_rank;      /*!< Upper bound on tokens this rank sends per dispatch. */
  int max_recv_tokens_per_rank; /*!< Upper bound on tokens this rank receives per dispatch
                                 *   (size for worst-case top_k fan-out, must be > 0). */
  int hidden_dim;               /*!< Token hidden dimension. */
  int max_num_sms;              /*!< Max SMs for EP kernels. 0 = auto. */
  int allow_handle_mem_reloc;   /*!< 0 (default): throw on relocated handle_mem for a cached
                                 *   handle_id. 1: silently rebuild. Keep 0 in production. */
} NVTEEpGroupConfig;

/*! \brief Per-layer EP configuration. */
typedef struct {
  int num_local_experts; /*!< Reserved for ABI stability (derived from group config). */
  int top_k;             /*!< Per-token expert fan-out. Required. */
  size_t dispatch_output_per_expert_alignment;
  /*!< Per-expert zone alignment in tokens (pow2; 0/1 = no padding). Must match
   *   between nvte_ep_get_handle_mem_size and nvte_ep_prepare. */
} NVTEEpLayerConfig;

/* ── Bootstrap ──────────────────────────────────────────────────────────── */

/*! \brief Bootstrap from an existing NCCL EP sub-communicator. Requires SM_90+.
 *
 *  ep_comm must span exactly group_config.ep_size ranks and is borrowed —
 *  the caller destroys it after nvte_ep_shutdown. Re-init after shutdown is
 *  supported; double-init is an error.
 *
 *  \param[in] ep_comm      Opaque ncclComm_t for the EP sub-group.
 *  \param[in] group_config Group-level EP configuration.
 */
void nvte_ep_initialize(void* ep_comm, NVTEEpGroupConfig group_config);

/*! \brief Tear down the EP backend. Idempotent. Does not destroy ep_comm. */
void nvte_ep_shutdown(void);

/* ── Per-layer size query ────────────────────────────────────────────────── */

/*! \brief Query handle_mem size in bytes for a per-layer config. */
size_t nvte_ep_get_handle_mem_size(NVTEEpLayerConfig layer_config);

/* ── Handle id allocation (host-only, eager) ─────────────────────────────── */

/*! \brief Reserve a fresh handle_id (non-zero) caching the layer config. Host-only. */
uint64_t nvte_ep_allocate_handle_id(NVTEEpLayerConfig layer_config);

/* ── Per-step ops (all allocation-free, CUDA graph-capturable) ──────────── */

/*! \brief AllGather the routing map and write per-expert token counts; stores
 *         routing metadata in handle_mem for subsequent dispatch/combine.
 *
 *  \param[in]     handle_id     Id from nvte_ep_allocate_handle_id.
 *  \param[in]     topk_idx      [T, top_k] int64 sparse routing indices.
 *  \param[out]    token_counts  [num_local_experts] int32 per-local-expert counts.
 *  \param[in,out] handle_mem    Device buffer sized by nvte_ep_get_handle_mem_size.
 *  \param[in]     dispatch_output_per_expert_alignment  Must match the handle_mem sizing.
 *  \param[in]     stream        CUDA stream.
 */
void nvte_ep_prepare(uint64_t handle_id, NVTETensor topk_idx, NVTETensor token_counts,
                     NVTETensor handle_mem, size_t dispatch_output_per_expert_alignment,
                     cudaStream_t stream);

/*! \brief Dispatch tokens (and routing weights) to expert ranks.
 *
 *  \param[in]     handle_id          Id from nvte_ep_allocate_handle_id.
 *  \param[in]     handle_mem         Buffer prepared by nvte_ep_prepare.
 *  \param[in]     topk_idx           [T, top_k] int64 sparse routing indices.
 *  \param[in]     tokens             [T, hidden_dim] input tokens.
 *  \param[in]     topk_weights       [T, top_k] float32 weights, or null in backward.
 *  \param[out]    recv_tokens        [recv_T, hidden_dim] received tokens.
 *  \param[out]    recv_topk_weights  [recv_T] float32 per-slot weights, or null in backward.
 *  \param[in]     stream             CUDA stream.
 */
void nvte_ep_dispatch(uint64_t handle_id, NVTETensor handle_mem, NVTETensor topk_idx,
                      NVTETensor tokens, NVTETensor topk_weights, NVTETensor recv_tokens,
                      NVTETensor recv_topk_weights, cudaStream_t stream);

/*! \brief Scatter-sum expert outputs back to originating ranks.
 *
 *  Combine is UNWEIGHTED: caller must pre-multiply expert_out by
 *  recv_topk_weights (and the valid-slot mask) before calling.
 *
 *  \param[in]  handle_id   Id from nvte_ep_allocate_handle_id.
 *  \param[in]  handle_mem  Buffer prepared by nvte_ep_prepare.
 *  \param[in]  expert_out  [recv_T, hidden_dim] pre-weighted expert outputs.
 *  \param[out] result      [T, hidden_dim] combined output.
 *  \param[in]  stream      CUDA stream.
 */
void nvte_ep_combine(uint64_t handle_id, NVTETensor handle_mem, NVTETensor expert_out,
                     NVTETensor result, cudaStream_t stream);

/*! \brief Backward of dispatch — routes token and weight grads back to the source.
 *
 *  \param[in]  handle_id            Id from nvte_ep_allocate_handle_id.
 *  \param[in]  handle_mem           Buffer prepared by nvte_ep_prepare.
 *  \param[in]  grad                 [recv_capacity, hidden_dim] grad w.r.t. recv_tokens.
 *  \param[in]  g_recv_topk_weights  [recv_capacity] f32 grad w.r.t. recv_topk_weights.
 *  \param[out] grad_tokens          [T, hidden_dim] grad w.r.t. tokens.
 *  \param[out] grad_topk_weights    [T, top_k] f32 grad w.r.t. topk_weights.
 *  \param[in]  stream               CUDA stream.
 */
void nvte_ep_dispatch_bwd(uint64_t handle_id, NVTETensor handle_mem, NVTETensor grad,
                          NVTETensor g_recv_topk_weights, NVTETensor grad_tokens,
                          NVTETensor grad_topk_weights, cudaStream_t stream);

/*! \brief Backward of combine. Padded slots in grad_expert_out are zeroed.
 *
 *  \param[in]  handle_id        Id from nvte_ep_allocate_handle_id.
 *  \param[in]  handle_mem       Buffer prepared by nvte_ep_prepare.
 *  \param[in]  grad             [T, hidden_dim] grad w.r.t. result.
 *  \param[out] grad_expert_out  [recv_capacity, hidden_dim] grad w.r.t. expert_out.
 *  \param[in]  stream           CUDA stream.
 */
void nvte_ep_combine_bwd(uint64_t handle_id, NVTETensor handle_mem, NVTETensor grad,
                         NVTETensor grad_expert_out, cudaStream_t stream);

/* Zero-copy peer-handle annotations live in transformer_engine/comm_handle.h
 * and per-backend headers (e.g. transformer_engine/nccl_comm.h). */

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_H_
