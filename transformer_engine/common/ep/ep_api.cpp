/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_api.cpp
 *  \brief nvte_ep_* C API: thin delegations to the EPBackend singleton.
 */

#include <nccl.h>
#include <transformer_engine/ep.h>

#include "../common.h"
#include "../util/logging.h"
#include "ep_backend.h"

using transformer_engine::ep::EPBackend;

void nvte_ep_initialize(void* ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  EPBackend::initialize(static_cast<ncclComm_t>(ep_comm), group_config);
}

void nvte_ep_shutdown(void) { EPBackend::shutdown(); }

size_t nvte_ep_get_handle_mem_size(NVTEEpLayerConfig layer_config) {
  return EPBackend::get().get_handle_mem_size(layer_config);
}

uint64_t nvte_ep_allocate_handle_id(NVTEEpLayerConfig layer_config) {
  return EPBackend::get().allocate_handle_id(layer_config);
}

void nvte_ep_prepare(uint64_t handle_id, NVTETensor topk_idx, NVTETensor token_counts,
                     NVTETensor handle_mem, size_t dispatch_output_per_expert_alignment,
                     cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().prepare(handle_id, topk_idx, token_counts, mem_ptr,
                           dispatch_output_per_expert_alignment, stream);
}

void nvte_ep_dispatch(uint64_t handle_id, NVTETensor handle_mem, NVTETensor topk_idx,
                      NVTETensor tokens, NVTETensor topk_weights, NVTETensor recv_tokens,
                      NVTETensor recv_topk_weights, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().dispatch(handle_id, mem_ptr, topk_idx, tokens, topk_weights, recv_tokens,
                            recv_topk_weights, stream);
}

void nvte_ep_combine(uint64_t handle_id, NVTETensor handle_mem, NVTETensor expert_out,
                     NVTETensor result, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().combine(handle_id, mem_ptr, expert_out, result, stream);
}

void nvte_ep_dispatch_bwd(uint64_t handle_id, NVTETensor handle_mem, NVTETensor grad,
                          NVTETensor g_recv_topk_weights, NVTETensor grad_tokens,
                          NVTETensor grad_topk_weights, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().dispatch_bwd(handle_id, mem_ptr, grad, g_recv_topk_weights, grad_tokens,
                                grad_topk_weights, stream);
}

void nvte_ep_combine_bwd(uint64_t handle_id, NVTETensor handle_mem, NVTETensor grad,
                         NVTETensor grad_expert_out, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().combine_bwd(handle_id, mem_ptr, grad, grad_expert_out, stream);
}
