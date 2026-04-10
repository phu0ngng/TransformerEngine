/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_api.cpp
 *  \brief Implementation of the public nvte_ep_* C API.
 *
 *  These functions are the stable public surface. They delegate to
 *  EPManager::get().backend() which owns the NCCLEPBackend instance.
 *  Nothing else in TE touches nccl_ep.h directly.
 */

#include <cstring>  // memcpy

#include <nccl.h>  // ncclUniqueId
#include <transformer_engine/ep.h>
#include "ep_manager.h"
#include "../util/logging.h"

void nvte_ep_initialize(const uint8_t* unique_id_bytes,
                        int world_size,
                        int rank,
                        NVTEEpGroupConfig group_config) {
  NVTE_CHECK(unique_id_bytes != nullptr, "unique_id_bytes must not be null");

  ncclUniqueId uid;
  memcpy(&uid, unique_id_bytes, sizeof(uid));

  transformer_engine::ep::EPManager::get().initialize(
      uid, world_size, rank, group_config);
}

size_t nvte_ep_get_handle_mem_size(NVTEEpLayerConfig layer_config) {
  return transformer_engine::ep::EPManager::get()
      .backend()
      .get_handle_mem_size(layer_config);
}

void nvte_ep_prepare(NVTETensor topk_idx,
                     NVTEEpLayerConfig layer_config,
                     NVTETensor token_counts,
                     NVTETensor handle_mem,
                     cudaStream_t stream) {
  // handle_mem data pointer is the actual buffer the backend writes into
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");

  transformer_engine::ep::EPManager::get()
      .backend()
      .prepare(mem_ptr, topk_idx, token_counts,
               layer_config, stream);
}

void nvte_ep_dispatch(NVTETensor handle_mem,
                      NVTETensor tokens,
                      NVTETensor topk_weights,
                      NVTETensor recv_tokens,
                      cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");

  transformer_engine::ep::EPManager::get()
      .backend()
      .dispatch(mem_ptr, tokens, topk_weights, recv_tokens, stream);
}

void nvte_ep_combine(NVTETensor handle_mem,
                     NVTETensor expert_out,
                     NVTETensor result,
                     cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");

  transformer_engine::ep::EPManager::get()
      .backend()
      .combine(mem_ptr, expert_out, result, stream);
}

void nvte_ep_dispatch_bwd(NVTETensor handle_mem,
                          NVTETensor grad,
                          NVTETensor result,
                          cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");

  transformer_engine::ep::EPManager::get()
      .backend()
      .dispatch_bwd(mem_ptr, grad, result, stream);
}

void nvte_ep_combine_bwd(NVTETensor handle_mem,
                         NVTETensor grad,
                         NVTETensor result,
                         cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle_mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");

  transformer_engine::ep::EPManager::get()
      .backend()
      .combine_bwd(mem_ptr, grad, result, stream);
}
