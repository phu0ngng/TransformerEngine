/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ep_manager.h"
#include "nccl_ep_backend.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace ep {

EPManager& EPManager::get() {
  static EPManager instance;
  return instance;
}

void EPManager::initialize(const ncclUniqueId& uid, int world_size, int rank,
                           NVTEEpGroupConfig group_config) {
  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(!initialized_,
             "EPManager::initialize() called more than once. "
             "EP can only be initialized once per process.");

  // Validate config
  NVTE_CHECK(group_config.ep_size > 0,
             "ep_size must be positive, got ", group_config.ep_size);
  NVTE_CHECK(group_config.num_experts > 0,
             "num_experts must be positive, got ", group_config.num_experts);
  NVTE_CHECK(group_config.max_tokens_per_rank > 0,
             "max_tokens_per_rank must be positive, got ",
             group_config.max_tokens_per_rank);
  NVTE_CHECK(group_config.hidden_dim > 0,
             "hidden_dim must be positive, got ", group_config.hidden_dim);
  NVTE_CHECK(group_config.num_experts % group_config.ep_size == 0,
             "num_experts (", group_config.num_experts,
             ") must be divisible by ep_size (",
             group_config.ep_size, ")");

  // Validate bootstrap parameters
  NVTE_CHECK(world_size > 0, "world_size must be positive, got ", world_size);
  NVTE_CHECK(rank >= 0 && rank < world_size,
             "rank must be in [0, world_size), got rank=", rank,
             " world_size=", world_size);
  NVTE_CHECK(world_size >= group_config.ep_size,
             "World size (", world_size, ") must be >= ep_size (",
             group_config.ep_size, ")");

  // SM_90+ runtime guard
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  int major;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major,
                                          cudaDevAttrComputeCapabilityMajor,
                                          device));
  NVTE_CHECK(major >= 9, "NCCL EP requires SM_90+ (Hopper or later), "
             "but current device has compute capability ", major, ".x");

  // Create world communicator from the broadcast unique ID.
  // This is the same path for both PyTorch and JAX — neither framework
  // exposes its internal ncclComm_t, so TE/Common bootstraps its own.
  ncclComm_t world_comm;
  NVTE_CHECK_NCCL(ncclCommInitRank(&world_comm, world_size, uid, rank));

  // Split world_comm into EP sub-communicator
  // color = rank / ep_size groups ranks into EP groups
  int color = rank / group_config.ep_size;
  ncclComm_t ep_comm;
  NVTE_CHECK_NCCL(ncclCommSplit(world_comm, color, rank, &ep_comm, nullptr));

  // Create the NCCLEPBackend and initialize it with the EP sub-comm
  auto backend = std::make_unique<NCCLEPBackend>();
  backend->init(ep_comm, group_config);
  backend_ = std::move(backend);

  // Destroy world_comm — we only needed it for the split.
  // NOTE: zero-copy future work may require retaining world_comm for
  // ncclWindow_t registration.
  NVTE_CHECK_NCCL(ncclCommDestroy(world_comm));

  group_config_ = group_config;
  initialized_ = true;
}

EPBackend& EPManager::backend() {
  NVTE_CHECK(initialized_,
             "EPManager not initialized. Call nvte_ep_initialize() first.");
  return *backend_;
}

}  // namespace ep
}  // namespace transformer_engine
