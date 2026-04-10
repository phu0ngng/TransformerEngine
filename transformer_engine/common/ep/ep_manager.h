/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_manager.h
 *  \brief Internal singleton managing the EP backend lifecycle.
 *
 *  NOT exposed to Python. Framework extensions call nvte_* which calls
 *  EPManager::get().
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_MANAGER_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_MANAGER_H_

#include <memory>
#include <mutex>

#include <nccl.h>  // ncclUniqueId, ncclComm_t (needed for ncclCommInitRank/Split)
#include "ep_backend.h"

namespace transformer_engine {
namespace ep {

/*! \brief Static singleton — one per process.
 *
 *  Created on first call to initialize(). Owns the EPBackend and the
 *  EP sub-communicator for the program lifetime. Destroyed at exit.
 */
class EPManager {
 public:
  /*! \brief Access the singleton instance. */
  static EPManager& get();

  /*! \brief Called once at program start via nvte_ep_initialize().
   *
   *  Creates a world-sized NCCL comm from the broadcast unique_id via
   *  ncclCommInitRank, splits it into the EP sub-comm (ncclCommSplit
   *  with color = rank / ep_size), creates ncclEpGroup_t, caches the
   *  EP group for the program lifetime, then destroys the world comm.
   *
   *  Neither the world comm nor the EP sub-comm is ever returned to
   *  the caller — TE/Common owns them for the process lifetime.
   *
   *  \param[in] uid         ncclUniqueId (broadcast by caller, identical on all ranks).
   *  \param[in] world_size  Total number of ranks.
   *  \param[in] rank        This process's rank.
   *  \param[in] config      Group-level EP configuration.
   */
  void initialize(const ncclUniqueId& uid, int world_size, int rank,
                  NVTEEpGroupConfig group_config);

  /*! \brief Access the active backend.
   *
   *  Asserts that initialize() has been called.
   */
  EPBackend& backend();

  /*! \brief Check whether the manager has been initialized. */
  bool is_initialized() const { return initialized_; }

  /*! \brief Access the group config (valid after initialization). */
  const NVTEEpGroupConfig& group_config() const { return group_config_; }

 private:
  EPManager() = default;
  ~EPManager() = default;

  // Non-copyable, non-movable
  EPManager(const EPManager&) = delete;
  EPManager& operator=(const EPManager&) = delete;

  std::unique_ptr<EPBackend> backend_;
  NVTEEpGroupConfig group_config_{};
  bool initialized_{false};
  std::mutex mutex_;
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_MANAGER_H_
