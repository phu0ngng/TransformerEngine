/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_
#define TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_

#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"
#include "transformer_engine/comm_gemm_overlap.h"
#include "../extensions.h"

namespace transformer_engine {
namespace jax {

#ifndef MAX_DEVICES
#define MAX_DEVICES 8
#endif



// Configuration singleton for CGEMM parameters
class CgemmConfig {
 public:
  int num_max_streams;
  int gemm_priority;
  int comm_priority;
  int num_comm_sm;
  bool use_ce;
  bool aggregate_ag;

  static void init(int _num_max_streams, int _gemm_priority, int _comm_priority, int _num_comm_sm,
                   bool _use_ce, bool _aggregate_ag) {
    auto &config = get(false);
    config._initialized = true;
    config.num_max_streams = _num_max_streams;
    config.gemm_priority = _gemm_priority;
    config.comm_priority = _comm_priority;
    config.num_comm_sm = _num_comm_sm;
    config.use_ce = _use_ce;
    config.aggregate_ag = _aggregate_ag;
  }

  static CgemmConfig &get(bool is_initialized = true) {
    static thread_local CgemmConfig instance;
    NVTE_CHECK(instance._initialized == is_initialized,
               "CgemmConfig must be initialized before using it, got is_initialized=", is_initialized);
    return instance;
  }

  CgemmConfig(const CgemmConfig &) = delete;
  CgemmConfig &operator=(const CgemmConfig &) = delete;

 private:
  CgemmConfig() = default;
  ~CgemmConfig() = default;
  bool _initialized = false;
};

// Forward declaration
class CollectiveGemmPlanRegistry;

// NCCL communicator handler for collective GEMM operations
// Support both single process single device AND single process multi device
// Two scenarios:
// 1. Single process multiple devices: TP domain = process (num_devices_per_process == tp_size)
// 2. Single process single device: TP domain spans processes (num_devices_per_process == 1)
class CommunicatorHandler {
 public:
  // Process-level information
  int num_total_devices = -1;  // Total number of devices across all processes
  int num_devices_per_process =
      -1;                  // Number of GPUs per process (1 for single GPU, tp_size for multi GPU)
  int process_id = -1;     // Process ID (0-based)
  int num_processes = -1;  // Total number of processes

  // Tensor Parallel (TP) information - calculated once during init
  int tp_size = -1;                                         // Tensor parallel group size
  int tp_num_nodes = -1;                                    // Number of TP nodes
  int local_device_ids_within_tp_node[MAX_DEVICES] = {-1};  // TP local device ID for each device
  int tp_node_ids[MAX_DEVICES] = {-1};                      // TP node ID for each device

  // Device-level information (arrays for multi-device support)
  int local_device_ids_within_process[MAX_DEVICES];  // CUDA device IDs within this process
  int global_device_ids[MAX_DEVICES];                // Global device ID for each local device
  ncclComm_t comms[MAX_DEVICES];                     // Global NCCL communicator for each local device
  ncclComm_t tp_comms[MAX_DEVICES];                  // TP-domain NCCL communicator for each local device

  // Process-level convenience accessors (NOT TP-domain specific)
  int get_global_rank() const {
    int device_idx = get_local_device_idx_for_current_device();
    return global_device_ids[device_idx];
  }

  // NCCL-based coordination methods for userbuffers
  void nccl_barrier_impl(ExtComm /* not used*/) {
    std::cout << "=== NCCL global barrier called! Process " << process_id << std::endl;
    NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using barrier");

    int device_idx = get_local_device_idx_for_current_device();
    ncclComm_t global_comm = comms[device_idx];  // Use global communicator for barriers

    std::cout << "=== NCCL global barrier executing AllReduce across all processes" << std::endl;
    NVTE_CHECK_NCCL(ncclAllReduce(_barrier, _barrier, 1, ncclInt, ncclSum, global_comm, 0));
    cudaStreamSynchronize(0);
    std::cout << "=== NCCL global barrier completed" << std::endl;
  }

  void nccl_allgather_impl(void *output_buf, size_t output_bytes, void *input_buf,
                           size_t input_bytes, ExtComm /*ExtComm - unused*/) {
    std::cout << "=== NCCL TP allgather called! Process " << process_id << ", input_bytes=" << input_bytes
              << ", output_bytes=" << output_bytes << std::endl;
    NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using allgather");

    int device_idx = get_local_device_idx_for_current_device();
    ncclComm_t tp_comm = tp_comms[device_idx];  // Use TP-domain communicator

    // Ensure input and output sizes are consistent with TP size (not total devices)
    size_t expected_output_bytes = input_bytes * tp_size;
    NVTE_CHECK(output_bytes == expected_output_bytes, "TP allgather buffer size mismatch: expected ",
               expected_output_bytes, ", got ", output_bytes);

    std::cout << "=== NCCL TP allgather executing within TP domain" << std::endl;
    // Use NULL stream to let NCCL handle stream management (CUDA graph friendly)
    NVTE_CHECK_NCCL(
        ncclAllGather(input_buf, output_buf, input_bytes, ncclChar, tp_comm, 0));
    cudaStreamSynchronize(0);
    std::cout << "=== NCCL TP allgather completed" << std::endl;
  }

  // Get communicator for current CUDA device
  ncclComm_t get_comm_for_current_device() const {
    int device_idx = get_local_device_idx_for_current_device();
    return comms[device_idx];
  }

  // Get local device index for current CUDA device
  // Thread-safe: reads immutable data after initialization, cudaGetDevice() is thread-safe
  int get_local_device_idx_for_current_device() const {
    int current_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));

    // Find the local device index that corresponds to the current CUDA device
    // This is thread-safe since local_device_ids is immutable after initialization
    for (int i = 0; i < num_devices_per_process; i++) {
      if (local_device_ids_within_process[i] == current_device) {
        return i;
      }
    }

    NVTE_ERROR("Current CUDA device ", current_device, " not found in local_device_ids");
  }

  // TP-domain-specific accessors for CommOverlapP2P
  // These methods return ranks/nodes within the TP (tensor parallel) domain, not process domain

  // Convenience methods for current device (most common usage)
  int get_local_device_id_within_tp_node() const {
    int device_idx = get_local_device_idx_for_current_device();
    return local_device_ids_within_tp_node[device_idx];
  }

  int get_tp_node_id() const {
    int device_idx = get_local_device_idx_for_current_device();
    return tp_node_ids[device_idx];
  }

  int get_tp_num_nodes() const {
    return tp_num_nodes;
  }

  // Explicit device index methods (for advanced usage)
  int get_local_device_id_within_tp_node(int local_device_idx) const {
    NVTE_CHECK(local_device_idx >= 0 && local_device_idx < num_devices_per_process,
               "Invalid local_device_idx=", local_device_idx, ", must be in [0, ", num_devices_per_process,
               ")");
    return local_device_ids_within_tp_node[local_device_idx];
  }

  int get_tp_node_id(int local_device_idx) const {
    NVTE_CHECK(local_device_idx >= 0 && local_device_idx < num_devices_per_process,
               "Invalid local_device_idx=", local_device_idx, ", must be in [0, ", num_devices_per_process,
               ")");
    return tp_node_ids[local_device_idx];
  }

  static void init(int num_total_devices, int num_devices_per_process, int process_id, int tp_size);

  static CommunicatorHandler &get(bool is_initialized = true) {
    std::cout << "CommunicatorHandler is called with is_initialized=" << is_initialized
              << std::endl;
    static CommunicatorHandler instance;
    NVTE_CHECK(instance._initialize == is_initialized,
               "interface._initialize=", instance._initialize, ", is_initialized=", is_initialized);
    return instance;
  }

  // Cached function objects for userbuffers coordination
  ExtAllgatherOp allgather_func;
  ExtBarrierOp barrier_func;

  CommunicatorHandler(const CommunicatorHandler &) = delete;
  CommunicatorHandler &operator=(const CommunicatorHandler &) = delete;

 private:
  CommunicatorHandler() : _barrier(nullptr) {
    // Initialize arrays to safe defaults
    for (int i = 0; i < MAX_DEVICES; i++) {
      local_device_ids_within_process[i] = -1;
      local_device_ids_within_tp_node[i] = -1;
      tp_node_ids[i] = -1;
      global_device_ids[i] = -1;
      comms[i] = nullptr;
      tp_comms[i] = nullptr;
    }

    // Initialize function objects - these will be set during init()
    allgather_func = [this](void *output_buf, size_t output_bytes, void *input_buf, size_t input_bytes, ExtComm comm) {
      std::cout << "=== Lambda allgather function called!" << std::endl;
      this->nccl_allgather_impl(output_buf, output_bytes, input_buf, input_bytes, comm);
    };
    barrier_func = [this](ExtComm comm) {
      std::cout << "=== Lambda barrier function called!" << std::endl;
      this->nccl_barrier_impl(comm);
    };
  }

  ~CommunicatorHandler() {
    // Clean up NCCL communicators
    if (_initialize) {
      for (int i = 0; i < num_devices_per_process; i++) {
        if (comms[i] != nullptr) {
          ncclCommDestroy(comms[i]);
        }
        if (tp_comms[i] != nullptr) {
          ncclCommDestroy(tp_comms[i]);
        }
      }
    }
    // Clean up device memory
    if (_barrier) cudaFree(_barrier);
  }

  bool _initialize = false;
  // Device memory for barrier operations (single buffer for in-place AllReduce)
  int *_barrier = nullptr;
};

// Plan registry for caching collective GEMM executors
class CollectiveGemmPlanRegistry {
 public:
  static CollectiveGemmPlanRegistry &getInstance() {
    static thread_local CollectiveGemmPlanRegistry instance;
    return instance;
  }

  CommOverlapCore *get_executor(std::vector<size_t> buffer_shape, DType dtype,
                                JAXX_Collective_Op collective_op);

 private:
  CollectiveGemmPlanRegistry() {}
  CollectiveGemmPlanRegistry(const CollectiveGemmPlanRegistry &) = delete;
  CollectiveGemmPlanRegistry &operator=(const CollectiveGemmPlanRegistry &) = delete;

  std::unordered_map<int64_t, std::unique_ptr<CommOverlapCore>> plan_map;
};

// Function declarations
void InitializeCgemmCommunicator(int num_total_devices, int num_devices_per_process, int process_id,
                                 int tp_size, int num_max_streams, int gemm_priority,
                                 int comm_priority, int num_comm_sm, bool use_ce,
                                 bool aggregate_ag);

int GetCgemmNumMaxStreams();

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_
