/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cgemm_helper.h"
#include "nccl.h"

namespace transformer_engine {
namespace jax {

void CommunicatorHandler::init(int num_total_devices, int num_devices_per_process, int process_id, int tp_size) {
  // Validate inputs
  NVTE_CHECK(num_devices_per_process <= MAX_DEVICES,
             "num_devices_per_process exceeds MAX_DEVICES=", MAX_DEVICES,
             ", got num_devices_per_process=", num_devices_per_process);
  NVTE_CHECK(num_devices_per_process >= 1,
             "num_devices_per_process must be >= 1, got num_devices_per_process=",
             num_devices_per_process);
  NVTE_CHECK(num_total_devices >= 1,
             "num_total_devices must be >= 1, got num_total_devices=", num_total_devices);
  NVTE_CHECK(
      num_total_devices % num_devices_per_process == 0,
      "num_total_devices must be divisible by num_devices_per_process, got num_total_devices=",
      num_total_devices, ", num_devices_per_process=", num_devices_per_process);

  // Validate TP size
  NVTE_CHECK(tp_size > 0, "tp_size must be > 0, got tp_size=", tp_size);
  NVTE_CHECK(num_total_devices % tp_size == 0,
             "num_total_devices must be divisible by tp_size, got num_total_devices=",
             num_total_devices, ", tp_size=", tp_size);

  std::cout << "=== Calling from init with num_total_devices=" << num_total_devices
            << ", num_devices_per_process=" << num_devices_per_process
            << ", process_id=" << process_id << ", tp_size=" << tp_size << std::endl;

  auto &handler = get(false);
  handler.num_total_devices = num_total_devices;
  handler.num_devices_per_process = num_devices_per_process;
  handler.process_id = process_id;
  handler.num_processes = num_total_devices / num_devices_per_process;
  handler.tp_size = tp_size;
  handler.tp_num_nodes = num_total_devices / tp_size;

  NVTE_CHECK(0 <= process_id && process_id < handler.num_processes,
             "Invalid process_id=", process_id, ", which is out of range [0, ",
             handler.num_processes, ")");

  // Initialize local devices and their global ranks
  for (int local_idx = 0; local_idx < num_devices_per_process; local_idx++) {
    // Use the device that JAX has already assigned to this process
    int current_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
    handler.local_device_ids_within_process[local_idx] = current_device;
    handler.global_device_ids[local_idx] = process_id * num_devices_per_process + local_idx;

    // Calculate TP-related values for this device
    int global_device_id = handler.global_device_ids[local_idx];
    if (num_devices_per_process == tp_size) {
      // Scenario 1: Multi-device per process - TP domain = single process
      handler.local_device_ids_within_tp_node[local_idx] = local_idx;
      handler.tp_node_ids[local_idx] = process_id;
    } else {
      // Scenario 2: Single device per process - TP domain spans multiple processes
      handler.local_device_ids_within_tp_node[local_idx] = global_device_id % tp_size;
      handler.tp_node_ids[local_idx] = global_device_id / tp_size;
    }

    std::cout << "=== Process " << process_id << ", local_idx=" << local_idx
              << " -> Using JAX-assigned CUDA device=" << current_device
              << ", global_device_id=" << global_device_id
              << ", tp_local_device_id=" << handler.local_device_ids_within_tp_node[local_idx]
              << ", tp_node_id=" << handler.tp_node_ids[local_idx] << std::endl;

    // Device is already set by JAX, no need to change it
  }

  // Create NCCL communicators for all local devices
  ncclUniqueId id;

  // Process 0 generates the unique ID, then broadcast via file system
  // Use process group ID to ensure different job runs don't interfere
  pid_t pgid = getpgid(0);
  std::string id_file = "/tmp/nccl_unique_id_pgid_" + std::to_string(pgid) + "_" +
                        std::to_string(num_total_devices) + "_" + std::to_string(tp_size) + ".bin";

  if (process_id == 0) {
    NVTE_CHECK_NCCL(ncclGetUniqueId(&id));
    std::cout << "=== Process 0 generated NCCL unique ID" << std::endl;

    // Write the ID to a temporary file
    std::ofstream file(id_file, std::ios::binary);
    NVTE_CHECK(file.is_open(), "Failed to create NCCL unique ID file: ", id_file);
    file.write(reinterpret_cast<const char*>(&id), sizeof(ncclUniqueId));
    file.close();
    std::cout << "=== Process 0 wrote NCCL unique ID to file: " << id_file << std::endl;
  } else {
    // Wait for the file to be created and read it
    std::cout << "=== Process " << process_id << " waiting for NCCL unique ID file: " << id_file << std::endl;
    int attempts = 0;
    const int max_attempts = 100; // 10 seconds with 100ms sleep
    while (attempts < max_attempts) {
      std::ifstream file(id_file, std::ios::binary);
      if (file.is_open()) {
        file.read(reinterpret_cast<char*>(&id), sizeof(ncclUniqueId));
        if (file.gcount() == sizeof(ncclUniqueId)) {
          file.close();
          std::cout << "=== Process " << process_id << " successfully read NCCL unique ID" << std::endl;
          break;
        }
        file.close();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      attempts++;
    }
    NVTE_CHECK(attempts < max_attempts, 
               "Timeout waiting for NCCL unique ID file from process 0: ", id_file);
  }

  // Initialize communicators using NCCL group API for efficiency
  std::cout << "=== Starting NCCL group initialization for " << num_devices_per_process
            << " devices" << std::endl;
  NVTE_CHECK_NCCL(ncclGroupStart());
  for (int local_idx = 0; local_idx < num_devices_per_process; local_idx++) {
    NVTE_CHECK_CUDA(cudaSetDevice(handler.local_device_ids_within_process[local_idx]));
    std::cout << "=== Initializing NCCL comm for local_idx=" << local_idx
              << ", global_device_id=" << handler.global_device_ids[local_idx]
              << ", device_id=" << handler.local_device_ids_within_process[local_idx]
              << std::endl;
    NVTE_CHECK_NCCL(ncclCommInitRank(&handler.comms[local_idx], handler.num_total_devices, id,
                                     handler.global_device_ids[local_idx]));
  }
  std::cout << "=== Ending NCCL group initialization" << std::endl;
  NVTE_CHECK_NCCL(ncclGroupEnd());

  std::cout << "=== Successfully initialized " << num_devices_per_process << " NCCL communicators"
            << std::endl;

  // Clean up the temporary file (only process 0 needs to do this)
  if (process_id == 0) {
    std::remove(id_file.c_str());
    std::cout << "=== Process 0 cleaned up NCCL unique ID file: " << id_file << std::endl;
  }

  // Allocate device memory for barrier operations
  NVTE_CHECK_CUDA(cudaMalloc(&handler._barrier, sizeof(int)));
  std::cout << "=== Allocated device memory for NCCL barrier operations" << std::endl;

  // Mark as initialized
  handler._initialize = true;
}

void InitializeCgemmCommunicator(int num_total_devices, int num_devices_per_process, int process_id,
                                 int tp_size, int num_max_streams, int gemm_priority,
                                 int comm_priority, int num_comm_sm, bool use_ce,
                                 bool aggregate_ag) {
  auto &config = CgemmConfig::get(false);
  config.init(num_max_streams, gemm_priority, comm_priority, num_comm_sm, use_ce, aggregate_ag);
  auto &handler = CommunicatorHandler::get(false);
  handler.init(num_total_devices, num_devices_per_process, process_id, tp_size);
}

// Accessor function to get cached num_max_streams for Python
int GetCgemmNumMaxStreams() {
  auto &config = CgemmConfig::get();
  return config.num_max_streams;
}

// Implementation of CollectiveGemmPlanRegistry::get_executor
CommOverlapCore *CollectiveGemmPlanRegistry::get_executor(std::vector<size_t> buffer_shape, DType dtype,
                                                          JAXX_Collective_Op collective_op) {
  auto &comm_handler = CommunicatorHandler::get();
  auto &cgemm_config = CgemmConfig::get();

  // Get device index from current CUDA context (JAX sets this via FFI)
  int device_idx = comm_handler.get_local_device_idx_for_current_device();

  // Include device_idx in plan cache key to ensure device-specific caching
  int64_t plan_id = 0;
  hash_combine(plan_id, buffer_shape[0], buffer_shape[1], static_cast<size_t>(dtype),
               static_cast<int>(collective_op), comm_handler.tp_size,
               cgemm_config.num_max_streams, cgemm_config.gemm_priority,
               cgemm_config.comm_priority, cgemm_config.num_comm_sm, cgemm_config.use_ce,
               cgemm_config.aggregate_ag, device_idx);

  // Check if plan already exists
  auto it = plan_map.find(plan_id);
  if (it != plan_map.end()) {
    return it->second.get();  // Return existing executor
  }
  
  std::cout << "=== CollectiveGemmPlanRegistry calls handler init" << std::endl;
  
  // Validate TP configuration and determine scenario
  if (comm_handler.num_devices_per_process == comm_handler.tp_size) {
    // Scenario 1: Multi-device per process - TP domain = single process
    std::cout << "=== TP Scenario 1: Multi-device per process (TP domain = single process)"
              << std::endl;
  } else if (comm_handler.num_devices_per_process == 1) {
    // Scenario 2: Single device per process - TP domain spans multiple processes
    NVTE_CHECK(comm_handler.num_total_devices % comm_handler.tp_size == 0,
               "For single device per process, num_total_devices must be divisible by tp_size, "
               "got num_total_devices=",
               comm_handler.num_total_devices, ", tp_size=", comm_handler.tp_size);
    std::cout << "=== TP Scenario 2: Single device per process (TP domain spans processes)"
              << std::endl;
  } else {
    NVTE_ERROR("Unsupported TP configuration: num_devices_per_process=",
               comm_handler.num_devices_per_process, ", tp_size=", comm_handler.tp_size,
               ". Supported scenarios: "
               "(1) num_devices_per_process == tp_size (multi-device per process), "
               "(2) num_devices_per_process == 1 (single device per process)");
  }
  
  printf(
      "Global rank %d, num_total_devices %d, tp_local_rank %d, tp_size %d, tp_node_id %d, "
      "tp_num_nodes %d",
      comm_handler.get_global_rank(), comm_handler.num_total_devices,
      comm_handler.get_local_device_id_within_tp_node(), comm_handler.tp_size,
      comm_handler.get_tp_node_id(), comm_handler.tp_num_nodes);

  // Create executor with device-specific parameters (device_idx determined above)
  std::unique_ptr<CommOverlapCore> executor;
  executor = std::make_unique<CommOverlapP2PBase>(
      buffer_shape, dtype, comm_handler.get_global_rank(), comm_handler.num_total_devices,
      comm_handler.get_local_device_id_within_tp_node(), comm_handler.tp_size,
      comm_handler.get_tp_node_id(), comm_handler.tp_num_nodes, comm_handler.tp_size,
      comm_handler.allgather_func, comm_handler.barrier_func, get_nvte_collective_op(collective_op),
      cgemm_config.num_max_streams, 1 /*comm_cga_size*/, cgemm_config.gemm_priority,
      cgemm_config.comm_priority, cgemm_config.num_comm_sm, true /*set_sm_margin*/,
      cgemm_config.use_ce, false /*atomic_gemm*/, cgemm_config.aggregate_ag);

  CommOverlapCore *executor_ptr = executor.get();
  plan_map[plan_id] = std::move(executor);
  return executor_ptr;
}

}  // namespace jax
}  // namespace transformer_engine
