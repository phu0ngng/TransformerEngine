/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"
#include "transformer_engine/comm_gemm_overlap.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>

#include "../extensions.h"
#include "common.h"
#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"
#include "common/util/cuda_runtime.h"
#include "common/util/string.h"
#include "common/util/system.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "transformer_engine/swizzle.h"
#include "xla/ffi/api/c_api.h"

#define MXFP8_BLOCK_SIZE 32

namespace transformer_engine {
namespace jax {

static uint8_t *move_ptr_to_next_256B_aligned(uint8_t *ptr) {
  // Move the pointer to the next 256B aligned address
  return reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(ptr) + 255) &
                                     ~static_cast<uintptr_t>(255));
}

std::tuple<TensorWrapper, std::vector<size_t>> xla_buffer_to_nvte_gemm_operand(
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, JAXX_Scaling_Mode scaling_mode,
    size_t axis_boundary, bool rowwise) {
  // Set tensor data with collapsed 2D shape
  auto buffer_dims = buffer.dimensions();
  std::vector<size_t> input_shape = {product(buffer_dims, 0, axis_boundary),
                                     product(buffer_dims, axis_boundary, buffer_dims.size())};
  auto input_dtype = convert_ffi_datatype_to_te_dtype(buffer.element_type());
  TensorWrapper input(get_nvte_scaling_mode(scaling_mode));

  if (rowwise) {
    input.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
  } else {
    input.set_columnwise_data(buffer.untyped_data(), input_dtype, input_shape);
  }

  // Set scaling factor for quantized tensors
  if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
    NVTE_CHECK(typeToSize(input_dtype) == 1, "Quantized GEMM requires 8-bit operands.");
    NVTE_CHECK(scale_inv.element_count() > 0, "Missing inverse scaling factor for quantized GEMM.");

    std::vector<size_t> scale_shape = {1};
    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Block scaling also needs to be collapsed to match 2D data
      scale_shape = {product(scale_inv.dimensions(), 0, axis_boundary),
                     product(scale_inv.dimensions(), axis_boundary, scale_inv.dimensions().size())};
    }

    auto scale_dtype = convert_ffi_datatype_to_te_dtype(scale_inv.element_type());
    if (rowwise) {
      input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    } else {
      input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    }
  }

  return std::make_tuple(std::move(input), input_shape);
}

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
    NVTE_CHECK(instance._initialized == is_initialized, "CgemmConfig must be initialized before using it, got is_initialized=", is_initialized);
    return instance;
  }

  CgemmConfig(const CgemmConfig &) = delete;
  CgemmConfig &operator=(const CgemmConfig &) = delete;

 private:
  CgemmConfig() = default;
  ~CgemmConfig() = default;
  bool _initialized = false;
};

#ifndef MAX_DEVICES
#define MAX_DEVICES 8
#endif

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
  ncclComm_t comms[MAX_DEVICES];                     // NCCL communicator for each local device

  // Process-level convenience accessors (NOT TP-domain specific)
  int get_global_device_id() const {
    int device_idx = get_local_device_idx_for_current_device();
    return global_device_ids[device_idx];
  }

  // NCCL-based coordination methods for userbuffers
  void nccl_barrier_impl(ExtComm /* not used*/) {
    NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using barrier");

    int device_idx = get_local_device_idx_for_current_device();
    ncclComm_t nccl_comm = comms[device_idx];

    NVTE_CHECK_NCCL(ncclAllReduce(_barrier, _barrier, 1, ncclInt, ncclSum, nccl_comm, nullptr));
  }

  void nccl_allgather_impl(void *output_buf, size_t output_bytes, void *input_buf,
                           size_t input_bytes, ExtComm /*ExtComm - unused*/) {
    NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using allgather");

    int device_idx = get_local_device_idx_for_current_device();
    ncclComm_t nccl_comm = comms[device_idx];

    // Ensure input and output sizes are consistent with the number of ranks
    size_t expected_output_bytes = input_bytes * num_total_devices;
    NVTE_CHECK(output_bytes == expected_output_bytes, "Output buffer size mismatch: expected ",
               expected_output_bytes, ", got ", output_bytes);

    // Use NULL stream to let NCCL handle stream management
    NVTE_CHECK_NCCL(
        ncclAllGather(input_buf, output_buf, input_bytes, ncclChar, nccl_comm, nullptr));
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

  // Explicit device index methods (for advanced usage)

  static void init(int num_total_devices, int num_devices_per_process, int process_id,
                   int tp_size) {
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
    std::string id_file = "/tmp/nccl_unique_id_" + std::to_string(num_total_devices) + "_" + std::to_string(tp_size) + ".bin";
    
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

    // Bootstrap UB via creating a dummy CommOverlapP2PBase object
    {
      std::vector<size_t> buffer_shape{0, 0};
      DType dtype = DType::kByte;
      auto &cgemm_config = CgemmConfig::get();
      CommOverlapP2PBase bootstrap_obj(
          buffer_shape, dtype, handler.global_device_ids[0], handler.num_total_devices,
          handler.get_local_device_id_within_tp_node(), handler.tp_size, handler.get_tp_node_id(),
          handler.tp_num_nodes, handler.tp_size,
          handler.allgather_func, handler.barrier_func, get_nvte_collective_op(JAXX_Collective_Op::ALL_GATHER),
          cgemm_config.num_max_streams, 1 /*comm_cga_size*/, cgemm_config.gemm_priority,
          cgemm_config.comm_priority, cgemm_config.num_comm_sm, true /*set_sm_margin*/,
          cgemm_config.use_ce, false /*atomic_gemm*/, cgemm_config.aggregate_ag);
    }

    handler._initialize = true;
  }

  static CommunicatorHandler &get(bool is_initialized = true) {
    std::cout << "CommunicatorHandler is called with is_initialized=" << is_initialized
              << std::endl;
    static CommunicatorHandler instance;
    NVTE_CHECK(instance._initialize == is_initialized,
               "interface._initialize=", instance._initialize, ", is_initialized=", is_initialized);
    return instance;
  }

  CommunicatorHandler(const CommunicatorHandler &) = delete;
  CommunicatorHandler &operator=(const CommunicatorHandler &) = delete;

  // Cached function objects for userbuffers coordination
  ExtAllgatherOp allgather_func;
  ExtBarrierOp barrier_func;

 private:
  CommunicatorHandler() : _barrier(nullptr) {
    // Initialize arrays to safe defaults
    for (int i = 0; i < MAX_DEVICES; i++) {
      local_device_ids_within_process[i] = -1;
      local_device_ids_within_tp_node[i] = -1;
      tp_node_ids[i] = -1;
      global_device_ids[i] = -1;
      comms[i] = nullptr;
    }

    // Initialize function objects - these will be set during init()
    allgather_func = [this](void *output_buf, size_t output_bytes, void *input_buf, size_t input_bytes, ExtComm comm) {
      this->nccl_allgather_impl(output_buf, output_bytes, input_buf, input_bytes, comm);
    };
    barrier_func = [this](ExtComm comm) {
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
      }
    }
    // Clean up device memory
    if (_barrier) cudaFree(_barrier);
  }

  bool _initialize = false;
  // Device memory for barrier operations (single buffer for in-place AllReduce)
  int *_barrier = nullptr;
};

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

class CollectiveGemmPlanRegistry {
 public:
  static CollectiveGemmPlanRegistry &getInstance() {
    static thread_local CollectiveGemmPlanRegistry instance;
    return instance;
  }

  CommOverlapCore *get_executor(std::vector<size_t> buffer_shape, DType dtype,
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
    std::cout << "=== CollectiveGemmPlanRegistry calls hanlder init" << std::endl;
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
        "tp_num_nodes "
        "%d",
        comm_handler.get_global_device_id(), comm_handler.num_total_devices,
        comm_handler.get_local_device_id_within_tp_node(), comm_handler.tp_size,
        comm_handler.get_tp_node_id(), comm_handler.tp_num_nodes);


    // Create executor with device-specific parameters (device_idx determined above)
    std::unique_ptr<CommOverlapCore> executor;
    executor = std::make_unique<CommOverlapP2PBase>(
        buffer_shape, dtype,
        comm_handler.get_global_device_id(), comm_handler.num_total_devices,
        comm_handler.get_local_device_id_within_tp_node(), comm_handler.tp_size,
        comm_handler.get_tp_node_id(), comm_handler.tp_num_nodes,
        comm_handler.tp_size,
        comm_handler.allgather_func, comm_handler.barrier_func,
        get_nvte_collective_op(collective_op),
        cgemm_config.num_max_streams, 1 /*comm_cga_size*/, cgemm_config.gemm_priority,
        cgemm_config.comm_priority, cgemm_config.num_comm_sm, true /*set_sm_margin*/,
        cgemm_config.use_ce, false /*atomic_gemm*/, cgemm_config.aggregate_ag);

    CommOverlapCore *executor_ptr = executor.get();
    plan_map[plan_id] = std::move(executor);
    return executor_ptr;
  }

 private:
  CollectiveGemmPlanRegistry() {}
  CollectiveGemmPlanRegistry(const CollectiveGemmPlanRegistry &) = delete;
  CollectiveGemmPlanRegistry &operator=(const CollectiveGemmPlanRegistry &) = delete;

  std::unordered_map<int64_t, std::unique_ptr<CommOverlapCore>> plan_map;
};

Error_Type CollectiveGemmInitFFI(Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                                 Buffer_Type rhs_scale_inv, Buffer_Type bias,
                                 Buffer_Type gelu_input, Result_Type output, Result_Type bias_grad,
                                 Result_Type pre_gelu_out, Result_Type workspace,
                                 JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                                 int64_t rhs_axis_boundary, bool lhs_transposed,
                                 bool rhs_transposed, bool fuse_bias, bool fuse_gelu, bool grad,
                                 bool use_split_accumulator, JAXX_Collective_Op collective_op) {
  nvte_cublas_handle_init();

  auto &comm_handler = CommunicatorHandler::get();

  // Init UB buffer
  if (collective_op != JAXX_Collective_Op::NONE) {
    std::vector<size_t> lhs_shape = {
        product(lhs.dimensions(), 0, lhs_axis_boundary),
        product(lhs.dimensions(), lhs_axis_boundary, lhs.dimensions().size())};
    std::vector<size_t> rhs_shape = {
        product(rhs.dimensions(), 0, rhs_axis_boundary),
        product(rhs.dimensions(), rhs_axis_boundary, rhs.dimensions().size())};

    std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                     (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};

    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
    if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
    }
    auto _ = CollectiveGemmPlanRegistry::getInstance().get_executor(buffer_shape, buffer_dtype,
                                                                    collective_op);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveGemmInitHandler, CollectiveGemmInitFFI,
                              FFI::Bind<FFI_Prepare>()
                                  .Arg<Buffer_Type>()  // lhs
                                  .Arg<Buffer_Type>()  // lhs_scale_inv
                                  .Arg<Buffer_Type>()  // rhs
                                  .Arg<Buffer_Type>()  // rhs_scale_inv
                                  .Arg<Buffer_Type>()  // bias
                                  .Arg<Buffer_Type>()  // gelu_input
                                  .Ret<Buffer_Type>()  // output
                                  .Ret<Buffer_Type>()  // bias_grad
                                  .Ret<Buffer_Type>()  // pre_gelu_out
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<JAXX_Collective_Op>("collective_op"));

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Result_Type output, Result_Type bias_grad, Result_Type pre_gelu_out,
                   Result_Type workspace, JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                   int64_t rhs_axis_boundary, bool lhs_transposed, bool rhs_transposed,
                   bool fuse_bias, bool fuse_gelu, bool grad, bool use_split_accumulator,
                   JAXX_Collective_Op collective_op) {
  std::cout << "=== GemmFFI is called" << std::endl;
  // NOTE: TensorWrapper operands are always rowwise for full-precision GEMM, or FP8 GEMM when
  //       device supports non-TN layouts (compute capability >= 10.0, excluding 12.x)
  bool always_rowwise = (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
                         (is_tensor_scaling(scaling_mode) && nvte_is_non_tn_fp8_gemm_supported()));
  bool make_lhs_rowwise = (always_rowwise) ? true : !lhs_transposed;
  bool make_rhs_rowwise = (always_rowwise) ? true : rhs_transposed;
  auto [lhs_, lhs_shape] = xla_buffer_to_nvte_gemm_operand(stream, lhs, lhs_scale_inv, scaling_mode,
                                                           lhs_axis_boundary, make_lhs_rowwise);
  auto [rhs_, rhs_shape] = xla_buffer_to_nvte_gemm_operand(stream, rhs, rhs_scale_inv, scaling_mode,
                                                           rhs_axis_boundary, make_rhs_rowwise);

  std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                   (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());

  // Bias input to forward pass or bias gradient output from backward pass
  void *bias_ptr = nullptr;
  std::vector<size_t> bias_shape = {0};
  DType bias_dtype = out_dtype;
  if (fuse_bias) {
    if (!grad) {
      NVTE_CHECK(bias_grad->untyped_data() == bias.untyped_data(),
                 "Missing operand-output aliasing in GemmPrimitive: bias <-> bias_grad");
    }
    bias_ptr = bias_grad->untyped_data();
    bias_shape.at(0) = bias_grad->dimensions().front();
    bias_dtype = convert_ffi_datatype_to_te_dtype(bias_grad->element_type());
  }
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);

  // Pre-GeLU output from forward pass or input to backward pass
  void *pre_gelu_ptr = nullptr;
  std::vector<size_t> pre_gelu_shape = {0};
  DType pre_gelu_dtype = out_dtype;
  if (gelu_input.element_count() > 0) {
    if (grad) {
      NVTE_CHECK(pre_gelu_out->untyped_data() == gelu_input.untyped_data(),
                 "Missing operand-output aliasing in GemmPrimitive: gelu_input <-> pre_gelu_out");
    }
    pre_gelu_ptr = pre_gelu_out->untyped_data();
    pre_gelu_shape = {product(pre_gelu_out->dimensions(), 0, pre_gelu_out->dimensions().size() - 1),
                      static_cast<size_t>(pre_gelu_out->dimensions().back())};
    pre_gelu_dtype = convert_ffi_datatype_to_te_dtype(pre_gelu_out->element_type());
  }
  auto pre_gelu_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, pre_gelu_dtype);

  // cuBLAS workspace + 256 alignment enforcement
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  workspace_ptr = move_ptr_to_next_256B_aligned(workspace_ptr);
  std::vector<size_t> workspace_shape = {static_cast<size_t>(workspace->element_count()) - 256};
  auto workspace_ = TensorWrapper(workspace_ptr, workspace_shape, DType::kByte);

  // Launch TE/common kernel with swapped LHS/RHS for cuBLAS column-major order
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  if (collective_op == JAXX_Collective_Op::NONE) {
    auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
    NVTE_CHECK(out_.numel() == output->element_count(),
               "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(), " elements ",
               to_string_like(out_shape), " but got ", output->element_count(), " elements ",
               to_string_like(output->dimensions()));

    nvte_cublas_gemm(rhs_.data(), lhs_.data(), out_.data(), bias_.data(), pre_gelu_.data(),
                     rhs_transposed, lhs_transposed, grad, workspace_.data(), false,
                     use_split_accumulator, num_math_sm, stream);
  } else {
    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = out_dtype;
    auto &comm_handler = CommunicatorHandler::get();
    if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      out_shape[0] = out_shape[0] * comm_handler.tp_size;
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
      out_shape[0] = out_shape[0] / comm_handler.tp_size;
    }
    auto executor = CollectiveGemmPlanRegistry::getInstance().get_executor(
        buffer_shape, buffer_dtype, collective_op);
    if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      auto ubuf_out_ = TensorWrapper(executor->get_ubuf_dptr(), buffer_shape, out_dtype);
      // Prepare the auxiliary buffer for the reduce-scattered GEMM output
      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));

      // Launch GEMM+RS
      executor->split_overlap_rs(rhs_, rhs_transposed, lhs_, lhs_transposed, ubuf_out_, bias_,
                                 pre_gelu_, workspace_, grad, false, use_split_accumulator, out_,
                                 stream);

      // TODO: Don't we need to copy the output back to the original buffer?
    } else if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      auto aux_out_ = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);  // Empty

      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));
      // Copy the distributed LHS operand into the local chunk of the communication buffer
      executor->copy_into_buffer(stream, lhs_, true, make_lhs_rowwise);

      // Launch AG+GEMM
      executor->split_overlap_ag(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                                 workspace_, grad, false, use_split_accumulator, aux_out_, stream);
    }
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // gelu_input
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<JAXX_Collective_Op>("collective_op"),
                              FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Buffer_Type lhs_data, Buffer_Type lhs_sinv,
                          Buffer_Type rhs_data, Buffer_Type rhs_sinv, Buffer_Type bias,
                          Buffer_Type group_sizes, Buffer_Type group_offset, Result_Type output,
                          Result_Type workspace, size_t m, size_t n, size_t k, bool lhs_is_trans,
                          bool rhs_is_trans, JAXX_Scaling_Mode scaling_mode, bool has_bias,
                          bool is_grouped_dense_wgrad) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major [m, k] for N - [k, m] for T
  //   B: row-major [k, n] for N - [n, k] for T
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m] for T - [m, k] for N
  //   B: column-major with size [n, k] for T - [k, n] for N
  //
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  int num_streams = nvte_get_num_compute_streams();

  // Inputs
  auto lhs_ptr = reinterpret_cast<uint8_t *>(lhs_data.untyped_data());
  auto rhs_ptr = reinterpret_cast<uint8_t *>(rhs_data.untyped_data());
  auto lhs_sinv_ptr = reinterpret_cast<uint8_t *>(lhs_sinv.untyped_data());
  auto rhs_sinv_ptr = reinterpret_cast<uint8_t *>(rhs_sinv.untyped_data());
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_data.element_type());
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_data.element_type());
  auto lhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(lhs_sinv.element_type());
  auto rhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(rhs_sinv.element_type());
  auto bias_ptr = has_bias ? reinterpret_cast<uint8_t *>(bias.untyped_data()) : nullptr;
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());

  NVTE_CHECK(group_sizes.dimensions().size() == 1);
  size_t num_gemms = group_sizes.dimensions()[0];

  // It is weird that TE/Common GEMM only use colwise for MXFP8
  const bool is_fp8_gemm = is_fp8_dtype(lhs_dtype);
  const bool is_tensor_scaling = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
                                 scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING;
  const bool is_mxfp8_scaling = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
  const bool rhs_use_colwise = is_mxfp8_scaling && !rhs_is_trans;
  const bool lhs_use_colwise = is_mxfp8_scaling && lhs_is_trans;

  // Outputs
  auto out_ptr = reinterpret_cast<uint8_t *>(output->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  // Here we clear the lower 8 bits of the buffer address to ensure the buffer is 256-aligned
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  workspace_ptr = move_ptr_to_next_256B_aligned(workspace_ptr);
  auto workspace_total_size = product(workspace->dimensions());

  auto lhs_sinv_size = product(lhs_sinv.dimensions());
  auto rhs_sinv_size = product(rhs_sinv.dimensions());
  const size_t workspace_alignment_padding = 256;
  const size_t tensor_scaling_sinv_aligment = 16;
  const size_t mxfp8_scaling_sinv_alignment_padding = 256;
  auto workspace_size = workspace_total_size - workspace_alignment_padding;
  if (is_mxfp8_scaling) {
    // For MXFP8 swizzled scale_inv buffers, only the first pointer needs to be with 256B alignment padding. Later pointers are guaranteed to be 256-aligned as the scale_inv shapes are padded by 128x4.
    workspace_size -= (lhs_sinv_size + rhs_sinv_size + 2 * mxfp8_scaling_sinv_alignment_padding);
  } else if (is_tensor_scaling) {
    // For tensor scaling, each matrix has a single scale value, and all scales need to be aligned
    // by 16 bytes to meet the requirement of CUDA 12.9.1 and later.
    workspace_size -= tensor_scaling_sinv_aligment * (lhs_sinv_size + rhs_sinv_size);
  }
  workspace_size = workspace_size / num_streams;
  auto swizzled_lhs_sinv_ptr = workspace_ptr + workspace_size * num_streams;
  swizzled_lhs_sinv_ptr = move_ptr_to_next_256B_aligned(swizzled_lhs_sinv_ptr);
  auto swizzled_rhs_sinv_ptr = swizzled_lhs_sinv_ptr + lhs_sinv_size;
  swizzled_rhs_sinv_ptr = move_ptr_to_next_256B_aligned(swizzled_rhs_sinv_ptr);
  auto lhs_scatter_aligned_ptr = swizzled_lhs_sinv_ptr;  // Already 256B aligned
  auto rhs_scatter_aligned_ptr = lhs_scatter_aligned_ptr + num_gemms * tensor_scaling_sinv_aligment;

  size_t lhs_dtype_bytes = te_dtype_bytes(lhs_dtype);
  size_t rhs_dtype_bytes = te_dtype_bytes(rhs_dtype);
  size_t lhs_sinv_dtype_bytes = te_dtype_bytes(lhs_sinv_dtype);
  size_t rhs_sinv_dtype_bytes = te_dtype_bytes(rhs_sinv_dtype);
  size_t bias_dtype_bytes = te_dtype_bytes(bias_dtype);
  size_t out_dtype_bytes = te_dtype_bytes(out_dtype);

  if (is_tensor_scaling) {
    size_t dpitch = tensor_scaling_sinv_aligment;
    size_t spitch = lhs_sinv_dtype_bytes;
    size_t width = lhs_sinv_dtype_bytes;
    size_t height = lhs_sinv_size;
    cudaMemcpy2DAsync(lhs_scatter_aligned_ptr, dpitch, lhs_sinv_ptr, spitch, width, height,
                      cudaMemcpyDeviceToDevice, stream);
    spitch = rhs_sinv_dtype_bytes;
    width = rhs_sinv_dtype_bytes;
    height = rhs_sinv_size;
    cudaMemcpy2DAsync(rhs_scatter_aligned_ptr, dpitch, rhs_sinv_ptr, spitch, width, height,
                      cudaMemcpyDeviceToDevice, stream);
    lhs_sinv_ptr = lhs_scatter_aligned_ptr;
    rhs_sinv_ptr = rhs_scatter_aligned_ptr;
  }

  NVTE_CHECK(lhs_dtype_bytes == rhs_dtype_bytes, "sizeof(lhs_dtype) != sizeof(rhs_dtype)");
  NVTE_CHECK(lhs_sinv_dtype_bytes == rhs_sinv_dtype_bytes,
             "sizeof(lhs_sinv_dtype) != sizeof(rhs_sinv_dtype)");

  size_t expected_lhs_size = m * k;
  size_t expected_rhs_size = is_grouped_dense_wgrad ? (k * n) : (num_gemms * k * n);
  size_t expected_out_size = is_grouped_dense_wgrad ? (num_gemms * m * n) : (m * n);
  size_t actual_lhs_size = product(lhs_data.dimensions());
  size_t actual_rhs_size = product(rhs_data.dimensions());
  size_t actual_out_size = product(output->dimensions());
  NVTE_CHECK(expected_lhs_size == actual_lhs_size, "Unexpected lhs size! Expect ",
             expected_lhs_size, ", got ", actual_lhs_size);
  if (!is_grouped_dense_wgrad) {
    NVTE_CHECK(expected_rhs_size == actual_rhs_size,
               "Unexpected rhs size! Expect num_gemms * n * k = ", num_gemms, " * ", n, " * ", k,
               " = ", expected_rhs_size, ", got ", actual_rhs_size);
    NVTE_CHECK(expected_out_size == actual_out_size, "Unexpected output size! Expect m * n = ", m,
               " * ", n, " = ", expected_out_size, ", got ", actual_out_size);
  } else {
    NVTE_CHECK(expected_rhs_size == actual_rhs_size, "Unexpected rhs size! Expect k * n = ", k,
               " * ", n, " = ", expected_rhs_size, ", got ", actual_rhs_size);
    NVTE_CHECK(expected_out_size == actual_out_size,
               "Unexpected output size! Expect num_gemms * m * n = ", num_gemms, " * ", m, " * ", n,
               " = ", expected_out_size, ", got ", actual_out_size);
  }

  size_t dim_list_bytes = sizeof(int32_t) * num_gemms;
  std::vector<int32_t> dim_list_host(num_gemms);
  auto dim_list_ptr = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  cudaMemcpyAsync(dim_list_host.data(), dim_list_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);
  size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
  if (!is_grouped_dense_wgrad) {
    NVTE_CHECK(m == sum_group_sizes, "Unexpected group_sizes! M = ", m,
               ", got sum(group_sizes)=", sum_group_sizes);
  } else {
    NVTE_CHECK(k == sum_group_sizes, "Unexpected group_sizes! K = ", k,
               ", got sum(group_sizes)=", sum_group_sizes);
  }

  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;
  auto bias_shape = std::vector<size_t>{has_bias ? n : 0};
  const int arch = cuda::sm_arch();

  if (arch < 100 && is_fp8_gemm) {
    NVTE_CHECK(!lhs_is_trans && rhs_is_trans,
               "For SM90 or older archs and FP8 input, only NT (row-major) GEMM is supported, ",
               "got lhs_is_trans=", lhs_is_trans, ", rhs_is_trans=", rhs_is_trans);
  }

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> lhs_swizzle_wrapper_list;  // For MXFP8 scale_inv swizzling
  std::vector<TensorWrapper> rhs_swizzle_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> lhs_swizzle_list;
  std::vector<NVTETensor> rhs_swizzle_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  size_t lhs_sinv_total_size = 0;
  size_t rhs_sinv_total_size = 0;

  std::vector<void *> zero_out_dptr_list;
  std::vector<size_t> zero_out_size_list;

  for (size_t i = 0; i < num_gemms; i++) {
    // Matrix data shapes
    size_t m_i = dim_list_host[i];
    auto lhs_shape_i = std::vector<size_t>{m_i, k};
    auto rhs_shape_i = std::vector<size_t>{rhs_is_trans ? n : k, rhs_is_trans ? k : n};
    auto out_shape_i = std::vector<size_t>{m_i, n};
    if (is_grouped_dense_wgrad) {
      size_t k_i = dim_list_host[i];
      lhs_shape_i[0] = lhs_is_trans ? k_i : m;
      lhs_shape_i[1] = lhs_is_trans ? m : k_i;
      rhs_shape_i[0] = rhs_is_trans ? n : k_i;
      rhs_shape_i[1] = rhs_is_trans ? k_i : n;
      out_shape_i[0] = m;
      out_shape_i[1] = n;
    }

    size_t lhs_size = lhs_shape_i[0] * lhs_shape_i[1];
    size_t rhs_size = rhs_shape_i[0] * rhs_shape_i[1];
    size_t out_size = out_shape_i[0] * out_shape_i[1];
    bool is_empty_gemm = lhs_size == 0 || rhs_size == 0;
    if (is_empty_gemm && out_size > 0) {
      zero_out_dptr_list.push_back(out_ptr);
      zero_out_size_list.push_back(out_size * out_dtype_bytes);
    }

    // Set matrix data pointers
    auto lhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto rhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto out_i = TensorWrapper(static_cast<void *>(out_ptr), out_shape_i, out_dtype);
    void *lhs_vptr = static_cast<void *>(lhs_ptr);
    void *rhs_vptr = static_cast<void *>(rhs_ptr);
    if (rhs_use_colwise)  // MatA to enter cuBLAS
      rhs_i.set_columnwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
    else
      rhs_i.set_rowwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
    if (lhs_use_colwise)  // MatB to enter cuBLAS
      lhs_i.set_columnwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
    else
      lhs_i.set_rowwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);

    // Set scale_inv shapes and pointers
    void *rhs_sinv_vptr = static_cast<void *>(rhs_sinv_ptr);
    void *lhs_sinv_vptr = static_cast<void *>(lhs_sinv_ptr);
    size_t lhs_sinv_size_i = 0;
    size_t rhs_sinv_size_i = 0;
    if (is_tensor_scaling) {
      auto tensor_scaling_sinv_shape = std::vector<size_t>{1};
      // If is_empty_gemm, scale_inv does not have the corresponding value, do not move the pointers
      if (!is_empty_gemm) {
        lhs_sinv_size_i = tensor_scaling_sinv_aligment / lhs_sinv_dtype_bytes;
        rhs_sinv_size_i = tensor_scaling_sinv_aligment / rhs_sinv_dtype_bytes;
      }
      if (rhs_use_colwise)  // MatA to enter cuBLAS
        rhs_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, tensor_scaling_sinv_shape);
      else
        rhs_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, tensor_scaling_sinv_shape);
      if (lhs_use_colwise)  // MatB to enter cuBLAS
        lhs_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, tensor_scaling_sinv_shape);
      else
        lhs_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, tensor_scaling_sinv_shape);
    } else if (is_mxfp8_scaling) {
      auto lhs_swizzle_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
      auto rhs_swizzle_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
      void *swizzled_lhs_sinv_vptr = static_cast<void *>(swizzled_lhs_sinv_ptr);
      void *swizzled_rhs_sinv_vptr = static_cast<void *>(swizzled_rhs_sinv_ptr);

      // {lhs, rhs}_swizzle_i point to unswizzled scale_inv data as input, while {lhs, rhs}_i
      // point to swizzled scale_inv data (store on workspace, only used for GEMM).
      // Note: even if is_empty_gemm is true, sinv are still non-empty, need to move the pointers
      auto lhs_sinv_shape_i =
          get_mxfp8_scale_shape(lhs_shape_i[0], lhs_shape_i[1], lhs_use_colwise);
      auto rhs_sinv_shape_i =
          get_mxfp8_scale_shape(rhs_shape_i[0], rhs_shape_i[1], rhs_use_colwise);
      lhs_sinv_size_i = lhs_sinv_shape_i[0] * lhs_sinv_shape_i[1];
      rhs_sinv_size_i = rhs_sinv_shape_i[0] * rhs_sinv_shape_i[1];
      if (lhs_use_colwise) {
        lhs_swizzle_i.set_columnwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
        lhs_swizzle_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
        lhs_i.set_columnwise_scale_inv(swizzled_lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      } else {
        lhs_swizzle_i.set_rowwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
        lhs_swizzle_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
        lhs_i.set_rowwise_scale_inv(swizzled_lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      }
      if (rhs_use_colwise) {
        rhs_swizzle_i.set_columnwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
        rhs_swizzle_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
        rhs_i.set_columnwise_scale_inv(swizzled_rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      } else {
        rhs_swizzle_i.set_rowwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
        rhs_swizzle_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
        rhs_i.set_rowwise_scale_inv(swizzled_rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      }

      if (!is_empty_gemm) {
        lhs_swizzle_wrapper_list.push_back(std::move(lhs_swizzle_i));
        rhs_swizzle_wrapper_list.push_back(std::move(rhs_swizzle_i));
        lhs_swizzle_list.push_back(lhs_swizzle_wrapper_list.back().data());
        rhs_swizzle_list.push_back(rhs_swizzle_wrapper_list.back().data());
      }
    } else {
      NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING,
                 "Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);

    // Update pointer for the next GEMM pair
    lhs_ptr += lhs_size * lhs_dtype_bytes;
    rhs_ptr += rhs_size * rhs_dtype_bytes;
    out_ptr += out_size * out_dtype_bytes;
    if (is_fp8_gemm) {
      lhs_sinv_ptr += lhs_sinv_size_i * lhs_sinv_dtype_bytes;
      rhs_sinv_ptr += rhs_sinv_size_i * rhs_sinv_dtype_bytes;
      lhs_sinv_total_size += lhs_sinv_size_i;
      rhs_sinv_total_size += rhs_sinv_size_i;
      if (is_mxfp8_scaling) {
        swizzled_lhs_sinv_ptr += lhs_sinv_size_i * lhs_sinv_dtype_bytes;
        swizzled_rhs_sinv_ptr += rhs_sinv_size_i * rhs_sinv_dtype_bytes;
      }
    }
    if (has_bias) bias_ptr += n * bias_dtype_bytes;

    // Move objects to the lists to keep them alive
    if (is_empty_gemm) continue;
    lhs_wrapper_list.push_back(std::move(lhs_i));
    rhs_wrapper_list.push_back(std::move(rhs_i));
    out_wrapper_list.push_back(std::move(out_i));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }

  if (is_fp8_gemm) {
    if (is_tensor_scaling) {
      lhs_sinv_size *= tensor_scaling_sinv_aligment;
      rhs_sinv_size *= tensor_scaling_sinv_aligment;
    }
    NVTE_CHECK(lhs_sinv_total_size <= lhs_sinv_size, "Actual total lhs_sinv size ",
               lhs_sinv_total_size, " exceeds estimated upper bound ", lhs_sinv_size);
    NVTE_CHECK(rhs_sinv_total_size <= rhs_sinv_size, "Actual total rhs_sinv size ",
               rhs_sinv_total_size, " exceeds estimated upper bound ", rhs_sinv_size);
  }

  size_t num_non_empty_gemms = lhs_list.size();

  if (is_mxfp8_scaling) {
    for (int i = 0; i < num_non_empty_gemms; i++) {
      // The i-th GEMM will use the (i % num_streams)-th stream to compute,
      // use the same stream to swizzle the scaling factors to make sure that
      // the swizzling is done before the GEMM computation starts.
      int stream_id = i % num_streams;
      cudaStream_t stream_i = nvte_get_compute_stream(stream_id);
      nvte_swizzle_scaling_factors(lhs_swizzle_list[i], lhs_list[i], stream_i);
      nvte_swizzle_scaling_factors(rhs_swizzle_list[i], rhs_list[i], stream_i);
    }
  }

  // Launch zero-out kernels before the GEMM calls to use the sync in the multi-stream GEMM
  size_t num_zero_outs = zero_out_dptr_list.size();
  for (int i = 0; i < num_zero_outs; i++) {
    int stream_id = i % num_streams;
    cudaStream_t stream_i = nvte_get_compute_stream(stream_id);
    void *dptr = zero_out_dptr_list[i];
    size_t count = zero_out_size_list[i];
    NVTE_CHECK_CUDA(cudaMemsetAsync(dptr, 0, count, stream_i));
  }

  nvte_multi_tensor_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                         pre_gelu_list.data(), num_non_empty_gemms, rhs_is_trans, lhs_is_trans,
                         grad, workspace_list.data(), accumulate, use_split_accumulator,
                         num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs_data
                                  .Arg<Buffer_Type>()      // lhs_sinv
                                  .Arg<Buffer_Type>()      // rhs_data
                                  .Arg<Buffer_Type>()      // rhs_sinv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Arg<Buffer_Type>()      // group_offset
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("M")
                                  .Attr<int64_t>("N")
                                  .Attr<int64_t>("K")
                                  .Attr<bool>("lhs_is_trans")
                                  .Attr<bool>("rhs_is_trans")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("has_bias")
                                  .Attr<bool>("is_grouped_dense_wgrad"));

}  // namespace jax
}  // namespace transformer_engine
