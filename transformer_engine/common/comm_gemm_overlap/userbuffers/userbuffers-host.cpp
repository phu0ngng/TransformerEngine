/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <map>
#include <utility>

#include "common/util/cuda_driver.h"
#include "common/util/cuda_nvml.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"
#include "common/util/system.h"
#include "ipcsocket.h"
#include "userbuffers.h"

#ifdef NVTE_UB_WITH_MPI
static MPI_Comm EXT_COMM_WORLD = MPI_COMM_WORLD;
static MPI_Comm EXT_COMM_INTRA;

#define UB_MPI_CHECK(expr)                                                                   \
  do {                                                                                       \
    const int mpicode = (expr);                                                              \
    if (mpicode != MPI_SUCCESS) {                                                            \
      char mpimsg[MPI_MAX_ERROR_STRING];                                                     \
      int mpilen;                                                                            \
      MPI_Error_string(mpicode, mpimsg, &mpilen);                                            \
      std::vector<char> errmsg(1024);                                                        \
      snprintf(errmsg.data(), errmsg.size(), "%s:%d in function %s: %s", __FILE__, __LINE__, \
               __func__, mpimsg);                                                            \
      throw std::runtime_error(errmsg.data());                                               \
    }                                                                                        \
  } while (false)

void ub_mpi_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                      ExtComm comm) {
  int numranks;
  UB_MPI_CHECK(MPI_Comm_size(comm, &numranks));
  assert(globalbytes == numranks * localbytes);
  UB_MPI_CHECK(
      MPI_Allgather(localdata, localbytes, MPI_BYTE, globaldata, localbytes, MPI_BYTE, comm));
}

void ub_mpi_barrier(ExtComm comm) { UB_MPI_CHECK(MPI_Barrier(comm)); }
#else
#define EXT_COMM_WORLD "world"
#define EXT_COMM_INTRA "intra"
#endif

#define MULTICAST_GB_TOTAL 512

#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define IPCCHECK(cmd)                                                                           \
  do {                                                                                          \
    ipcSocketResult_t r = cmd;                                                                  \
    if (r != ipcSocketSuccess) {                                                                \
      printf("Failed, UDS error %s:%d '%s'\n", __FILE__, __LINE__, ipcSocketGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while (0)

#define IPCCHECKGOTO(call, RES, label)                           \
  do {                                                           \
    RES = call;                                                  \
    if (RES != ipcSocketSuccess && RES != ipcSocketInProgress) { \
      goto label;                                                \
    }                                                            \
  } while (0);

bool has_mnnvl_fabric(int device_id) {
#if CUDA_VERSION < 12040
  if (getenv("NVTE_UBDEBUG")) {
    printf(
        "TransformerEngine does not support multi-node NVLINK "
        "since it was not built with CUDA version >= 12.4.\n");
  }
  return false;
#else
  // Check run-time CUDA version
  if (transformer_engine::cuda::cudart_version() < 12040) {
    if (getenv("NVTE_UBDEBUG")) {
      printf(
          "TransformerEngine does not support multi-node NVLINK "
          "since it is not being run with CUDA version >= 12.4.\n");
    }
    return false;
  }

  bool mnnvl_fabric_support = false;
  CUdevice dev;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &dev, device_id);
  int fabric_handle_supported = 0;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGetAttribute, &fabric_handle_supported,
                              CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
  if (fabric_handle_supported) {
    NVTE_CALL_CHECK_CUDA_NVML(nvmlInit_v2);
    nvmlDevice_t local_device;
    NVTE_CALL_CHECK_CUDA_NVML(nvmlDeviceGetHandleByIndex_v2, device_id, &local_device);
    nvmlGpuFabricInfoV_t fabricInfo = {};
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    fabricInfo.clusterUuid[0] = '\0';
    NVTE_CALL_CHECK_CUDA_NVML(nvmlDeviceGetGpuFabricInfoV, local_device, &fabricInfo);
    NVTE_CALL_CHECK_CUDA_NVML(nvmlShutdown);
    if (fabricInfo.state >= NVML_GPU_FABRIC_STATE_COMPLETED && fabricInfo.clusterUuid[0] != '\0') {
      mnnvl_fabric_support = true;
    }
  }
  if (getenv("NVTE_UBDEBUG")) {
    if (mnnvl_fabric_support) {
      printf("MNNVL NVLINK is supported on this platform.\n");
    } else {
      printf("MNNVL NVLINK is not supported on this platform.\n");
    }
  }
  return mnnvl_fabric_support;
#endif
}

int create_communicator_grouped2(communicator **comm, int myrank, int numranks, int mylocal,
                                 int numlocal, int mynode, int numnodes,
                                 ExtAllgatherOp ext_allgather, ExtBarrierOp ext_barrier,
                                 int pipegpus, int pipenodes, int tensorgpus, int tensornodes,
                                 bool spmd) {
  *comm = new communicator();

  (*comm)->comm_world = EXT_COMM_WORLD;
  (*comm)->_allgather = ext_allgather;
  (*comm)->_barrier = ext_barrier;
  (*comm)->nranks = numranks;
  (*comm)->myrank = myrank;
  (*comm)->free_region = 0;
  (*comm)->launch_mode = NVTE_LAUNCH_GPU | NVTE_LAUNCH_CPU;
  
  // Initialize SPMD-specific fields
  // Initialize unified per-device storage
  (*comm)->is_spmd = spmd;
  if (spmd) {
    // SPMD mode: Size = numlocal (number of devices in process)
    (*comm)->per_device_send_id.resize(numlocal, nullptr);
    (*comm)->per_device_recv_id.resize(numlocal, nullptr);
    (*comm)->per_device_flags_baseptr.resize(numlocal, nullptr);
    (*comm)->per_device_flags.resize(numlocal, nullptr);
    (*comm)->device_to_tp_rank.resize(numlocal);
    
    // Initialize device to TP rank mapping
    // For DP+TP: Device ID maps to TP rank within its TP domain
    // Example: DP=2, TP=4, 8 devices total
    //   Devices 0,1,2,3 → TP domain 0 → TP ranks 0,1,2,3
    //   Devices 4,5,6,7 → TP domain 1 → TP ranks 0,1,2,3
    for (int dev_idx = 0; dev_idx < numlocal; dev_idx++) {
      (*comm)->device_to_tp_rank[dev_idx] = dev_idx % tensorgpus;  // tensorgpus = tp_size
    }
    
    printf("[DEBUG] SPMD: Device to TP rank mapping initialized (tp_size=%d)\n", tensorgpus);
    for (int dev_idx = 0; dev_idx < numlocal; dev_idx++) {
      printf("[DEBUG] SPMD: Device %d → TP rank %d\n", dev_idx, (*comm)->device_to_tp_rank[dev_idx]);
    }
    fflush(stdout);
  } else {
    // Single-process-single-device mode: Size = 1
    (*comm)->per_device_send_id.resize(1, nullptr);
    (*comm)->per_device_recv_id.resize(1, nullptr);
    (*comm)->per_device_flags_baseptr.resize(1, nullptr);
    (*comm)->per_device_flags.resize(1, nullptr);
    (*comm)->device_to_tp_rank.resize(1);
    (*comm)->device_to_tp_rank[0] = mylocal % tensorgpus;
  }

  if (spmd) {
    printf("[DEBUG] SPMD bootstrap: Setting up communicator for %d devices in single process\n", numlocal);
    printf("[DEBUG] SPMD: myrank=%d, numranks=%d, mylocal=%d, numlocal=%d\n", 
           myrank, numranks, mylocal, numlocal);
    fflush(stdout);
  }

  int cur_dev, ndev;
  cudaDeviceProp device_prop;
  NVTE_CHECK_CUDA(cudaGetDevice(&cur_dev));
  NVTE_CHECK_CUDA(cudaGetDeviceCount(&ndev));
  
  if (spmd) {
    // SPMD mode: Initialize all devices in this process during bootstrap
    printf("[DEBUG] SPMD: Initializing all %d devices in process (bootstrap from device %d)\n", 
           numlocal, cur_dev);
    fflush(stdout);
    
    // Verify all devices have similar properties (use device 0 as reference)
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
    (*comm)->sm_arch = device_prop.major;
    
    // Initialize each device in the process
    for (int dev_idx = 0; dev_idx < numlocal; dev_idx++) {
      printf("[DEBUG] SPMD: Initializing device %d\n", dev_idx);
      fflush(stdout);
      
      NVTE_CHECK_CUDA(cudaSetDevice(dev_idx));
      
      // Verify device properties are consistent
      cudaDeviceProp dev_prop;
      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, dev_idx));
      if (dev_prop.major != device_prop.major) {
        printf("[WARNING] Device %d has different SM arch (%d) than device 0 (%d)\n", 
               dev_idx, dev_prop.major, device_prop.major);
      }
      
      // Initialize device context
      NVTE_CHECK_CUDA(cudaFree(0));  // Initialize CUDA context on this device
    }
    
    // Set timeout based on device 0
    int device_clock = 0;
    int sec_timeout = getenv("UB_TIMEOUT") ? atoi(getenv("UB_TIMEOUT")) : 110;
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, 0));
    (*comm)->ub_timeout = 1000ull * device_clock * sec_timeout;
    
    // Restore original device context
    NVTE_CHECK_CUDA(cudaSetDevice(cur_dev));
    
    printf("[DEBUG] SPMD: All %d devices initialized, restored to device %d\n", numlocal, cur_dev);
    fflush(stdout);
  } else {
    // Multi-process mode: Original single device setup
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&device_prop, cur_dev));
    (*comm)->sm_arch = device_prop.major;
    
    int device_clock = 0;
    int sec_timeout = getenv("UB_TIMEOUT") ? atoi(getenv("UB_TIMEOUT")) : 110;
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
    (*comm)->ub_timeout = 1000ull * device_clock * sec_timeout;
  }
  
  // (*comm)->use_rr_kernel = device_prop.major == 8;
  (*comm)->use_rr_kernel = 0;
  (*comm)->push = 1;
  (*comm)->use_ce = 0;
  (*comm)->cga_size = 2;
  for (int i = 0; i < userbuffers_op_types; i++) (*comm)->basecounter[i] = 0;
  
  int device_clock = 0;
  int sec_timeout = getenv("UB_TIMEOUT") ? atoi(getenv("UB_TIMEOUT")) : 110;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
  (*comm)->ub_timeout = 1000ull * device_clock * sec_timeout;
  
  if ((*comm)->myrank == 0) {
    printf("UB_TIMEOUT is set to %d sec, %" PRIu64 " cycles, freq: %dkhz\n", sec_timeout,
           (*comm)->ub_timeout, device_clock);
  }

  (*comm)->comm_intra = EXT_COMM_INTRA;
  (*comm)->nvrank = mylocal;
  (*comm)->nvsize = numlocal;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int core;
  if (mylocal == 0) core = 50;
  if (mylocal == 1) core = 58;
  if (mylocal == 2) core = 18;
  if (mylocal == 3) core = 26;
  if (mylocal == 4) core = 114;
  if (mylocal == 5) core = 122;
  if (mylocal == 6) core = 82;
  if (mylocal == 7) core = 90;

  CPU_SET(core, &cpuset);
  if (!getenv("NVTE_NODOUBLE")) {
    if (core > 128)
      CPU_SET(core - 128, &cpuset);
    else
      CPU_SET(core + 128, &cpuset);
  }
  if (getenv("NVTE_DOPIN")) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  if (ndev == numlocal) {  // all visible devices
    if (cur_dev != mylocal)
      printf("%d: device used %d[%d] ,resetting device to %d\n", myrank, cur_dev, ndev, mylocal);
    NVTE_CHECK_CUDA(cudaSetDevice(mylocal));
  }
  (*comm)->mydev = cur_dev;
  // FIXME need to check that numlocal is multiple of pipegpus x tensorgpus
  // ar1 is data
  int divgpus = pipegpus * tensorgpus;
  int datagpus = numlocal / divgpus;
  (*comm)->ar_nvsize = datagpus;
  (*comm)->ar_firstgpu = mylocal - ((mylocal / tensorgpus) % datagpus) * tensorgpus;
  (*comm)->ar_nvrank = (mylocal - (*comm)->ar_firstgpu) / tensorgpus;
  // ar2 is tensor
  (*comm)->ar2_nvsize = tensorgpus;
  (*comm)->ar2_firstgpu = mylocal - mylocal % tensorgpus;
  (*comm)->ar2_nvrank = mylocal - (*comm)->ar2_firstgpu;
  // ar2 has step equal to ar_nvsize
  int allnodes = numranks / numlocal;
  int nodeid = myrank / numlocal;

  (*comm)->num_nodes = numnodes;
  (*comm)->my_node = mynode;

#define NBUF 2

#if CUDART_VERSION >= 12010
  bool mnnvl_fabric = has_mnnvl_fabric(cur_dev);
  if (!transformer_engine::getenv<bool>("UB_SKIPMC") &&
      transformer_engine::cuda::supports_multicast() && (*comm)->ar2_nvsize > 1 && !spmd) {
    // multicast init only for TP ops (____2 operations)
    size_t mc_maxsize = MULTICAST_GB_TOTAL * (1ull << 30);
    (*comm)->mc_offset = 0;
    (*comm)->use_mc = 1;
    size_t gran;
    CUmulticastObjectProp mcProp = {};
    mcProp.numDevices = (*comm)->ar2_nvsize;
    mcProp.size = (*comm)->mc_maxsize;
    mcProp.handleTypes =
        mnnvl_fabric ? CU_MEM_HANDLE_TYPE_FABRIC : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    NVTE_CALL_CHECK_CUDA_DRIVER(
        cuMulticastGetGranularity, &gran, &mcProp,
        static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mc_maxsize = ((mc_maxsize + gran - 1) / gran) * gran;
    mcProp.size = mc_maxsize;
    (*comm)->mc_maxsize = mc_maxsize;
    if ((*comm)->ar2_nvrank == 0)
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastCreate, &(*comm)->mc_handle, &mcProp);

    if (mnnvl_fabric) {
      CUmemFabricHandle *exphndl =
          reinterpret_cast<CUmemFabricHandle *>(malloc(sizeof(CUmemFabricHandle)));
      CUmemFabricHandle *tmphndl =
          reinterpret_cast<CUmemFabricHandle *>(malloc(sizeof(CUmemFabricHandle)));
      CUmemFabricHandle *exphndls;
      NVTE_CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&exphndls),
                                     (*comm)->nvsize * sizeof(CUmemFabricHandle)));
      if ((*comm)->ar2_nvrank == 0)
        NVTE_CALL_CHECK_CUDA_DRIVER(cuMemExportToShareableHandle, static_cast<void *>(tmphndl),
                                    (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
      for (int grp = 0; grp < (*comm)->ar_nvsize;
           grp++) {  // we do N broadcasts for N TP groups in NVL domain
        int root = grp * (*comm)->ar2_nvsize;

        // It just needs to be a bcast but reuse existing allgather comm
        (*comm)->_allgather(
            reinterpret_cast<void *>(exphndls), (*comm)->nvsize * sizeof(CUmemFabricHandle),
            reinterpret_cast<void *>(tmphndl), sizeof(CUmemFabricHandle), (*comm)->comm_intra);

        //save data if brodcast was from rank 0 in our group
        if ((*comm)->ar2_firstgpu == root)
          memcpy(exphndl, exphndls + root, sizeof(CUmemFabricHandle));
      }
      if ((*comm)->ar2_nvrank != 0)
        NVTE_CALL_CHECK_CUDA_DRIVER(cuMemImportFromShareableHandle, &(*comm)->mc_handle,
                                    reinterpret_cast<void *>(exphndl), CU_MEM_HANDLE_TYPE_FABRIC);
      free(exphndl);
      free(tmphndl);
      NVTE_CHECK_CUDA(cudaFreeHost(exphndls));
    } else {
      // Broadcast the a POSIX file descriptor from the local root rank to other local ranks.
      // NOTE: This cannot be done via MPI_Bcast or other external comm libraries. They mangle the
      //       file descriptor and prevent cuMemImportFromShareableHandle() from correctly
      //       interpreting the file. Instead, we use Unix domain sockets for the kernel to
      //       recreate the correct file descriptor on every receiving rank.
      int fd;
      volatile uint32_t abortFlag = 0;
      IpcSocketHandle ipcSock = {0};
      uint64_t opId = 0xdeadcafe0000 + (*comm)->my_node;
      ipcSocketResult_t ret = ipcSocketSuccess;
      IPCCHECK(ipcSocketInit(&ipcSock, (*comm)->ar2_nvrank, (uint64_t)opId, &abortFlag));
      (*comm)->_barrier((*comm)->comm_world);

      if ((*comm)->ar2_nvrank == 0) {
        NVTE_CALL_CHECK_CUDA_DRIVER(
            cuMemExportToShareableHandle, reinterpret_cast<void *>(&fd), (*comm)->mc_handle,
            static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            (uint64_t)0);

        for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
          (*comm)->_barrier((*comm)->comm_intra);
          IPCCHECKGOTO(ipcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId), ret, error);
        }
      } else {
        for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
          (*comm)->_barrier((*comm)->comm_intra);
          if ((*comm)->ar2_nvrank == p) IPCCHECKGOTO(ipcSocketRecvFd(&ipcSock, &fd), ret, error);
        }
      }

    error:
      if ((*comm)->ar2_nvrank != 0) {
        NVTE_CALL_CHECK_CUDA_DRIVER(
            cuMemImportFromShareableHandle, &(*comm)->mc_handle, reinterpret_cast<void *>(fd),
            static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      }
      IPCCHECK(ipcSocketClose(&ipcSock));
      close(fd);
    }
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastAddDevice, (*comm)->mc_handle,
                                (CUdeviceptr)(*comm)->mydev);

    CUdeviceptr mc_va;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressReserve, &mc_va, mc_maxsize, (size_t)0, (CUdeviceptr)0U,
                                (uint64_t)0);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemMap, mc_va, mc_maxsize, (size_t)0, (*comm)->mc_handle,
                                (uint64_t)0);

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = (*comm)->mydev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemSetAccess, mc_va, mc_maxsize,
                                const_cast<CUmemAccessDesc *>(&accessDesc), (size_t)1);

    (*comm)->mc_baseptr = reinterpret_cast<void *>(mc_va);
    (*comm)->_barrier((*comm)->comm_world);
    if (!(*comm)->myrank) printf("MC initialized succesfully, window size = %ld\n", mc_maxsize);
  } else {
#endif
    if (!(*comm)->myrank) printf("MC NOT initialized and used\n");
    (*comm)->mc_maxsize = 0;
    (*comm)->mc_offset = 0;
    (*comm)->use_mc = 0;
#if CUDART_VERSION >= 12010
  }
#endif

#define LOCALSIZE 4 * (NVTE_REG0_OFFSET(*comm) + NVTE_REG0_FLAGS + NVTE_REG0_COMMBUFFER * NBUF)
  // peer pointers + op flags + comm buffer
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm, true, spmd); 
  
  // Define GPU page constants before use
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

  // Unified per-device array allocation
  int num_devices_per_process = spmd ? (*comm)->nvsize : 1;
  printf("[DEBUG] Allocating communication arrays for %d device(s) (SPMD=%s)\n", 
         num_devices_per_process, spmd ? "true" : "false");
  fflush(stdout);
  
  // Store current device to restore later
  int original_device;
  NVTE_CHECK_CUDA(cudaGetDevice(&original_device));
  
  // Allocate communication arrays on each device
  for (int dev_idx = 0; dev_idx < num_devices_per_process; dev_idx++) {
    int target_device = spmd ? dev_idx : original_device;
    printf("[DEBUG] Allocating communication arrays on device %d\n", target_device);
    fflush(stdout);
    
    NVTE_CHECK_CUDA(cudaSetDevice(target_device));
    
    // Allocate communication arrays
    NVTE_CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void **>(&(*comm)->per_device_send_id[dev_idx]), 
                   (*comm)->nranks * sizeof(int)));
    NVTE_CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void **>(&(*comm)->per_device_recv_id[dev_idx]),
                   NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    
    // Allocate flags arrays (GPU page-aligned)
    NVTE_CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void **>(&(*comm)->per_device_flags_baseptr[dev_idx]), 
                   2 * GPU_PAGE_SIZE));
    NVTE_CHECK_CUDA(cudaMemset((*comm)->per_device_flags_baseptr[dev_idx], 0, 2 * GPU_PAGE_SIZE));
    (*comm)->per_device_flags[dev_idx] = reinterpret_cast<int *>(
        ((CUdeviceptr)(*comm)->per_device_flags_baseptr[dev_idx] + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK);
    
    // Initialize arrays
    NVTE_CHECK_CUDA(cudaMemset((*comm)->per_device_send_id[dev_idx], 0, (*comm)->nranks * sizeof(int)));
    NVTE_CHECK_CUDA(
        cudaMemset((*comm)->per_device_recv_id[dev_idx], 0, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    
    printf("[DEBUG] Device %d arrays allocated (send_id: %zu bytes, recv_id: %zu bytes, flags: %zu bytes)\n",
           target_device, (*comm)->nranks * sizeof(int), NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int), 2 * GPU_PAGE_SIZE);
    fflush(stdout);
  }
  
  // Set legacy pointers to first device for backward compatibility
  (*comm)->flags_baseptr = (*comm)->per_device_flags_baseptr[0];
  (*comm)->flags = (*comm)->per_device_flags[0];
  
  // Restore original device
  NVTE_CHECK_CUDA(cudaSetDevice(original_device));
      
  printf("[DEBUG] Communication arrays allocated successfully for %d device(s)\n", num_devices_per_process);
  fflush(stdout);
  (*comm)->sms = 16;
  (*comm)->threads = 1024;

  using namespace std;

  sched_param param;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_getschedparam(&attr, &param);
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);

  pthread_attr_setschedparam(&attr, &param);

  if (getenv("NVTE_UBDEBUG"))
    printf(
        "%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP x%d TPGROUP "
        "%dx%d\n",
        myrank, numranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
        (*comm)->ar_nvrank, (*comm)->my_node, (*comm)->ar2_nvrank, (*comm)->ar_nvsize,
        (*comm)->num_nodes, (*comm)->ar2_nvsize);
  fflush(NULL);

  return 0;
}

// SPMD helper methods for dynamic device-aware computation
int communicator::get_current_nvrank() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;  // nvrank = device ID in SPMD
  } else {
    return nvrank;  // Use stored value for multi-process
  }
}

int communicator::get_current_ar2_nvrank() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device - ar2_firstgpu;  // ar2_nvrank = device - first_gpu
  } else {
    return ar2_nvrank;  // Use stored value for multi-process
  }
}

int communicator::get_current_mydev() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return current_device;  // mydev = current device in SPMD
  } else {
    return mydev;  // Use stored value for multi-process
  }
}

int communicator::get_current_tp_rank() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return device_to_tp_rank[current_device];  // Get TP rank from mapping
  } else {
    return nvrank % ar2_nvsize;  // Compute TP rank for multi-process
  }
}

int* communicator::get_current_send_id() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return per_device_send_id[current_device];
  } else {
    return per_device_send_id[0];  // Use index 0 for single-device
  }
}

int* communicator::get_current_recv_id() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return per_device_recv_id[current_device];
  } else {
    return per_device_recv_id[0];  // Use index 0 for single-device
  }
}

int* communicator::get_current_flags() const {
  if (is_spmd) {
    int current_device;
    cudaGetDevice(&current_device);
    return per_device_flags[current_device];
  } else {
    return per_device_flags[0];  // Use index 0 for single-device
  }
}

int create_communicator_grouped(communicator **comm, int myrank, int numranks, int mylocal,
                                int numlocal, int mynode, int numnodes,
                                ExtAllgatherOp ext_allgather, ExtBarrierOp ext_barrier,
                                int pipegpus, int pipenodes) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_allgather, ext_barrier, pipegpus, pipenodes, 1, 1, false);
}

int create_communicator(communicator **comm, int myrank, int numranks, int mylocal, int numlocal,
                        int mynode, int numnodes, ExtAllgatherOp ext_allgather,
                        ExtBarrierOp ext_barrier) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_allgather, ext_barrier, 1, 1, 1, 1, false);
}

int create_communicator_grouped2_mpi(communicator **comm, int pipegpus, int pipenodes,
                                     int tensorgpus, int tensornodes) {
#ifdef NVTE_UB_WITH_MPI
  // get global numbers
  int myrank, numranks;
  UB_MPI_CHECK(MPI_Comm_rank(EXT_COMM_WORLD, &myrank));
  UB_MPI_CHECK(MPI_Comm_size(EXT_COMM_WORLD, &numranks));

  int mylocal, numlocal;
  UB_MPI_CHECK(MPI_Comm_split(EXT_COMM_WORLD, myrank / tensorgpus, myrank, &EXT_COMM_INTRA));
  UB_MPI_CHECK(MPI_Comm_rank(EXT_COMM_INTRA, &mylocal));
  UB_MPI_CHECK(MPI_Comm_size(EXT_COMM_INTRA, &numlocal));

  // find internode numbers and make internode communicator
  NVTE_CHECK_CUDA(cudaFree(0));
  int mynode, numnodes;
  mynode = myrank / numlocal;
  numnodes = numranks / numlocal;

  // finally call the abstracted constructor with MPI info
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      &ub_mpi_allgather, &ub_mpi_barrier, pipegpus, pipenodes,
                                      tensorgpus, tensornodes, false);
#else
  NVTE_ERROR(std::string("Bootstrapping Userbuffers with MPI requires building") +
             std::string("Transformer Engine with NVTE_UB_WITH_MPI=1 and MPI_HOME=/path/to/mpi"));
#endif
}

int create_communicator_grouped_mpi(communicator **comm, int pipegpus, int pipenodes) {
  return create_communicator_grouped2_mpi(comm, pipegpus, pipenodes, 1, 1);
}

int create_communicator_mpi(communicator **comm) {
  return create_communicator_grouped2_mpi(comm, 1, 1, 1, 1);
}

void destroy_communicator(communicator *comm) {
  // Clear memory allocated in register_user_buffer_collective calls
  for (int hndl = comm->free_region - 1; hndl >= 0; hndl--) {
    if (comm->use_mc && comm->mem_dealloc[hndl]) {
      // Unbind the local device buffer from the Multicast handle
      CUdevice dev;
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &dev, comm->get_current_mydev());
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastUnbind, comm->mc_handle, dev, comm->uc_offsets[hndl],
                                  comm->mem_size[hndl]);

      // Unmap memory addresses and release handles for both peer and own buffers
      for (int rank = 0; rank < comm->nvsize; rank++) {
        NVTE_CALL_CHECK_CUDA_DRIVER(cuMemUnmap,
                                    reinterpret_cast<CUdeviceptr>(comm->peer_ptr[hndl][rank]),
                                    comm->mem_size[hndl]);
        NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->uchandles[hndl][rank]);
      }
      free(reinterpret_cast<void *>(comm->uchandles[hndl]));

      // Free memory reserved for buffer allocations
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressFree, comm->ucbase_ptr[hndl],
                                  static_cast<size_t>(comm->mem_size[hndl] * comm->nvsize));
    } else {
    for (int rank = 0; rank < comm->nvsize; rank++) {
      if (rank != comm->get_current_nvrank()) {
        NVTE_CHECK_CUDA(cudaIpcCloseMemHandle(comm->peer_ptr[hndl][rank]));
      } else if (comm->mem_dealloc[hndl]) {
          NVTE_CHECK_CUDA(cudaFree(comm->peer_ptr[hndl][rank]));
        } else {
          comm->peer_ptr[hndl][rank] = nullptr;  // remove reference to external buffer
        }
      }
    }
    free(comm->peer_ptr[hndl]);
    comm->mem_ptr[hndl] = nullptr;  // this points to already cleaned up local device buffer
  }
  // Clear memory allocated in the communicator constructor
  // Clean up unified per-device communication arrays
  printf("[DEBUG] Cleaning up per-device communication arrays (%zu devices)\n", comm->per_device_send_id.size());
  fflush(stdout);
  
  for (size_t dev_idx = 0; dev_idx < comm->per_device_send_id.size(); dev_idx++) {
    if (comm->per_device_send_id[dev_idx]) {
      NVTE_CHECK_CUDA(cudaFree(reinterpret_cast<void *>(comm->per_device_send_id[dev_idx])));
    }
    if (comm->per_device_recv_id[dev_idx]) {
      NVTE_CHECK_CUDA(cudaFree(reinterpret_cast<void *>(comm->per_device_recv_id[dev_idx])));
    }
    if (comm->per_device_flags_baseptr[dev_idx]) {
      NVTE_CHECK_CUDA(cudaFree(reinterpret_cast<void *>(comm->per_device_flags_baseptr[dev_idx])));
    }
  }
  if (comm->use_mc) {
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemUnmap, reinterpret_cast<CUdeviceptr>(comm->mc_baseptr),
                                comm->mc_maxsize);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressFree, comm->mc_baseptr, comm->mc_maxsize);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->mc_handle);
  }
  delete comm;
}

void destroy_communicator_mpi(communicator *comm) {
#ifdef NVTE_UB_WITH_MPI
  MPI_Comm_free(static_cast<MPI_Comm *>(&(comm->comm_intra)));
  destroy_communicator(comm);
#else
  NVTE_ERROR(std::string("Communicator is not bootstrapped with MPI and ") +
             std::string("can only be deallocated with destroy_communicator()."));
#endif
}

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc, bool spmd) {
  if (comm->free_region >= NVTE_MAX_REGIONS) return -1;
  int hndl = comm->free_region;
  comm->peer_ptr[hndl] = reinterpret_cast<void **>(malloc(sizeof(void *) * (comm->nvsize)));
  size_t aligned_size = bytes;
  comm->memflags[hndl] = 0;
  comm->mem_dealloc[hndl] = alloc;

  if (spmd) {
    // SPMD mode: Allocate buffers on all devices and enable peer-to-peer access
    printf("[DEBUG] SPMD register_user_buffer_collective: Allocating on all %d devices\n", comm->nvsize);
    fflush(stdout);
    
    // Store current device to restore later
    int original_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&original_device));
    
    // Allocate buffer on each device
    for (int dev_idx = 0; dev_idx < comm->nvsize; dev_idx++) {
      printf("[DEBUG] SPMD: Allocating buffer on device %d (%zu bytes)\n", dev_idx, bytes);
      fflush(stdout);
      
      NVTE_CHECK_CUDA(cudaSetDevice(dev_idx));
      
      if (alloc) {
        printf("[DEBUG] SPMD: cudaMalloc on device %d (%zu bytes)...\n", dev_idx, bytes);
        fflush(stdout);
        
        cudaError_t err = cudaMalloc(&comm->peer_ptr[hndl][dev_idx], bytes);
        if (err != cudaSuccess) {
          printf("[ERROR] SPMD: cudaMalloc failed on device %d: %s\n", dev_idx, cudaGetErrorString(err));
          fflush(stdout);
          NVTE_CHECK_CUDA(err);
        }
        
        printf("[DEBUG] SPMD: cudaMalloc succeeded, ptr=%p, about to memset...\n", comm->peer_ptr[hndl][dev_idx]);
        fflush(stdout);
        
        err = cudaMemset(comm->peer_ptr[hndl][dev_idx], 0, bytes);
        if (err != cudaSuccess) {
          printf("[ERROR] SPMD: cudaMemset failed on device %d: %s\n", dev_idx, cudaGetErrorString(err));
          fflush(stdout);
          NVTE_CHECK_CUDA(err);
        }
        
        printf("[DEBUG] SPMD: Device %d buffer allocated at %p\n", dev_idx, comm->peer_ptr[hndl][dev_idx]);
        fflush(stdout);
      }
    }
    
    // Enable peer-to-peer access between all devices
    printf("[DEBUG] SPMD: Enabling peer-to-peer access between all devices\n");
    fflush(stdout);
    
    for (int src_dev = 0; src_dev < comm->nvsize; src_dev++) {
      NVTE_CHECK_CUDA(cudaSetDevice(src_dev));
      for (int dst_dev = 0; dst_dev < comm->nvsize; dst_dev++) {
        if (src_dev != dst_dev) {
          int can_access_peer;
          NVTE_CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_peer, src_dev, dst_dev));
          if (can_access_peer) {
            cudaError_t err = cudaDeviceEnablePeerAccess(dst_dev, 0);
            if (err == cudaSuccess) {
              printf("[DEBUG] SPMD: Enabled peer access from device %d to device %d\n", src_dev, dst_dev);
            } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
              printf("[DEBUG] SPMD: Peer access already enabled from device %d to device %d\n", src_dev, dst_dev);
            } else {
              printf("[WARNING] SPMD: Failed to enable peer access from device %d to device %d: %s\n", 
                     src_dev, dst_dev, cudaGetErrorString(err));
            }
            fflush(stdout);
          } else {
            printf("[WARNING] SPMD: Device %d cannot access device %d (no peer access support)\n", src_dev, dst_dev);
            fflush(stdout);
          }
        }
      }
    }
    
    // Set the main buffer pointer to the buffer on the original device
    *gpubuff = comm->peer_ptr[hndl][original_device];
    
    // Restore original device
    NVTE_CHECK_CUDA(cudaSetDevice(original_device));
    
    // Set memory flags
    comm->memflags[hndl] = NVTE_UB_MEM_ALLOCATED;
    comm->free_region++;
    
    printf("[DEBUG] SPMD: register_user_buffer_collective completed for handle %d\n", hndl);
    fflush(stdout);
    
    return hndl;
  }

#if CUDART_VERSION >= 12010
  if (comm->use_mc && alloc) {
    bool mnnvl_fabric = has_mnnvl_fabric(comm->get_current_mydev());
    int nranks = comm->nvsize;  // total GPUs in NVLINK domain
    int myrank = comm->get_current_nvrank();
    void **remptrs = reinterpret_cast<void **>(malloc(nranks * sizeof(void *)));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = comm->get_current_mydev();
    prop.requestedHandleTypes =
        mnnvl_fabric ? CU_MEM_HANDLE_TYPE_FABRIC : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity = 0;
    NVTE_CALL_CHECK_CUDA_DRIVER(
        cuMemGetAllocationGranularity, &granularity, &prop,
        static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_MINIMUM));
    // MPI_Allreduce MAX of granularity check
    aligned_size = (bytes + granularity - 1) / granularity * granularity;

    if (comm->use_mc) {
      CUmulticastObjectProp mcProp = {};
      mcProp.numDevices = nranks;
      mcProp.size = aligned_size;
      mcProp.handleTypes = prop.requestedHandleTypes;
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMulticastGetGranularity, &granularity, &mcProp,
          static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_MINIMUM));
      aligned_size = (aligned_size + granularity - 1) / granularity * granularity;
    }

    prop.location.id = comm->get_current_mydev();
    comm->uchandles[hndl] = reinterpret_cast<CUmemGenericAllocationHandle *>(
        malloc(nranks * sizeof(CUmemGenericAllocationHandle)));
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemCreate, &(comm->uchandles[hndl][myrank]), aligned_size, &prop,
                                (uint64_t)0);

    if (mnnvl_fabric) {
      CUmemFabricHandle *exphndl;
      CUmemFabricHandle myhndl;
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMemExportToShareableHandle, &myhndl,
                                  comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_FABRIC, 0);
      NVTE_CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&exphndl),
                                     comm->nvsize * sizeof(CUmemFabricHandle)));
      comm->_allgather(reinterpret_cast<void *>(exphndl), comm->nvsize * sizeof(CUmemFabricHandle),
                       reinterpret_cast<void *>(&myhndl), sizeof(CUmemFabricHandle),
                       comm->comm_intra);
      for (int p = 0; p < nranks; p++)
        if (p != myrank)
          NVTE_CALL_CHECK_CUDA_DRIVER(cuMemImportFromShareableHandle, &comm->uchandles[hndl][p],
                                      reinterpret_cast<void *>(&exphndl[p]),
                                      CU_MEM_HANDLE_TYPE_FABRIC);
      NVTE_CHECK_CUDA(cudaFreeHost(exphndl));
    } else {
      int *peerfd = reinterpret_cast<int *>(malloc(nranks * sizeof(int)));
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMemExportToShareableHandle, reinterpret_cast<void *>(&peerfd[myrank]),
          comm->uchandles[hndl][myrank],
          static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          (uint64_t)0);

      volatile uint32_t abortFlag = 0;
      IpcSocketHandle ipcSock = {0};
      uint64_t opId = 0xdeadcafe0000 + comm->my_node;
      ipcSocketResult_t ret = ipcSocketSuccess;

      // All-gather POSIX file descriptors across local ranks
      IPCCHECK(ipcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag));
      for (int p = 1; p < nranks; p++) {
        int send_to = (myrank + p) % nranks;
        int recv_from = (myrank + nranks - p) % nranks;
        comm->_barrier(comm->comm_intra);
        IPCCHECKGOTO(ipcSocketSendFd(&ipcSock, peerfd[myrank], send_to, (uint64_t)opId), ret,
                     error);
        IPCCHECKGOTO(ipcSocketRecvFd(&ipcSock, &peerfd[recv_from]), ret, error);
      }

    error:
      IPCCHECK(ipcSocketClose(&ipcSock));

      for (int p = 0; p < nranks; p++) {
        if (p != myrank)
          NVTE_CALL_CHECK_CUDA_DRIVER(
              cuMemImportFromShareableHandle, &comm->uchandles[hndl][p],
              reinterpret_cast<void *>(peerfd[p]),
              static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(peerfd[p]);
      }
      free(peerfd);
    }
    CUdeviceptr ptr;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressReserve, &ptr, (size_t)(aligned_size * nranks),
                                (size_t)0, (CUdeviceptr)0, (uint64_t)0);
    comm->ucbase_ptr[hndl] = reinterpret_cast<void *>(ptr);
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = comm->get_current_mydev();

    for (int i = 0; i < nranks; i++) {
      remptrs[i] = reinterpret_cast<void *>(ptr + (aligned_size * i));
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMemMap, reinterpret_cast<CUdeviceptr>(remptrs[i]), aligned_size,
                                  (size_t)0, comm->uchandles[hndl][i], (uint64_t)0);
      if (i == comm->get_current_nvrank()) {
        if (hndl)
          *gpubuff = remptrs[i];
        else
          comm->gpu_ptrs = remptrs[i];
      }
      comm->peer_ptr[hndl][i] = remptrs[i];
    }
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemSetAccess, ptr, (size_t)(aligned_size * nranks),
                                const_cast<CUmemAccessDesc *>(&accessDesc), (size_t)1);

    if (hndl == 0) NVTE_CHECK_CUDA(cudaMemset(comm->gpu_ptrs, 0, aligned_size));
    NVTE_CHECK_CUDA(
        cudaMemcpy((reinterpret_cast<char *>(comm->gpu_ptrs)) + (hndl * nranks * sizeof(void *)),
                   remptrs, nranks * sizeof(void *), cudaMemcpyHostToDevice));
    free(remptrs);
    comm->memflags[hndl] = NVTE_UB_MEM_UC_CONTIG | NVTE_UB_MEM_ALLOCATED;

    if (comm->use_mc && comm->mc_maxsize >= comm->mc_offset + aligned_size) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastBindMem, comm->mc_handle, comm->mc_offset,
                                  comm->uchandles[hndl][myrank], (size_t)0 /*memOffset*/,
                                  aligned_size, (uint64_t)0);
      comm->memflags[hndl] |= NVTE_UB_MEM_MC_CREATED;
      comm->mc_ptr[hndl] = reinterpret_cast<char *>(comm->mc_baseptr) + comm->mc_offset;
      comm->uc_offsets[hndl] = comm->mc_offset;
      comm->mc_offset += aligned_size;
    } else if (!comm->myrank) {
      printf("UB: warning region %d size %ld MB registered without MC access\n", hndl,
             aligned_size / 1024 / 1024);
    }

  } else {
#endif
    if (spmd) {
      // SPMD runtime mode: Direct memory access (no IPC in single process)
      printf("[DEBUG] SPMD runtime register_user_buffer_collective: device context\n");
      fflush(stdout);
      
      if (alloc) {
        int current_device;
        NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
        printf("[DEBUG] SPMD runtime: Allocating on current device %d\n", current_device);
        fflush(stdout);
        
        NVTE_CHECK_CUDA(cudaMalloc(gpubuff, bytes));
        NVTE_CHECK_CUDA(cudaMemset(*gpubuff, 0, bytes));
        
        printf("[DEBUG] SPMD runtime: Allocated buffer at %p\n", *gpubuff);
        fflush(stdout);
      }
      
      // Set peer pointer for current device (P2P already enabled during bootstrap)
      int my_idx = comm->get_current_nvrank();
      comm->peer_ptr[hndl][my_idx] = *gpubuff;
      
      printf("[DEBUG] SPMD runtime: Set peer_ptr[%d][%d]=%p\n", hndl, my_idx, *gpubuff);
      fflush(stdout);
      
      // SPMD runtime path ends here - skip IPC code below
      // Continue to common finalization at the end
      
    } else {
      // Multi-process mode: Use CUDA IPC
      if (alloc) {
        NVTE_CHECK_CUDA(cudaMalloc(gpubuff, bytes));
        NVTE_CHECK_CUDA(cudaMemset(*gpubuff, 0, bytes));
      }

      NVTE_CHECK(comm->nvsize <= 8, "CUDA IPC supports only up to 8 GPUs in an NVLink domain.");
      cudaIpcMemHandle_t memhndl;
      NVTE_CHECK_CUDA(cudaIpcGetMemHandle(&memhndl, *gpubuff));

    cudaIpcMemHandle_t *tmp =
        reinterpret_cast<cudaIpcMemHandle_t *>(malloc(comm->nvsize * sizeof(cudaIpcMemHandle_t)));
    comm->_allgather(reinterpret_cast<void *>(tmp), comm->nvsize * sizeof(cudaIpcMemHandle_t),
                     reinterpret_cast<void *>(&memhndl), sizeof(cudaIpcMemHandle_t),
                     comm->comm_intra);

    // Check for NVLINK support before attempting IPC operations
    if (comm->nvsize > 1) {
      int current_device;
      NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
      cudaDeviceProp deviceProp;
      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, current_device));
      bool peer_access_available = false;
      for (int i = 0; i < comm->nvsize; i++) {
        if (i != comm->get_current_nvrank()) {
          int can_access_peer;
          cudaError_t peer_result = cudaDeviceCanAccessPeer(&can_access_peer, current_device, i);
          if (peer_result == cudaSuccess && can_access_peer) {
            peer_access_available = true;
            break;
          }
        }
      }
      if (!peer_access_available) {
        free(tmp);
        NVTE_ERROR(
            "No peer-to-peer access available between GPUs. This platform does not support the "
            "GPU-to-GPU "
            "communication required for multi-GPU userbuffers. Consider using single-GPU mode.");
        return 1;
      }
    }

      for (int i = 0; i < comm->nvsize; i++) {
        if (i != comm->get_current_nvrank()) {
          NVTE_CHECK_CUDA(cudaIpcOpenMemHandle(&(comm->peer_ptr[hndl][i]), tmp[i],
                                               cudaIpcMemLazyEnablePeerAccess));
        }
      }
      comm->peer_ptr[hndl][comm->get_current_nvrank()] = *gpubuff;
      NVTE_CHECK_CUDA(cudaDeviceSynchronize());

      NVTE_CHECK_CUDA(cudaMemcpy(
          reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
          comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));

      NVTE_CHECK_CUDA(cudaDeviceSynchronize());
      free(tmp);
    }  // end else (multi-process)
#if CUDART_VERSION >= 12010
  }
#endif
  comm->mem_size[hndl] = aligned_size;

  comm->mem_ptr[hndl] = *gpubuff;

  return comm->free_region++;
  printf("***** Returning *****\n");
}
