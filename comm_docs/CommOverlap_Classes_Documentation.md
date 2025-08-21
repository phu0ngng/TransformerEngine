# CommOverlap Classes Documentation

## Overview
This document describes the three main classes in Transformer Engine's communication-GEMM overlap system, located in `transformer_engine/common/include/transformer_engine/comm_gemm_overlap.h`. These classes are used by both PyTorch and JAX extensions to enable efficient communication-GEMM overlap.

## Class Hierarchy
```
CommOverlapCore (Base Class)
├── CommOverlapBase (Collective Operations)
└── CommOverlapP2PBase (Point-to-Point Operations)
```

## CommOverlapCore (Base Class)

### Purpose
The fundamental base class that provides core functionality for communication-GEMM overlap.

### Key Features
- Manages communication infrastructure (MPI or external collectives)
- Handles user buffers (`_ubuf`)
- Manages CUDA streams and events for compute and communication
- Provides virtual methods that are meant to be overridden by derived classes
- Contains common member variables like `_rank`, `_tp_size`, `_num_splits`, etc.

### Key Member Variables
```cpp
static inline communicator *_ub_comm{nullptr};
static inline bool _comm_created{false};
int _rank;
int _tp_id;
int _tp_size;
int _num_splits;
int _math_sms;
int _num_comm_sm;
int _cga_size;
int _use_ce;
int _ub_reg;
int _gemm_priority;
int _comm_priority;
bool _atomic_gemm{false};  // Key parameter for JAX integration
bool _is_p2p{false};
TensorWrapper _ubuf;
TensorWrapper _counter;
float *_ubuf_scale_inv;
bool _ubuf_scale_inv_initialized{false};
std::vector<cudaStream_t> _stream_compute;
cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _comm_launch_event;
```

### Virtual Methods (Default: "not implemented")
- `copy_into_buffer()`: Copy data into user buffer
- `bulk_overlap()`: Full GEMM + communication overlap
- `atomic_gemm_overlap_rs()`: Atomic GEMM + ReduceScatter
- `split_overlap_rs()`: Split GEMM + ReduceScatter
- `atomic_gemm_overlap_ag()`: Atomic GEMM + AllGather
- `split_overlap_ag()`: Split GEMM + AllGather

### Constructors
- **External/framework collectives-based constructor**: Uses external collective operations
- **MPI-based constructor**: Uses MPI for communication

### JAX Integration Notes
- Used by JAX extension through `CreateCommOverlapBuffer()` function
- `_atomic_gemm` parameter controls whether to use atomic GEMM operations
- Supports both collective and point-to-point communication patterns

---

## CommOverlapBase (Derived from CommOverlapCore)

### Purpose
Implements the standard communication-GEMM overlap using collective operations (AllGather, ReduceScatter).

### Key Features
- **Inherits from**: `CommOverlapCore`
- **Communication Method**: Uses collective operations (AllGather, ReduceScatter)
- **Buffer Management**: Single user buffer (`_ubuf`)
- **JAX Usage**: Primary class used for collective-based CommGemm in JAX

### Additional Member Variables
```cpp
int _rs_kernel_type;
bool _rs_overlap_first_gemm;
cudaStream_t _stream_comm;
cudaEvent_t _start_d2dcopy;
```

### Implemented Methods
- ✅ `bulk_overlap()`: Full GEMM + communication overlap
- ✅ `atomic_gemm_overlap_rs()`: Atomic GEMM + ReduceScatter
- ✅ `split_overlap_rs()`: Split GEMM + ReduceScatter
- ✅ `copy_into_buffer()`: Copy data into user buffer

### Not Supported Methods
- ❌ `atomic_gemm_overlap_ag()`: AllGather operations (throws "not supported")
- ❌ `split_overlap_ag()`: AllGather operations (throws "not supported")

### Use Case
Use when you want standard collective-based communication-GEMM overlap. This is the default choice for most JAX CommGemm operations.

### JAX Integration
- Used when `comm_overlap_method != RING_EXCHANGE` in JAX extension
- Supports both `atomic_gemm = true/false` modes
- Handles ReduceScatter operations efficiently

---

## CommOverlapP2PBase (Derived from CommOverlapCore)

### Purpose
Implements communication-GEMM overlap using **Point-to-Point (P2P) communication** instead of collectives.

### Key Features
- **Inherits from**: `CommOverlapCore`
- **Communication Method**: Uses point-to-point communication (ring exchange)
- **Buffer Management**: Multiple user buffers (`_ubufs` vector)
- **JAX Usage**: Used for ring-based communication in JAX

### Additional Member Variables
```cpp
bool _is_reduce_scatter{false};
bool _use_multiatomic_ag{false};
bool _aggregate;
int _next_rank;
int _prev_rank;
int _rank_round_tp;
int _num_ubuf_chunks;
int _self_chunk_id;
std::vector<TensorWrapper> _ubufs;
std::vector<cudaStream_t> _stream_send;
cudaStream_t _stream_recv;
cudaEvent_t _stop_send, _stop_recv;
```

### Implemented Methods
- ✅ `atomic_gemm_overlap_ag()`: Atomic GEMM + AllGather using P2P
- ✅ `split_overlap_ag()`: Split GEMM + AllGather using P2P
- ✅ `atomic_gemm_overlap_rs()`: Atomic GEMM + ReduceScatter using P2P
- ✅ `split_overlap_rs()`: Split GEMM + ReduceScatter using P2P
- ✅ `copy_into_buffer()`: Copy data into user buffers
- ✅ `get_buffer_chunk_by_id()`: Get specific buffer chunk

### Not Supported Methods
- ❌ `bulk_overlap()`: Bulk operations (throws "not supported")

### Use Case
Use when you want ring-based point-to-point communication for potentially better performance or when collectives aren't available.

### JAX Integration
- Used when `comm_overlap_method == RING_EXCHANGE` in JAX extension
- Supports both AllGather and ReduceScatter operations
- Provides better performance for certain communication patterns

---

## Comparison Table

| Aspect | CommOverlapCore | CommOverlapBase | CommOverlapP2PBase |
|--------|----------------|-----------------|-------------------|
| **Communication** | Abstract base | Collective ops | Point-to-Point |
| **Buffer Management** | Single `_ubuf` | Single `_ubuf` | Multiple `_ubufs` |
| **Stream Management** | Basic streams | + comm stream | + send/recv streams |
| **AllGather Support** | Virtual (unimplemented) | ❌ Not supported | ✅ P2P implementation |
| **ReduceScatter Support** | Virtual (unimplemented) | ✅ Collective impl | ✅ P2P implementation |
| **Bulk Overlap** | Virtual (unimplemented) | ✅ Implemented | ❌ Not supported |
| **JAX Usage** | Base class | Collective CommGemm | Ring-based CommGemm |
| **Atomic GEMM** | ✅ Supported | ✅ Supported | ✅ Supported |

## Communication Types

### CommOverlapType Enum
```cpp
enum class CommOverlapType : int64_t { 
    NONE = 0, 
    RS = 1,    // ReduceScatter
    AG = 2     // AllGather
};
```

### CommOverlapMethod Enum
```cpp
enum class CommOverlapMethod : int64_t { 
    NONE = 0, 
    BULK = 1, 
    PIPELINE = 2, 
    RING_EXCHANGE = 3 
};
```

### CommOverlapAlgo Enum
```cpp
enum class CommOverlapAlgo : int64_t {
    NO_OVERLAP = 0,
    BULK_OVERLAP_AG = 1,
    BULK_OVERLAP_RS = 2,
    SPLIT_PIPELINED_AG_P2P = 3,
    SPLIT_PIPELINED_RS = 4,
    SPLIT_PIPELINED_RS_P2P = 5,
    ATOMIC_GEMM_RS = 6,
    ATOMIC_GEMM_AG_P2P = 7,
    ATOMIC_GEMM_RS_P2P = 8
};
```

## Key Differences Summary

1. **CommOverlapCore**: Abstract base class with virtual methods
2. **CommOverlapBase**: Collective-based implementation (AllGather not supported)
3. **CommOverlapP2PBase**: Point-to-point implementation (bulk overlap not supported)

## When to Use Each Class

- **CommOverlapCore**: Never use directly (abstract base class)
- **CommOverlapBase**: Use for standard collective-based communication-GEMM overlap
- **CommOverlapP2PBase**: Use for ring-based point-to-point communication

## JAX Integration Details

### Creation in JAX
```cpp
// In JAX extension (gemm.cpp)
if (method == CommOverlapMethod::RING_EXCHANGE) {
    comm_overlaps[unique_id] = new CommOverlapP2PBase(...);
} else {
    comm_overlaps[unique_id] = new CommOverlapBase(...);
}
```

### Atomic GEMM Support
- Both `CommOverlapBase` and `CommOverlapP2PBase` support atomic GEMM operations
- Controlled by `atomic_gemm` parameter during creation
- Affects the internal implementation of GEMM operations

### Communication Patterns
- **Collective**: Uses NCCL or MPI collectives for communication
- **Ring Exchange**: Uses point-to-point communication in a ring topology
- **Performance**: Ring exchange can provide better performance for certain workloads

## Related Documentation
- **JAX CommGemm Extension**: See `JAX_CommGemm_Extension_Documentation.md`
- **JAX CommGemm Implementations**: See `JAX_CommGemm_Implementations_Documentation.md`

## File Location
`transformer_engine/common/include/transformer_engine/comm_gemm_overlap.h`
