# JAX CommGemm Extension Documentation

## Overview
This document describes the CommGemm-related implementations in the JAX extension of Transformer Engine, located in `transformer_engine/jax/csrc/extensions/gemm.cpp`.

## File Location
`transformer_engine/jax/csrc/extensions/gemm.cpp`

## Key Components

### 1. CommOverlap Buffer Management

#### Global Storage
```cpp
static std::unordered_map<int64_t, CommOverlapCore *> comm_overlaps;
```
- **Purpose**: Global registry to store CommOverlap instances by unique ID
- **Type**: Maps unique 64-bit integer IDs to CommOverlapCore pointers
- **Lifetime**: Persists throughout the program execution

#### CreateCommOverlapBuffer Function
```cpp
int64_t CreateCommOverlapBuffer(CommOverlapType comm_type, CommOverlapMethod method,
                                const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                                int tp_size, int num_splits, int num_max_streams, int comm_cga_size,
                                int gemm_priority, int comm_priority, int num_comm_sm,
                                int set_sm_margin, bool use_ce, bool atomic_gemm,
                                bool rs_overlap_first_gemm, bool aggregate_ag)
```

**Purpose**: Creates and manages CommOverlap buffer instances for communication-GEMM overlap.

**Parameters**:
- `comm_type`: Type of communication (NONE, RS, AG)
- `method`: Communication method (NONE, BULK, PIPELINE, RING_EXCHANGE)
- `buffer_shape`: Shape of the communication buffer
- `buffer_dtype`: Data type of the buffer
- `tp_size`: Tensor parallelism size
- `num_splits`: Number of splits for pipelined operations
- `num_max_streams`: Maximum number of CUDA streams
- `comm_cga_size`: Communication CGA size
- `gemm_priority`: Priority for GEMM operations
- `comm_priority`: Priority for communication operations
- `num_comm_sm`: Number of communication SMs
- `set_sm_margin`: Whether to set SM margin
- `use_ce`: Whether to use collective engine
- `atomic_gemm`: Whether to use atomic GEMM (key parameter for CommGemm)
- `rs_overlap_first_gemm`: Whether to overlap ReduceScatter with first GEMM
- `aggregate_ag`: Whether to aggregate AllGather results

**Return Value**: Unique 64-bit integer ID for the created buffer

**Implementation Logic**:
1. **Hash Generation**: Creates a unique hash based on all parameters
2. **Instance Creation**: 
   - If `method == RING_EXCHANGE`: Creates `CommOverlapP2PBase` instance
   - Otherwise: Creates `CommOverlapBase` instance
3. **Storage**: Stores the instance in the global `comm_overlaps` map

#### DestroyCommOverlapBuffer Function
```cpp
void DestroyCommOverlapBuffer(size_t unique_id)
```
**Purpose**: Destroys a specific CommOverlap buffer instance.

#### DestroyAllCommOverlapBuffers Function
```cpp
void DestroyAllCommOverlapBuffers()
```
**Purpose**: Destroys all CommOverlap buffer instances (cleanup function).

### 2. Buffer Conversion Functions

#### xla_buffer_to_nvte_gemm_operand Function
```cpp
std::tuple<TensorWrapper, std::vector<size_t>> xla_buffer_to_nvte_gemm_operand(
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, Result_Type swizzled_scale_inv,
    JAXX_Scaling_Mode scaling_mode, size_t axis_boundary, bool rowwise)
```

**Purpose**: Converts XLA buffers to NVTETensor operands for GEMM operations.

**Key Features**:
- **Shape Collapse**: Collapses multi-dimensional tensors to 2D for GEMM
- **Scaling Support**: Handles different scaling modes (NO_SCALING, TENSOR_SCALING, MXFP8_SCALING)
- **Swizzling**: Supports MXFP8 scaling factor swizzling
- **Layout Management**: Handles row-wise vs column-wise data layouts

**Scaling Modes**:
- `NO_SCALING`: Standard FP32/FP16 operations
- `DELAYED_TENSOR_SCALING`: Delayed tensor scaling
- `CURRENT_TENSOR_SCALING`: Current tensor scaling
- `MXFP8_1D_SCALING`: MXFP8 block-wise scaling

### 3. GemmFFI Function

#### Function Signature
```cpp
Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Buffer_Type aux_in, Result_Type output, Result_Type bias_grad,
                   Result_Type pre_gelu_out, Result_Type aux_out, Result_Type lhs_swizzle,
                   Result_Type rhs_swizzle, Result_Type workspace, JAXX_Scaling_Mode scaling_mode,
                   CommOverlapMethod comm_overlap_method, CommOverlapType comm_type,
                   int64_t comm_overlap_id, int64_t lhs_axis_boundary, int64_t rhs_axis_boundary,
                   int64_t aux_axis_boundary, bool lhs_transposed, bool rhs_transposed,
                   bool fuse_bias, bool fuse_gelu, bool grad, bool use_split_accumulator)
```

#### Key Features

**1. Operand Processing**:
- Converts XLA buffers to NVTETensor operands using `xla_buffer_to_nvte_gemm_operand`
- Handles scaling factors for quantized operations
- Supports swizzling for MXFP8 scaling
- Manages row-wise vs column-wise data layouts

**2. Communication Overlap Support**:
- **No Overlap** (`comm_type == NONE`): Standard cuBLAS GEMM
- **Bulk Overlap**: Full GEMM + communication overlap
- **Split Overlap**: Pipelined GEMM + communication

**3. Communication Types**:
- **ReduceScatter (RS)**: Reduces and scatters results across ranks
- **AllGather (AG)**: Gathers data from all ranks

#### Communication Overlap Modes

##### A. No Communication Overlap
```cpp
if (comm_type == CommOverlapType::NONE) {
    nvte_cublas_gemm(rhs_.data(), lhs_.data(), out_.data(), bias_.data(), pre_gelu_.data(),
                     rhs_transposed, lhs_transposed, grad, workspace_.data(), false,
                     use_split_accumulator, num_math_sm, stream);
}
```

##### B. Bulk Communication Overlap
```cpp
if (comm_overlap_method == CommOverlapMethod::BULK) {
    // Prepare auxiliary output tensor
    // Copy auxiliary data into communication buffer
    // Launch GEMM with bulk overlap
    executor->bulk_overlap(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                          workspace_, grad, false, use_split_accumulator, comm_type, aux_out_,
                          stream);
}
```

##### C. Split Communication Overlap

**ReduceScatter Mode**:
```cpp
if (comm_type == CommOverlapType::RS) {
    // Prepare reduce-scattered GEMM output
    // Launch GEMM+RS
    executor->split_overlap_rs(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                              workspace_, grad, false, use_split_accumulator, rs_out_, stream);
}
```

**AllGather Mode**:
```cpp
if (comm_type == CommOverlapType::AG) {
    // Prepare auxiliary buffer for all-gathered LHS
    // Copy distributed LHS operand into communication buffer
    // Launch AG+GEMM
    executor->split_overlap_ag(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                              workspace_, grad, false, use_split_accumulator, aux_out_, stream);
}
```

### 3. XLA FFI Handler Registration

#### GemmHandler
```cpp
XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // gelu_input
                                  .Arg<Buffer_Type>()      // aux_in
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // aux_out
                                  .Ret<Buffer_Type>()      // lhs_swizzled
                                  .Ret<Buffer_Type>()      // rhs_swizzled
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<CommOverlapMethod>("comm_overlap_method")
                                  .Attr<CommOverlapType>("comm_type")
                                  .Attr<int64_t>("comm_overlap_id")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<int64_t>("aux_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("use_split_accumulator"),
                              FFI_CudaGraph_Traits);
```

## Key Implementation Details

### 1. Buffer Management
- **User Buffer**: Managed by CommOverlap instances
- **Workspace**: 256-byte aligned for cuBLAS requirements
- **Swizzling**: Handles MXFP8 scaling factor swizzling
- **Memory Alignment**: Proper alignment for optimal performance

### 2. Communication Integration
- **Buffer Sharing**: GEMM output can be redirected to communication buffer
- **Stream Management**: Separate streams for compute and communication
- **Event Synchronization**: CUDA events for timing and synchronization

### 3. Scaling Support
- **No Scaling**: Standard FP32/FP16 operations
- **Tensor Scaling**: Delayed/current tensor scaling
- **MXFP8 Scaling**: Block-wise scaling with swizzling

### 4. Performance Optimizations
- **Split Accumulator**: Optional split accumulator for better numerical stability
- **Stream Overlap**: Compute and communication can overlap
- **Memory Alignment**: Proper memory alignment for optimal performance

## Usage Patterns

### 1. Standard GEMM (No Communication)
```cpp
// Set comm_type = CommOverlapType::NONE
// Standard cuBLAS GEMM execution
```

### 2. Bulk Communication Overlap
```cpp
// Set comm_overlap_method = CommOverlapMethod::BULK
// Full GEMM + communication overlap
```

### 3. Pipelined Communication Overlap
```cpp
// Set comm_overlap_method = CommOverlapMethod::PIPELINE
// Split GEMM + communication overlap
```

### 4. Ring Exchange Communication
```cpp
// Set comm_overlap_method = CommOverlapMethod::RING_EXCHANGE
// Point-to-point communication with ring topology
```

## Error Handling
- **Buffer Size Validation**: Checks for correct buffer sizes
- **Type Compatibility**: Validates data type compatibility
- **Communication Validation**: Ensures proper communication setup
- **CUDA Error Checking**: Wraps operations with CUDA error checking

## Performance Considerations
1. **Memory Layout**: Row-wise vs column-wise data layout optimization
2. **Stream Management**: Proper stream allocation and synchronization
3. **Buffer Alignment**: 256-byte alignment for optimal performance
4. **Communication Overlap**: Maximizing overlap between compute and communication
5. **Scaling Factor Handling**: Efficient handling of quantization scaling factors
6. **Multi-stream Execution**: Parallel execution for grouped operations

## Related Documentation
- **CommOverlap Classes**: See `CommOverlap_Classes_Documentation.md`
- **JAX CommGemm Implementations**: See `JAX_CommGemm_Implementations_Documentation.md`
