# JAX CommGemm Implementations Documentation

## Overview
This document describes the specific CommGemm (Communication + GEMM) implementations in the JAX extension of Transformer Engine, located in `transformer_engine/jax/csrc/extensions/gemm.cpp`.

## File Location
`transformer_engine/jax/csrc/extensions/gemm.cpp`

## CommGemm Implementation Overview

The JAX extension implements three main types of CommGemm operations:

1. **Bulk CommGemm Overlap**: Full GEMM + communication overlap
2. **Split CommGemm with ReduceScatter**: Pipelined GEMM + ReduceScatter
3. **Split CommGemm with AllGather**: Pipelined GEMM + AllGather

## 1. Bulk CommGemm Overlap Implementation

### Function Call
```cpp
executor->bulk_overlap(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                      workspace_, grad, false, use_split_accumulator, comm_type, aux_out_,
                      stream);
```

### Implementation Details

#### A. Auxiliary Output Preparation
```cpp
// Prepare the auxiliary output tensor
auto aux_out_dims = aux_out->dimensions();
std::vector<size_t> aux_out_shape = {0};
auto aux_out_dtype = convert_ffi_datatype_to_te_dtype(aux_out->element_type());
if ((comm_type == CommOverlapType::AG && aux_out->element_count() > 0) ||
    comm_type == CommOverlapType::RS) {
    std::vector<size_t> aux_out_shape = {
        product(aux_out_dims, 0, aux_axis_boundary),
        product(aux_out_dims, aux_axis_boundary, aux_out_dims.size())};
}
auto aux_out_ = TensorWrapper(aux_out->untyped_data(), aux_out_shape, aux_out_dtype);
```

#### B. Auxiliary Input Processing
```cpp
// Copy the auxiliary data into the communications buffer
auto aux_in_dims = aux_in.dimensions();
std::vector<size_t> aux_in_shape = {
    product(aux_in_dims, 0, aux_axis_boundary),
    product(aux_in_dims, aux_axis_boundary, aux_in_dims.size())};
auto aux_in_dtype = convert_ffi_datatype_to_te_dtype(aux_in.element_type());
auto aux_in_ = TensorWrapper(aux_in.untyped_data(), aux_in_shape, aux_in_dtype);
```

#### C. Size Validation
```cpp
if (comm_type == CommOverlapType::AG && aux_out->element_count() > 0) {
    NVTE_CHECK(aux_in_shape[0] == tp_size * aux_out_shape[0],
               "cuBLAS GEMM w/ bulk AG overlap auxiliary output is sized incorrectly, ",
               "expected (", aux_in_shape[0] / tp_size, ",", aux_in_shape[1], ") but got ",
               to_string_like(aux_out_dims));
} else if (comm_type == CommOverlapType::RS) {
    NVTE_CHECK(tp_size * aux_in_shape[0] == aux_out_shape[0],
               "cuBLAS GEMM w/ bulk RS overlap auxiliary output is sized incorrectly, ",
               "expected (", aux_in_shape[0] * tp_size, ",", aux_in_shape[1], ") but got ",
               to_string_like(aux_out_dims));
}
```

#### D. Buffer Copy and Execution
```cpp
// Copy auxiliary data into communication buffer
executor->copy_into_buffer(stream, aux_in_, (comm_type == CommOverlapType::AG));

// Launch GEMM with bulk overlap
executor->bulk_overlap(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                      workspace_, grad, false, use_split_accumulator, comm_type, aux_out_,
                      stream);
```

## 2. Split CommGemm with ReduceScatter Implementation

### Function Call
```cpp
executor->split_overlap_rs(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                          workspace_, grad, false, use_split_accumulator, rs_out_, stream);
```

### Implementation Details

#### A. ReduceScatter Output Preparation
```cpp
// Prepare the auxiliary buffer for the reduce-scattered GEMM output
auto rs_out_shape = std::vector<size_t>(out_shape);
rs_out_shape.at(0) /= tp_size;  // Reduce the first dimension by tensor parallelism size
auto rs_out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
auto rs_out_ = TensorWrapper(output->untyped_data(), rs_out_shape, rs_out_dtype);
```

#### B. Size Validation
```cpp
NVTE_CHECK(rs_out_.numel() == output->element_count(),
           "cuBLAS GEMM->RS overlap output buffer is sized incorrectly, expected ",
           rs_out_.numel(), " elements ", to_string_like(rs_out_shape), " but got ",
           output->element_count(), " elements ", to_string_like(output->dimensions()));
```

#### C. Execution
```cpp
// Launch GEMM+RS
executor->split_overlap_rs(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                          workspace_, grad, false, use_split_accumulator, rs_out_, stream);
```

## 3. Split CommGemm with AllGather Implementation

### Function Call
```cpp
executor->split_overlap_ag(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                          workspace_, grad, false, use_split_accumulator, aux_out_, stream);
```

### Implementation Details

#### A. AllGather Output Preparation
```cpp
// Prepare the auxiliary buffer for all-gathered LHS
std::vector<size_t> aux_out_shape = {0};
auto aux_out_dtype = convert_ffi_datatype_to_te_dtype(aux_out->element_type());
if (aux_out->element_count() > 0) {
    aux_out_shape = std::vector<size_t>(lhs_shape);
    aux_out_shape.at(0) *= tp_size;  // Expand the first dimension by tensor parallelism size
    auto aux_out_numel = aux_out_shape[0] * aux_out_shape[1];
    NVTE_CHECK(aux_out_numel == aux_out->element_count(),
               "cuBLAS AG->GEMM overlap auxiliary buffer is sized incorrectly, expected ",
               aux_out_numel, " elements ", to_string_like(aux_out_shape), " but got ",
               aux_out->element_count(), " elements ", to_string_like(aux_out->dimensions()));
}
auto aux_out_ = TensorWrapper(aux_out->untyped_data(), aux_out_shape, aux_out_dtype);
```

#### B. LHS Buffer Copy
```cpp
// Copy the distributed LHS operand into the local chunk of the communication buffer
executor->copy_into_buffer(stream, lhs_, true, make_lhs_rowwise);
```

#### C. Execution
```cpp
// Launch AG+GEMM
executor->split_overlap_ag(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                          workspace_, grad, false, use_split_accumulator, aux_out_, stream);
```

## 4. Atomic GEMM Support

### Overview
All CommGemm implementations support atomic GEMM operations, controlled by the `atomic_gemm` parameter during CommOverlap buffer creation.

### Atomic GEMM Usage
```cpp
// In CreateCommOverlapBuffer function
if (method == CommOverlapMethod::RING_EXCHANGE) {
    comm_overlaps[unique_id] = new CommOverlapP2PBase(..., atomic_gemm, ...);
} else {
    comm_overlaps[unique_id] = new CommOverlapBase(..., atomic_gemm, ...);
}
```

### Atomic GEMM Methods
- **CommOverlapBase**: `atomic_gemm_overlap_rs()` for ReduceScatter operations
- **CommOverlapP2PBase**: `atomic_gemm_overlap_rs()` and `atomic_gemm_overlap_ag()` for both operations

### Benefits of Atomic GEMM
- **Better Performance**: Can provide better performance for certain workloads
- **Memory Efficiency**: More efficient memory usage patterns
- **Numerical Stability**: Can improve numerical stability in some cases


## Key Implementation Features

### 1. Buffer Management
- **Output Redirection**: For ReduceScatter operations, GEMM output goes into communication buffer
- **Auxiliary Buffer Handling**: Proper management of auxiliary input/output buffers
- **Memory Alignment**: 256-byte alignment for optimal performance

### 2. Shape Transformations
- **ReduceScatter**: Reduces first dimension by `tp_size`
- **AllGather**: Expands first dimension by `tp_size`
- **Axis Boundary Handling**: Proper handling of axis boundaries for tensor operations

### 3. Validation and Error Checking
- **Size Validation**: Comprehensive size checks for all buffers
- **Type Compatibility**: Ensures data type compatibility across operations
- **Error Messages**: Detailed error messages for debugging

### 4. Communication Integration
- **Buffer Sharing**: Efficient sharing of buffers between GEMM and communication
- **Stream Management**: Proper CUDA stream management for overlap
- **Synchronization**: Event-based synchronization between operations

### 5. Atomic GEMM Support
- **Performance Optimization**: Atomic GEMM can provide better performance
- **Memory Efficiency**: More efficient memory usage patterns
- **Numerical Stability**: Can improve numerical stability

## Parameter Details

### Common Parameters for All CommGemm Operations
- `rhs_`, `lhs_`: Input tensors (right-hand side and left-hand side)
- `rhs_transposed`, `lhs_transposed`: Transpose flags for input tensors
- `out_`: Output tensor
- `bias_`: Bias tensor (optional)
- `pre_gelu_`: Pre-GELU output tensor (optional)
- `workspace_`: Workspace buffer for cuBLAS operations
- `grad`: Gradient computation flag
- `use_split_accumulator`: Split accumulator flag for numerical stability
- `stream`: CUDA stream for execution

### Communication-Specific Parameters
- `comm_type`: Type of communication (RS or AG)
- `aux_out_`: Auxiliary output buffer for communication results
- `tp_size`: Tensor parallelism size
- `aux_axis_boundary`: Axis boundary for auxiliary tensor operations

### Atomic GEMM Parameters
- `atomic_gemm`: Whether to use atomic GEMM operations
- `executor`: CommOverlap instance (CommOverlapBase or CommOverlapP2PBase)

## Performance Considerations

### 1. Memory Layout Optimization
- **Row-wise vs Column-wise**: Proper data layout for optimal performance
- **Buffer Alignment**: 256-byte alignment for optimal performance
- **Memory Coalescing**: Efficient memory access patterns

### 2. Communication Overlap
- **Stream Overlap**: Maximizing overlap between compute and communication
- **Buffer Sharing**: Minimizing memory overhead through buffer sharing
- **Pipelining**: Efficient pipelining of operations

### 3. Numerical Stability
- **Split Accumulator**: Optional split accumulator for better numerical stability
- **Scaling Factors**: Proper handling of quantization scaling factors
- **Precision Management**: Appropriate precision for different operations

### 4. Atomic GEMM Optimization
- **Performance Tuning**: Atomic GEMM can provide better performance for certain workloads
- **Memory Patterns**: More efficient memory access patterns
- **Stream Utilization**: Better utilization of CUDA streams

## Error Handling

### 1. Size Validation
```cpp
NVTE_CHECK(condition, "Error message with details");
```

### 2. Type Compatibility
- Ensures data types are compatible across operations
- Validates scaling factor types for quantized operations

### 3. Communication Validation
- Validates communication buffer sizes
- Ensures proper tensor parallelism setup

### 4. Atomic GEMM Validation
- Validates atomic GEMM parameter compatibility
- Ensures proper CommOverlap instance selection

## Usage Examples

### 1. Bulk CommGemm with AllGather
```cpp
// Set comm_type = CommOverlapType::AG
// Set comm_overlap_method = CommOverlapMethod::BULK
// Set atomic_gemm = true/false based on performance requirements
// Prepare aux_in and aux_out buffers
// Call bulk_overlap
```

### 2. Split CommGemm with ReduceScatter
```cpp
// Set comm_type = CommOverlapType::RS
// Set comm_overlap_method = CommOverlapMethod::PIPELINE
// Set atomic_gemm = true/false based on performance requirements
// Prepare output buffer with reduced size
// Call split_overlap_rs or atomic_gemm_overlap_rs
```

### 3. Split CommGemm with AllGather
```cpp
// Set comm_type = CommOverlapType::AG
// Set comm_overlap_method = CommOverlapMethod::PIPELINE
// Set atomic_gemm = true/false based on performance requirements
// Prepare aux_out buffer with expanded size
// Copy LHS into communication buffer
// Call split_overlap_ag
```

## Integration with XLA FFI

The CommGemm implementations are integrated into the XLA FFI system through the `GemmHandler`:

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

This integration allows JAX to call these CommGemm operations seamlessly while maintaining the performance benefits of communication overlap.

## Related Documentation
- **CommOverlap Classes**: See `CommOverlap_Classes_Documentation.md`
- **JAX CommGemm Extension**: See `JAX_CommGemm_Extension_Documentation.md`
