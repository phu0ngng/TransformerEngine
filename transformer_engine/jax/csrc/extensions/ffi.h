/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include <xla/ffi/api/ffi.h>

#include <numeric>

#include "common/util/logging.h"
#include "misc.h"

namespace transformer_engine {
namespace jax {

using Buffer_Type = xla::ffi::AnyBuffer;
using Result_Type = xla::ffi::Result<xla::ffi::AnyBuffer>;
using Error_Type = xla::ffi::Error;
using FFI = xla::ffi::Ffi;
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>;
constexpr auto FFI_CudaGraph_Traits = {xla::ffi::Traits::kCmdBufferCompatible};

DType convert_ffi_datatype_to_te_dtype(const xla::ffi::DataType &type);

Error_Type ffi_with_cuda_error_check();

}  // namespace jax
}  // namespace transformer_engine

// This registration need to stay in the global namespace
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    transformer_engine::jax::Attn_Attr_Type, xla::ffi::StructMember<int64_t>("input_batch"),
    xla::ffi::StructMember<int64_t>("bias_batch"), xla::ffi::StructMember<int64_t>("q_max_seqlen"),
    xla::ffi::StructMember<int64_t>("kv_max_seqlen"), xla::ffi::StructMember<int64_t>("attn_heads"),
    xla::ffi::StructMember<int64_t>("num_gqa_groups"),
    xla::ffi::StructMember<int64_t>("bias_heads"), xla::ffi::StructMember<int64_t>("head_dim"),
    xla::ffi::StructMember<int64_t>("max_segments_per_seq"),
    xla::ffi::StructMember<int64_t>("window_size_left"),
    xla::ffi::StructMember<int64_t>("window_size_right"),
    xla::ffi::StructMember<int64_t>("bias_type"), xla::ffi::StructMember<int64_t>("mask_type"),
    xla::ffi::StructMember<int64_t>("qkv_layout"), xla::ffi::StructMember<double>("scaling_factor"),
    xla::ffi::StructMember<double>("dropout_probability"),
    xla::ffi::StructMember<bool>("is_training"), xla::ffi::StructMember<bool>("deterministic"));
