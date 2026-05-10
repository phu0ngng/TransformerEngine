/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <cstdint>

#include "../extensions.h"
#include "common.h"

namespace transformer_engine {
namespace jax {

// ── Config structs ────────────────────────────────────────────────────────────

// EpPrepare has no per-call config — token_counts size and handle_mem size are
// derived from the cached group config inside the C++ backend.

struct EpDispatchConfig {
  int64_t recv_capacity;  // recv_tokens first dim (= recv_topk_weights size)
  int64_t top_k;          // routing top_k (recv_topk_weights is 1D under HT+EM)
};

struct EpCombineConfig {
  int64_t num_local_tokens;  // T — size of result first dim
};

struct EpDispatchBwdConfig {
  int64_t num_local_tokens;  // T — size of grad_tokens first dim
};

struct EpCombineBwdConfig {
  int64_t recv_capacity;  // size of grad_expert_out first dim (from forward expert_out shape)
};

// ── Bootstrap helpers (called once, exposed via pybind11) ─────────────────────

void EpInitialize(pybind11::bytes unique_id_bytes_obj, int world_size, int rank,
                  int ep_size, int num_experts, int max_tokens_per_rank,
                  int max_recv_tokens_per_rank, int hidden_dim) {
  std::string uid_str = unique_id_bytes_obj;
  NVTE_CHECK(static_cast<int>(uid_str.size()) >= 128,
             "unique_id_bytes must be at least 128 bytes (ncclUniqueId size).");
  NVTEEpGroupConfig cfg{.ep_size = ep_size,
                        .num_experts = num_experts,
                        .max_tokens_per_rank = max_tokens_per_rank,
                        .max_recv_tokens_per_rank = max_recv_tokens_per_rank,
                        .hidden_dim = hidden_dim};
  nvte_ep_initialize(reinterpret_cast<const uint8_t*>(uid_str.data()), world_size, rank, cfg);
}

size_t EpGetHandleMemSize(int top_k) {
  // num_local_experts is no longer used by the backend; pass 0 as a placeholder.
  NVTEEpLayerConfig layer_cfg{0, top_k};
  return nvte_ep_get_handle_mem_size(layer_cfg);
}

// ── ep_prepare ────────────────────────────────────────────────────────────────
// Inputs:  topk_idx [T, top_k] int64
// Outputs: token_counts [num_local_experts] int32
//          handle_mem [handle_mem_size] uint8

Error_Type EpPrepareFFI(cudaStream_t stream,
                        Buffer_Type topk_idx,
                        Result_Type token_counts,
                        Result_Type handle_mem) {
  auto topk_dims = topk_idx.dimensions();
  NVTE_CHECK(topk_dims.size() == 2, "topk_idx must be 2D [T, top_k], got ndim=",
             topk_dims.size());

  std::vector<size_t> topk_shape = {static_cast<size_t>(topk_dims[0]),
                                    static_cast<size_t>(topk_dims[1])};
  auto topk_idx_ =
      TensorWrapper(topk_idx.untyped_data(), topk_shape, DType::kInt64);

  std::vector<size_t> tc_shape = {static_cast<size_t>(token_counts->element_count())};
  auto token_counts_ =
      TensorWrapper(token_counts->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ =
      TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  nvte_ep_prepare(topk_idx_.data(), token_counts_.data(), handle_mem_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EpPrepareHandler, EpPrepareFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()   // stream
        .Arg<Buffer_Type>()       // topk_idx
        .Ret<Buffer_Type>()       // token_counts (shape from JAX abstract())
        .Ret<Buffer_Type>(),      // handle_mem    (size queried in JAX abstract())
    FFI_CudaGraph_Traits);

// ── ep_dispatch ───────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          tokens [T, H]
//          topk_weights [T, top_k] float32
// Outputs: recv_tokens [recv_capacity, H]
//          recv_topk_weights [recv_capacity] float32  (HT+EM: 1 weight per slot)

Error_Type EpDispatchFFI(cudaStream_t stream,
                         Buffer_Type handle_mem,
                         Buffer_Type topk_idx,
                         Buffer_Type tokens,
                         Buffer_Type topk_weights,
                         Result_Type recv_tokens,
                         Result_Type recv_topk_weights,
                         EpDispatchConfig config) {
  auto token_dims = tokens.dimensions();
  NVTE_CHECK(token_dims.size() == 2, "tokens must be 2D [T, H], got ndim=",
             token_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  auto idx_dims = topk_idx.dimensions();
  NVTE_CHECK(idx_dims.size() == 2, "topk_idx must be 2D [T, top_k], got ndim=",
             idx_dims.size());
  std::vector<size_t> idx_shape = {static_cast<size_t>(idx_dims[0]),
                                   static_cast<size_t>(idx_dims[1])};
  auto topk_idx_ = TensorWrapper(topk_idx.untyped_data(), idx_shape, DType::kInt64);

  std::vector<size_t> tok_shape = {static_cast<size_t>(token_dims[0]),
                                   static_cast<size_t>(token_dims[1])};
  auto token_dtype = convert_ffi_datatype_to_te_dtype(tokens.element_type());
  auto tokens_ = TensorWrapper(tokens.untyped_data(), tok_shape, token_dtype);

  // topk_weights may be empty (null tensor passed for dense routing)
  void* topk_w_ptr = topk_weights.element_count() > 0 ? topk_weights.untyped_data() : nullptr;
  std::vector<size_t> tw_shape = {topk_weights.element_count() > 0
                                      ? static_cast<size_t>(topk_weights.dimensions()[0])
                                      : 0,
                                  topk_weights.element_count() > 0
                                      ? static_cast<size_t>(topk_weights.dimensions()[1])
                                      : 0};
  auto topk_weights_ = TensorWrapper(topk_w_ptr, tw_shape, DType::kFloat32);

  std::vector<size_t> recv_shape = {static_cast<size_t>(config.recv_capacity),
                                    static_cast<size_t>(token_dims[1])};
  auto recv_tokens_ =
      TensorWrapper(recv_tokens->untyped_data(), recv_shape, token_dtype);

  std::vector<size_t> recv_w_shape = {static_cast<size_t>(config.recv_capacity)};
  auto recv_topk_weights_ =
      TensorWrapper(recv_topk_weights->untyped_data(), recv_w_shape, DType::kFloat32);

  nvte_ep_dispatch(handle_mem_.data(), topk_idx_.data(), tokens_.data(), topk_weights_.data(),
                   recv_tokens_.data(), recv_topk_weights_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EpDispatchHandler, EpDispatchFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()  // stream
        .Arg<Buffer_Type>()      // handle_mem
        .Arg<Buffer_Type>()      // topk_idx
        .Arg<Buffer_Type>()      // tokens
        .Arg<Buffer_Type>()      // topk_weights
        .Ret<Buffer_Type>()      // recv_tokens
        .Ret<Buffer_Type>()      // recv_topk_weights
        .Attrs<EpDispatchConfig>(),
    FFI_CudaGraph_Traits);

// ── ep_combine ────────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          expert_out [recv_capacity, H]
// Outputs: result [num_local_tokens, H]

Error_Type EpCombineFFI(cudaStream_t stream,
                        Buffer_Type handle_mem,
                        Buffer_Type expert_out,
                        Result_Type result,
                        EpCombineConfig config) {
  auto eo_dims = expert_out.dimensions();
  NVTE_CHECK(eo_dims.size() == 2, "expert_out must be 2D, got ndim=", eo_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  std::vector<size_t> eo_shape = {static_cast<size_t>(eo_dims[0]),
                                  static_cast<size_t>(eo_dims[1])};
  auto eo_dtype = convert_ffi_datatype_to_te_dtype(expert_out.element_type());
  auto expert_out_ = TensorWrapper(expert_out.untyped_data(), eo_shape, eo_dtype);

  std::vector<size_t> res_shape = {static_cast<size_t>(config.num_local_tokens),
                                   static_cast<size_t>(eo_dims[1])};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  nvte_ep_combine(handle_mem_.data(), expert_out_.data(), result_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EpCombineHandler, EpCombineFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()  // stream
        .Arg<Buffer_Type>()      // handle_mem
        .Arg<Buffer_Type>()      // expert_out
        .Ret<Buffer_Type>()      // result
        .Attrs<EpCombineConfig>(),
    FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          grad [recv_capacity, H]  (grad w.r.t. recv_tokens)
// Outputs: grad_tokens [num_local_tokens, H]

Error_Type EpDispatchBwdFFI(cudaStream_t stream,
                             Buffer_Type handle_mem,
                             Buffer_Type grad,
                             Result_Type grad_tokens,
                             EpDispatchBwdConfig config) {
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() == 2, "grad must be 2D, got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  std::vector<size_t> g_shape = {static_cast<size_t>(grad_dims[0]),
                                 static_cast<size_t>(grad_dims[1])};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  std::vector<size_t> out_shape = {static_cast<size_t>(config.num_local_tokens),
                                   static_cast<size_t>(grad_dims[1])};
  auto grad_tokens_ = TensorWrapper(grad_tokens->untyped_data(), out_shape, g_dtype);

  nvte_ep_dispatch_bwd(handle_mem_.data(), grad_.data(), grad_tokens_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EpDispatchBwdHandler, EpDispatchBwdFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()  // stream
        .Arg<Buffer_Type>()      // handle_mem
        .Arg<Buffer_Type>()      // grad (w.r.t. recv_tokens)
        .Ret<Buffer_Type>()      // grad_tokens
        .Attrs<EpDispatchBwdConfig>(),
    FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          grad [num_local_tokens, H]  (grad w.r.t. result)
// Outputs: grad_expert_out [recv_capacity, H]

Error_Type EpCombineBwdFFI(cudaStream_t stream,
                            Buffer_Type handle_mem,
                            Buffer_Type grad,
                            Result_Type grad_expert_out,
                            EpCombineBwdConfig config) {
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() == 2, "grad must be 2D, got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  std::vector<size_t> g_shape = {static_cast<size_t>(grad_dims[0]),
                                 static_cast<size_t>(grad_dims[1])};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  std::vector<size_t> out_shape = {static_cast<size_t>(config.recv_capacity),
                                   static_cast<size_t>(grad_dims[1])};
  auto grad_expert_out_ =
      TensorWrapper(grad_expert_out->untyped_data(), out_shape, g_dtype);

  nvte_ep_combine_bwd(handle_mem_.data(), grad_.data(), grad_expert_out_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EpCombineBwdHandler, EpCombineBwdFFI,
    FFI::Bind()
        .Ctx<FFI_Stream_Type>()  // stream
        .Arg<Buffer_Type>()      // handle_mem
        .Arg<Buffer_Type>()      // grad (w.r.t. result)
        .Ret<Buffer_Type>()      // grad_expert_out
        .Attrs<EpCombineBwdConfig>(),
    FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpDispatchConfig,
                                       ::xla::ffi::StructMember<int64_t>("recv_capacity"),
                                       ::xla::ffi::StructMember<int64_t>("top_k"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpCombineConfig,
                                       ::xla::ffi::StructMember<int64_t>("num_local_tokens"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpDispatchBwdConfig,
                                       ::xla::ffi::StructMember<int64_t>("num_local_tokens"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpCombineBwdConfig,
                                       ::xla::ffi::StructMember<int64_t>("recv_capacity"));

#endif  // NVTE_WITH_NCCL_EP
