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
#include "transformer_engine/gemm.h"

namespace transformer_engine {
namespace jax {

// ── Config structs ────────────────────────────────────────────────────────────

// EpPrepare carries the per-handle dispatch_output_per_expert_alignment knob;
// token_counts size and handle_mem size still come from the cached group config.

struct EpPrepareConfig {
  int64_t dispatch_output_per_expert_alignment;
};

struct EpDispatchConfig {
  int64_t recv_capacity;  // recv_tokens first dim (= recv_topk_weights size)
  int64_t top_k;          // routing top_k (recv_topk_weights is 1D)
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

void EpInitialize(pybind11::bytes unique_id_bytes_obj, int world_size, int rank, int ep_size,
                  int num_experts, int max_tokens_per_rank, int max_recv_tokens_per_rank,
                  int hidden_dim) {
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

size_t EpGetHandleMemSize(int top_k, size_t dispatch_output_per_expert_alignment) {
  // num_local_experts is no longer used by the backend; pass 0 as a placeholder.
  NVTEEpLayerConfig layer_cfg{0, top_k, dispatch_output_per_expert_alignment};
  return nvte_ep_get_handle_mem_size(layer_cfg);
}

// ── ep_prepare ────────────────────────────────────────────────────────────────
// Inputs:  topk_idx [..., top_k] int32 or int64 (N-D leading dims flattened internally;
//          int32 is upcast to int64 on-stream via a scratch workspace).
// Outputs: token_counts [num_local_experts] int32
//          handle_mem [handle_mem_size] uint8

Error_Type EpPrepareFFI(cudaStream_t stream, Buffer_Type topk_idx, Result_Type token_counts,
                        Result_Type handle_mem, EpPrepareConfig config) {
  auto topk_dims = topk_idx.dimensions();
  NVTE_CHECK(topk_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", topk_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));

  // Flatten leading dims; keep last dim as top_k.
  std::vector<size_t> topk_shape = {product(topk_dims, 0, topk_dims.size() - 1),
                                    static_cast<size_t>(topk_dims.back())};
  // Upcast int32 → int64 on-stream into a scratch buffer when needed.
  int64_t* topk_idx_i64_workspace = nullptr;
  void* topk_idx_data = topk_idx.untyped_data();
  if (idx_etype == ::xla::ffi::DataType::S32) {
    const size_t n = topk_shape[0] * topk_shape[1];
    NVTE_CHECK_CUDA(cudaMallocAsync(&topk_idx_i64_workspace, n * sizeof(int64_t), stream));
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t*>(topk_idx_data),
                                topk_idx_i64_workspace, n, stream);
    topk_idx_data = topk_idx_i64_workspace;
  }
  auto topk_idx_ = TensorWrapper(topk_idx_data, topk_shape, DType::kInt64);

  std::vector<size_t> tc_shape = {static_cast<size_t>(token_counts->element_count())};
  auto token_counts_ = TensorWrapper(token_counts->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  nvte_ep_prepare(topk_idx_.data(), token_counts_.data(), handle_mem_.data(),
                  static_cast<size_t>(config.dispatch_output_per_expert_alignment), stream);

  if (topk_idx_i64_workspace != nullptr) {
    NVTE_CHECK_CUDA(cudaFreeAsync(topk_idx_i64_workspace, stream));
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpPrepareHandler, EpPrepareFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // topk_idx
                                  .Ret<Buffer_Type>()      // token_counts
                                  .Ret<Buffer_Type>()      // handle_mem
                                  .Attrs<EpPrepareConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch ───────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          tokens [..., H]            (N-D, flattened internally)
//          topk_idx [..., top_k]      (N-D, flattened internally)
//          topk_weights [..., top_k] float32 (N-D, flattened internally; required)
// Outputs: recv_tokens [recv_capacity, H]      (always 2D)
//          recv_topk_weights [recv_capacity] f32 (always 1D, 1 weight per slot)

Error_Type EpDispatchFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type topk_idx,
                         Buffer_Type tokens, Buffer_Type topk_weights, Result_Type recv_tokens,
                         Result_Type recv_topk_weights, EpDispatchConfig config) {
  auto token_dims = tokens.dimensions();
  NVTE_CHECK(token_dims.size() >= 2,
             "tokens must be at least 2D [..., H], got ndim=", token_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  auto idx_dims = topk_idx.dimensions();
  NVTE_CHECK(idx_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", idx_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));
  NVTE_CHECK(static_cast<int64_t>(idx_dims.back()) == config.top_k, "top_k attr (", config.top_k,
             ") must match topk_idx last dim (", idx_dims.back(), ")");
  std::vector<size_t> idx_shape = {product(idx_dims, 0, idx_dims.size() - 1),
                                   static_cast<size_t>(idx_dims.back())};
  // Upcast int32 → int64 on-stream into a scratch buffer when needed.
  int64_t* topk_idx_i64_workspace = nullptr;
  void* topk_idx_data = topk_idx.untyped_data();
  if (idx_etype == ::xla::ffi::DataType::S32) {
    const size_t n = idx_shape[0] * idx_shape[1];
    NVTE_CHECK_CUDA(cudaMallocAsync(&topk_idx_i64_workspace, n * sizeof(int64_t), stream));
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t*>(topk_idx_data),
                                topk_idx_i64_workspace, n, stream);
    topk_idx_data = topk_idx_i64_workspace;
  }
  auto topk_idx_ = TensorWrapper(topk_idx_data, idx_shape, DType::kInt64);

  const size_t T_flat = product(token_dims, 0, token_dims.size() - 1);
  const size_t H = static_cast<size_t>(token_dims.back());
  std::vector<size_t> tok_shape = {T_flat, H};
  auto token_dtype = convert_ffi_datatype_to_te_dtype(tokens.element_type());
  auto tokens_ = TensorWrapper(tokens.untyped_data(), tok_shape, token_dtype);

  // dispatch FWD always carries per-token routing weights; null topk_weights
  // is reserved for the BWD path (which reuses the C++ dispatch entry through
  // a different code path).
  auto tw_dims = topk_weights.dimensions();
  NVTE_CHECK(tw_dims.size() >= 2,
             "topk_weights must be at least 2D [..., top_k], got ndim=", tw_dims.size());
  std::vector<size_t> tw_shape = {product(tw_dims, 0, tw_dims.size() - 1),
                                  static_cast<size_t>(tw_dims.back())};
  auto topk_weights_ = TensorWrapper(topk_weights.untyped_data(), tw_shape, DType::kFloat32);

  auto recv_dims = recv_tokens->dimensions();
  NVTE_CHECK(recv_dims.size() == 2,
             "recv_tokens must be 2D [recv_capacity, H], got ndim=", recv_dims.size());
  std::vector<size_t> recv_shape = {static_cast<size_t>(config.recv_capacity), H};
  auto recv_tokens_ = TensorWrapper(recv_tokens->untyped_data(), recv_shape, token_dtype);

  auto recv_w_dims = recv_topk_weights->dimensions();
  NVTE_CHECK(recv_w_dims.size() == 1,
             "recv_topk_weights must be 1D [recv_capacity], got ndim=", recv_w_dims.size());
  std::vector<size_t> recv_w_shape = {static_cast<size_t>(config.recv_capacity)};
  auto recv_topk_weights_ =
      TensorWrapper(recv_topk_weights->untyped_data(), recv_w_shape, DType::kFloat32);

  nvte_ep_dispatch(handle_mem_.data(), topk_idx_.data(), tokens_.data(), topk_weights_.data(),
                   recv_tokens_.data(), recv_topk_weights_.data(), stream);

  if (topk_idx_i64_workspace != nullptr) {
    NVTE_CHECK_CUDA(cudaFreeAsync(topk_idx_i64_workspace, stream));
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchHandler, EpDispatchFFI,
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
//          expert_out [recv_capacity, H]   (always 2D)
// Outputs: result     [..., H]             (N-D; leading dims flattened internally)

Error_Type EpCombineFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type expert_out,
                        Result_Type result, EpCombineConfig config) {
  auto eo_dims = expert_out.dimensions();
  NVTE_CHECK(eo_dims.size() == 2,
             "expert_out must be 2D [recv_capacity, H], got ndim=", eo_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  std::vector<size_t> eo_shape = {static_cast<size_t>(eo_dims[0]), static_cast<size_t>(eo_dims[1])};
  auto eo_dtype = convert_ffi_datatype_to_te_dtype(expert_out.element_type());
  auto expert_out_ = TensorWrapper(expert_out.untyped_data(), eo_shape, eo_dtype);

  auto res_dims = result->dimensions();
  NVTE_CHECK(res_dims.size() >= 2,
             "result must be at least 2D [..., H] (N-D supported; leading dims flattened "
             "internally); got ndim=",
             res_dims.size());
  const size_t res_T_flat = product(res_dims, 0, res_dims.size() - 1);
  NVTE_CHECK(static_cast<int64_t>(res_T_flat) == config.num_local_tokens,
             "result leading-dim product (", res_T_flat, ") must equal num_local_tokens (",
             config.num_local_tokens, ")");
  std::vector<size_t> res_shape = {res_T_flat, static_cast<size_t>(eo_dims[1])};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  nvte_ep_combine(handle_mem_.data(), expert_out_.data(), result_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineHandler, EpCombineFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // expert_out
                                  .Ret<Buffer_Type>()      // result
                                  .Attrs<EpCombineConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          grad [recv_capacity, H]  (grad w.r.t. recv_tokens; always 2D)
// Outputs: grad_tokens [..., H]     (N-D; matches original tokens shape)

Error_Type EpDispatchBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type grad,
                            Result_Type grad_tokens, EpDispatchBwdConfig config) {
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() == 2,
             "grad must be 2D [recv_capacity, H], got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  std::vector<size_t> g_shape = {static_cast<size_t>(grad_dims[0]),
                                 static_cast<size_t>(grad_dims[1])};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto out_dims = grad_tokens->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_tokens must be at least 2D [..., H], got ndim=", out_dims.size());
  const size_t T_flat = product(out_dims, 0, out_dims.size() - 1);
  NVTE_CHECK(static_cast<int64_t>(T_flat) == config.num_local_tokens,
             "grad_tokens leading-dim product (", T_flat, ") must equal num_local_tokens (",
             config.num_local_tokens, ")");
  std::vector<size_t> out_shape = {T_flat, static_cast<size_t>(grad_dims[1])};
  auto grad_tokens_ = TensorWrapper(grad_tokens->untyped_data(), out_shape, g_dtype);

  nvte_ep_dispatch_bwd(handle_mem_.data(), grad_.data(), grad_tokens_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchBwdHandler, EpDispatchBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // grad (w.r.t. recv_tokens)
                                  .Ret<Buffer_Type>()      // grad_tokens
                                  .Attrs<EpDispatchBwdConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          grad [..., H]            (N-D grad w.r.t. result; flattened internally)
// Outputs: grad_expert_out [recv_capacity, H]   (always 2D)

Error_Type EpCombineBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type grad,
                           Result_Type grad_expert_out, EpCombineBwdConfig config) {
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., H], got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t T_flat = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {T_flat, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto out_dims = grad_expert_out->dimensions();
  NVTE_CHECK(out_dims.size() == 2,
             "grad_expert_out must be 2D [recv_capacity, H], got ndim=", out_dims.size());
  std::vector<size_t> out_shape = {static_cast<size_t>(config.recv_capacity), H};
  auto grad_expert_out_ = TensorWrapper(grad_expert_out->untyped_data(), out_shape, g_dtype);

  nvte_ep_combine_bwd(handle_mem_.data(), grad_.data(), grad_expert_out_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineBwdHandler, EpCombineBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // grad (w.r.t. result)
                                  .Ret<Buffer_Type>()      // grad_expert_out
                                  .Attrs<EpCombineBwdConfig>(),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    transformer_engine::jax::EpPrepareConfig,
    ::xla::ffi::StructMember<int64_t>("dispatch_output_per_expert_alignment"));

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
