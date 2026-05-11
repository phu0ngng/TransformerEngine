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
//          handle_id [1] int64 — fresh uint64_t id from EPBackend's atomic counter;
//                   the cache key for subsequent dispatch/combine/*_bwd ops.
//                   Flows through @jax.jit as a regular pytree leaf so XLA's
//                   buffer relocation of handle_mem doesn't break the lookup.

Error_Type EpPrepareFFI(cudaStream_t stream, Buffer_Type topk_idx, Result_Type token_counts,
                        Result_Type handle_mem, Result_Type handle_id, EpPrepareConfig config) {
  auto topk_dims = topk_idx.dimensions();
  NVTE_CHECK(topk_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", topk_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));

  // Flatten leading dims; keep last dim as top_k.
  std::vector<size_t> topk_shape = {product(topk_dims, 0, topk_dims.size() - 1),
                                    static_cast<size_t>(topk_dims.back())};
  // WAR: NCCL EP currently requires topk_idx as int64. Until it accepts int32
  // natively, upcast int32 → int64 on-stream into a scratch buffer here.
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

  uint64_t new_handle_id = 0;
  nvte_ep_prepare(topk_idx_.data(), token_counts_.data(), handle_mem_.data(), &new_handle_id,
                  static_cast<size_t>(config.dispatch_output_per_expert_alignment), stream);

  // Stream-async H→D copy so the id buffer is populated by the time downstream
  // ops on the same stream read it. The id was already allocated host-side by
  // EPBackend::prepare; the copy just publishes it to JAX's pytree leaf.
  NVTE_CHECK(handle_id->element_count() == 1, "handle_id output must be 1 element; got ",
             handle_id->element_count());
  static_assert(sizeof(uint64_t) == sizeof(int64_t),
                "handle_id uint64<->int64 reinterpret requires equal width");
  NVTE_CHECK_CUDA(cudaMemcpyAsync(handle_id->untyped_data(), &new_handle_id, sizeof(uint64_t),
                                  cudaMemcpyHostToDevice, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

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
                                  .Ret<Buffer_Type>()      // handle_id
                                  .Attrs<EpPrepareConfig>(),
                              FFI_CudaGraph_Traits);

// Synchronously read the int64 handle_id from a 1-element device buffer.
// Used by dispatch/combine/*_bwd FFI ops as the cache lookup key.
static uint64_t read_handle_id_sync(Buffer_Type handle_id, cudaStream_t stream) {
  NVTE_CHECK(handle_id.element_count() == 1, "handle_id input must be 1 element; got ",
             handle_id.element_count());
  uint64_t host_id = 0;
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&host_id, handle_id.untyped_data(), sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  return host_id;
}

// ── ep_dispatch ───────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8
//          handle_id  [1] int64 (cache key from ep_prepare)
//          tokens [..., H]            (N-D, flattened internally)
//          topk_idx [..., top_k]      (N-D, flattened internally)
//          topk_weights [..., top_k] float32 (N-D, flattened internally; required)
// Outputs: recv_tokens [recv_capacity, H]      (always 2D)
//          recv_topk_weights [recv_capacity] f32 (always 1D, 1 weight per slot)

Error_Type EpDispatchFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type handle_id,
                         Buffer_Type topk_idx, Buffer_Type tokens, Buffer_Type topk_weights,
                         Result_Type recv_tokens, Result_Type recv_topk_weights,
                         EpDispatchConfig config) {
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
  // config.recv_capacity is the GLOBAL recv_capacity (= ep_size * per-rank);
  // FFI sees per-rank via recv_dims[1]. Kept as attr for ABI symmetry only.
  (void)config.recv_capacity;
  std::vector<size_t> idx_shape = {product(idx_dims, 0, idx_dims.size() - 1),
                                   static_cast<size_t>(idx_dims.back())};
  // WAR: NCCL EP currently requires topk_idx as int64. Until it accepts int32
  // natively, upcast int32 → int64 on-stream into a scratch buffer here.
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

  // Per-shard view from JAX: recv_tokens is 2D [recv_capacity_per_rank, H]
  // (one EP slice; global [recv_capacity, H] sharded on ep_resource).
  auto recv_dims = recv_tokens->dimensions();
  const size_t recv_capacity_per_rank = static_cast<size_t>(recv_dims[0]);
  std::vector<size_t> recv_shape = {recv_capacity_per_rank, H};
  auto recv_tokens_ = TensorWrapper(recv_tokens->untyped_data(), recv_shape, token_dtype);

  auto recv_w_dims = recv_topk_weights->dimensions();
  (void)recv_w_dims;
  std::vector<size_t> recv_w_shape = {recv_capacity_per_rank};
  auto recv_topk_weights_ =
      TensorWrapper(recv_topk_weights->untyped_data(), recv_w_shape, DType::kFloat32);

  uint64_t hid = read_handle_id_sync(handle_id, stream);
  nvte_ep_dispatch(hid, handle_mem_.data(), topk_idx_.data(), tokens_.data(), topk_weights_.data(),
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
                                  .Arg<Buffer_Type>()      // handle_id
                                  .Arg<Buffer_Type>()      // topk_idx
                                  .Arg<Buffer_Type>()      // tokens
                                  .Arg<Buffer_Type>()      // topk_weights
                                  .Ret<Buffer_Type>()      // recv_tokens
                                  .Ret<Buffer_Type>()      // recv_topk_weights
                                  .Attrs<EpDispatchConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine ────────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8 (per-shard from [ep_size, N])
//          handle_id  [1] int64 (cache key from ep_prepare)
//          expert_out [1, recv_capacity_per_rank, H] (per-shard from [ep_size, recv_pr, H])
// Outputs: result     [..., H]             (N-D; leading dims flattened internally)

Error_Type EpCombineFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type handle_id,
                        Buffer_Type expert_out, Result_Type result, EpCombineConfig config) {
  // Per-shard view: expert_out is 2D [recv_capacity_per_rank, H].
  auto eo_dims = expert_out.dimensions();

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = static_cast<size_t>(eo_dims[0]);
  const size_t H = static_cast<size_t>(eo_dims[1]);
  std::vector<size_t> eo_shape = {recv_capacity_per_rank, H};
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
  std::vector<size_t> res_shape = {res_T_flat, H};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  uint64_t hid = read_handle_id_sync(handle_id, stream);
  nvte_ep_combine(hid, handle_mem_.data(), expert_out_.data(), result_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineHandler, EpCombineFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // handle_id
                                  .Arg<Buffer_Type>()      // expert_out
                                  .Ret<Buffer_Type>()      // result
                                  .Attrs<EpCombineConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8 (per-shard)
//          handle_id  [1] int64 (cache key from ep_prepare)
//          grad [1, recv_capacity_per_rank, H] (per-shard from [ep_size, recv_pr, H])
// Outputs: grad_tokens [..., H]     (N-D; matches original tokens shape)

Error_Type EpDispatchBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type handle_id,
                            Buffer_Type grad, Result_Type grad_tokens, EpDispatchBwdConfig config) {
  // Per-shard view: grad is 2D [recv_capacity_per_rank, H].
  auto grad_dims = grad.dimensions();

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = static_cast<size_t>(grad_dims[0]);
  const size_t H = static_cast<size_t>(grad_dims[1]);
  std::vector<size_t> g_shape = {recv_capacity_per_rank, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto out_dims = grad_tokens->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_tokens must be at least 2D [..., H], got ndim=", out_dims.size());
  const size_t T_flat = product(out_dims, 0, out_dims.size() - 1);
  NVTE_CHECK(static_cast<int64_t>(T_flat) == config.num_local_tokens,
             "grad_tokens leading-dim product (", T_flat, ") must equal num_local_tokens (",
             config.num_local_tokens, ")");
  std::vector<size_t> out_shape = {T_flat, H};
  auto grad_tokens_ = TensorWrapper(grad_tokens->untyped_data(), out_shape, g_dtype);

  uint64_t hid = read_handle_id_sync(handle_id, stream);
  nvte_ep_dispatch_bwd(hid, handle_mem_.data(), grad_.data(), grad_tokens_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchBwdHandler, EpDispatchBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // handle_id
                                  .Arg<Buffer_Type>()      // grad (w.r.t. recv_tokens)
                                  .Ret<Buffer_Type>()      // grad_tokens
                                  .Attrs<EpDispatchBwdConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────
// Inputs:  handle_mem [N] uint8 (per-shard)
//          handle_id  [1] int64 (cache key from ep_prepare)
//          grad [..., H]            (N-D grad w.r.t. result; flattened internally)
// Outputs: grad_expert_out [1, recv_capacity_per_rank, H] (per-shard from
//                          [ep_size, recv_pr, H])

Error_Type EpCombineBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type handle_id,
                           Buffer_Type grad, Result_Type grad_expert_out,
                           EpCombineBwdConfig config) {
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., H], got ndim=", grad_dims.size());
  (void)config.recv_capacity;  // see note below; we read per-rank from output dims.

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t T_flat = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {T_flat, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  // Python `abstract` enforces ndim==3; only the per-shard leading-dim==1
  // invariant needs to be re-checked here (can't be expressed at global abstract).
  auto out_dims = grad_expert_out->dimensions();
  NVTE_CHECK(out_dims[0] == 1, "grad_expert_out leading dim must be 1 per shard (ep-sliced); got ",
             out_dims[0]);
  const size_t recv_capacity_per_rank = static_cast<size_t>(out_dims[1]);
  std::vector<size_t> out_shape = {recv_capacity_per_rank, H};
  auto grad_expert_out_ = TensorWrapper(grad_expert_out->untyped_data(), out_shape, g_dtype);

  uint64_t hid = read_handle_id_sync(handle_id, stream);
  nvte_ep_combine_bwd(hid, handle_mem_.data(), grad_.data(), grad_expert_out_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineBwdHandler, EpCombineBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // handle_id
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
