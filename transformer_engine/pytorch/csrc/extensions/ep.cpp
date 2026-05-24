/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <tuple>
#include <vector>

#include "transformer_engine/comm_window.h"

#ifdef NCCL_HAS_SYMMEM_SUPPORT
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#endif

#include "../common.h"
#include "../extensions.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine::pytorch {

namespace {

// EP process group name, captured at ep_bootstrap. Used by per-step ops to
// look up SymmetricMemory for payload tensors. Empty until bootstrap.
std::string g_ep_group_name;  // NOLINT(runtime/string)

// True while the EP backend holds a borrowed reference to torch's NCCL comm.
bool g_ep_initialized = false;

// When false, per-step ops skip symm-mem window annotation and the backend
// takes the staged-copy path. Atomic so the Python-side `_zero_copy_scope`
// toggle is safe against concurrent ep_dispatch/combine (which release the GIL).
std::atomic<bool> g_zero_copy_enabled{true};

// Warn-once per role when an EP payload tensor isn't symm-mem-backed (the
// staged-copy fallback is correct but slower). Set NVTE_EP_SILENCE_NONSYMM_WARN=1
// to silence (CI or known-non-symm paths).
void warn_if_not_symm(const at::Tensor& t, const char* role) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  if (g_ep_group_name.empty()) return;
  if (c10d::symmetric_memory::is_symm_mem_tensor(t)) return;
  static const bool silenced = []() {
    const char* e = std::getenv("NVTE_EP_SILENCE_NONSYMM_WARN");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  if (silenced) return;
  static std::atomic<uint32_t> warned_mask{0};
  uint32_t h = 5381;
  for (const char* p = role; *p; ++p) h = h * 33u + static_cast<unsigned char>(*p);
  const uint32_t bit = 1u << (h & 31u);
  if ((warned_mask.fetch_or(bit) & bit) != 0) return;
  std::fprintf(stderr,
               "[NVTE EP] WARNING: %s tensor is not backed by an NCCL symmetric memory "
               "window; falling back to staged copy. Allocate this buffer via the "
               "framework's symm-mem API and rendezvous it for the zero-copy fast path. "
               "Set NVTE_EP_SILENCE_NONSYMM_WARN=1 to suppress.\n",
               role);
#else
  (void)t;
  (void)role;
#endif
}

// Build the NVTECommWindow descriptor for ``t`` so the backend can take the
// zero-copy one-sided path. Returns ``{nullptr, 0}`` when symm-mem is disabled,
// not yet bootstrapped, or ``t`` isn't symm-mem-backed — in which case the
// backend falls back to the raw-pointer staged path.
NVTECommWindow maybe_make_window(const at::Tensor& t) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  if (!g_zero_copy_enabled.load(std::memory_order_relaxed)) return NVTECommWindow{nullptr, 0};
  if (g_ep_group_name.empty()) return NVTECommWindow{nullptr, 0};
  if (!c10d::symmetric_memory::is_symm_mem_tensor(t)) return NVTECommWindow{nullptr, 0};
  auto sm = c10d::symmetric_memory::rendezvous(t, g_ep_group_name);
  NVTE_CHECK(sm != nullptr,
             "EP payload tensor is symm-mem-backed but not rendezvoused on the EP group; "
             "call symm_mem_alloc (or rendezvous explicitly at allocation time) before EP ops.");
  auto* nccl_sm = dynamic_cast<c10d::symmetric_memory::NCCLSymmetricMemory*>(sm.get());
  NVTE_CHECK(nccl_sm != nullptr,
             "Symm-mem backend mismatch: expected NCCLSymmetricMemory. Set the backend to "
             "\"NCCL\" before allocating EP payload buffers.");
  return NVTECommWindow{static_cast<ncclWindow_t>(nccl_sm->get_window()),
                        static_cast<uint64_t>(nccl_sm->get_offset())};
#else
  (void)t;
  return NVTECommWindow{nullptr, 0};
#endif
}

// The backend only accepts int64 topk_idx. The PyTorch wrapper enforces this
// at the boundary so the per-step ops don't need an upcast workspace.
void check_topk_idx_int64(at::Tensor topk_idx) {
  NVTE_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
  NVTE_CHECK(topk_idx.scalar_type() == at::kLong,
             "topk_idx must be int64; got dtype=", c10::toString(topk_idx.scalar_type()),
             ". Cast with topk_idx.long() before calling.");
}

using Shape = std::vector<size_t>;

}  // namespace

void ep_set_zero_copy(bool enabled) {
  g_zero_copy_enabled.store(enabled, std::memory_order_relaxed);
}

bool ep_get_zero_copy() { return g_zero_copy_enabled.load(std::memory_order_relaxed); }

// ── Bootstrap ────────────────────────────────────────────────────────────────
// Borrows torch's NCCL host comm (from ``ProcessGroupNCCL._comm_ptr()``) so
// symm-mem windows are visible to NCCL EP's shadow pool.

void ep_initialize(uintptr_t comm_ptr, const std::string& group_name, int64_t num_experts,
                   int64_t max_tokens_per_rank, int64_t max_recv_tokens_per_rank,
                   int64_t hidden_dim, int64_t max_num_sms, bool allow_handle_mem_reloc) {
  NVTE_CHECK(!group_name.empty(), "group_name must be non-empty (used for symm-mem lookup)");
  NVTE_CHECK(comm_ptr != 0, "comm_ptr must be non-null (torch NCCL host comm pointer)");
  NVTE_CHECK(!g_ep_initialized, "ep_initialize called twice without ep_finalize");

  auto ep_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  int ep_size = 0;
  NVTE_CHECK(ncclCommCount(ep_comm, &ep_size) == ncclSuccess, "ncclCommCount failed");
  NVTEEpGroupConfig cfg{
      /*ep_size=*/ep_size,
      /*num_experts=*/static_cast<int>(num_experts),
      /*max_tokens_per_rank=*/static_cast<int>(max_tokens_per_rank),
      /*max_recv_tokens_per_rank=*/static_cast<int>(max_recv_tokens_per_rank),
      /*hidden_dim=*/static_cast<int>(hidden_dim),
      /*max_num_sms=*/static_cast<int>(max_num_sms),
      /*allow_handle_mem_reloc=*/allow_handle_mem_reloc ? 1 : 0,
  };
  nvte_ep_initialize(static_cast<void*>(ep_comm), cfg);
  g_ep_initialized = true;
  g_ep_group_name = group_name;
}

void ep_finalize() {
  if (!g_ep_initialized) return;
  // The borrowed comm is owned by torch's symm-mem layer; don't destroy it.
  nvte_ep_shutdown();
  g_ep_initialized = false;
  g_ep_group_name.clear();
}

std::tuple<int64_t, int64_t> ep_register_layer(int64_t top_k,
                                               int64_t dispatch_output_per_expert_alignment) {
  NVTEEpLayerConfig layer_cfg{0, static_cast<int>(top_k),
                              static_cast<size_t>(dispatch_output_per_expert_alignment)};
  size_t handle_mem_size = 0;
  uint64_t handle_id = nvte_ep_register_layer(layer_cfg, &handle_mem_size);
  return std::make_tuple(static_cast<int64_t>(handle_id), static_cast<int64_t>(handle_mem_size));
}

// ── Per-step ops ─────────────────────────────────────────────────────────────

void ep_prepare(at::Tensor handle_mem, int64_t handle_id, at::Tensor topk_idx,
                at::Tensor token_counts, int64_t dispatch_output_per_expert_alignment) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  check_topk_idx_int64(topk_idx);
  const size_t T_flat = topk_idx.numel() / topk_idx.size(-1);
  const size_t topk_n = static_cast<size_t>(topk_idx.size(-1));

  auto topk_idx_te =
      makeTransformerEngineTensor(topk_idx.data_ptr(), Shape{T_flat, topk_n}, DType::kInt64);
  auto token_counts_te = makeTransformerEngineTensor(
      token_counts.data_ptr(), Shape{static_cast<size_t>(token_counts.numel())}, DType::kInt32);
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);

  NVTEEpHandle handle{static_cast<uint64_t>(handle_id), handle_mem_te.data()};
  nvte_ep_prepare(handle, topk_idx_te.data(), token_counts_te.data(),
                  static_cast<size_t>(dispatch_output_per_expert_alignment), stream);
}

void ep_dispatch(at::Tensor handle_mem, int64_t handle_id, at::Tensor topk_idx, at::Tensor tokens,
                 at::Tensor topk_weights, at::Tensor recv_tokens, at::Tensor recv_topk_weights) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(tokens.dim() >= 2, "tokens must be at least 2D [..., H]");
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  NVTE_CHECK(topk_weights.dim() >= 2, "topk_weights must be at least 2D [..., top_k]");
  NVTE_CHECK(recv_tokens.dim() >= 2, "recv_tokens must be at least 2D [..., recv_pr, H]");
  check_topk_idx_int64(topk_idx);

  const size_t H = static_cast<size_t>(tokens.size(-1));
  const size_t T_flat = tokens.numel() / H;
  const size_t topk_n = static_cast<size_t>(topk_idx.size(-1));
  const size_t recv_pr = recv_tokens.numel() / H;

  NVTE_CHECK(static_cast<size_t>(topk_weights.size(-1)) == topk_n,
             "topk_weights last dim must equal topk_idx last dim");
  NVTE_CHECK(static_cast<size_t>(recv_topk_weights.numel()) == recv_pr,
             "recv_topk_weights total size must equal recv_tokens recv_pr");
  NVTE_CHECK(recv_tokens.scalar_type() == tokens.scalar_type(), "recv_tokens dtype (",
             c10::toString(recv_tokens.scalar_type()), ") must match tokens dtype (",
             c10::toString(tokens.scalar_type()), ")");

  auto tok_dtype = GetTransformerEngineDType(tokens.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto topk_idx_te =
      makeTransformerEngineTensor(topk_idx.data_ptr(), Shape{T_flat, topk_n}, DType::kInt64);
  auto tokens_te = makeTransformerEngineTensor(tokens.data_ptr(), Shape{T_flat, H}, tok_dtype);
  warn_if_not_symm(tokens, "dispatch input (tokens)");
  NVTECommWindow tokens_win = maybe_make_window(tokens);
  auto topk_w_te =
      makeTransformerEngineTensor(topk_weights.data_ptr(), Shape{T_flat, topk_n}, DType::kFloat32);
  // topk_weights symm-mem backing is nice-to-have, not required — silently
  // fall back to the staged path if the caller didn't allocate it via symm-mem.
  NVTECommWindow topk_weights_win = maybe_make_window(topk_weights);
  auto recv_tokens_te =
      makeTransformerEngineTensor(recv_tokens.data_ptr(), Shape{recv_pr, H}, tok_dtype);
  warn_if_not_symm(recv_tokens, "dispatch output (recv_tokens)");
  NVTECommWindow recv_tokens_win = maybe_make_window(recv_tokens);
  auto recv_topk_w_te =
      makeTransformerEngineTensor(recv_topk_weights.data_ptr(), Shape{recv_pr}, DType::kFloat32);
  NVTECommWindow recv_topk_weights_win = maybe_make_window(recv_topk_weights);

  NVTEEpHandle handle{static_cast<uint64_t>(handle_id), handle_mem_te.data()};
  nvte_ep_dispatch(handle, topk_idx_te.data(), tokens_te.data(), tokens_win, topk_w_te.data(),
                   topk_weights_win, recv_tokens_te.data(), recv_tokens_win, recv_topk_w_te.data(),
                   recv_topk_weights_win, stream);
}

void ep_combine(at::Tensor handle_mem, int64_t handle_id, at::Tensor expert_out,
                at::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(expert_out.dim() >= 2, "expert_out must be at least 2D [..., recv_pr, H]");
  NVTE_CHECK(result.dim() >= 2, "result must be at least 2D [..., H]");

  const size_t H = static_cast<size_t>(expert_out.size(-1));
  const size_t recv_pr = expert_out.numel() / H;
  const size_t T_flat = result.numel() / H;
  NVTE_CHECK(static_cast<size_t>(result.size(-1)) == H,
             "result hidden dim must equal expert_out hidden dim");
  NVTE_CHECK(result.scalar_type() == expert_out.scalar_type(), "result dtype (",
             c10::toString(result.scalar_type()), ") must match expert_out dtype (",
             c10::toString(expert_out.scalar_type()), ")");

  auto eo_dtype = GetTransformerEngineDType(expert_out.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto expert_out_te =
      makeTransformerEngineTensor(expert_out.data_ptr(), Shape{recv_pr, H}, eo_dtype);
  warn_if_not_symm(expert_out, "combine input (expert_out)");
  NVTECommWindow expert_out_win = maybe_make_window(expert_out);
  // combine ``result`` is local accumulation (not cross-rank put/get); leave it
  // un-annotated so the backend uses the raw-pointer path regardless of how it
  // was allocated.
  auto result_te = makeTransformerEngineTensor(result.data_ptr(), Shape{T_flat, H}, eo_dtype);

  NVTEEpHandle handle{static_cast<uint64_t>(handle_id), handle_mem_te.data()};
  nvte_ep_combine(handle, expert_out_te.data(), expert_out_win, result_te.data(), stream);
}

void ep_dispatch_bwd(at::Tensor handle_mem, int64_t handle_id, at::Tensor grad,
                     at::Tensor g_recv_topk_weights, at::Tensor grad_tokens,
                     at::Tensor grad_topk_weights) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(grad.dim() >= 2, "grad must be at least 2D [..., recv_pr, H]");
  NVTE_CHECK(grad_tokens.dim() >= 2, "grad_tokens must be at least 2D [..., H]");
  NVTE_CHECK(grad_topk_weights.dim() >= 2, "grad_topk_weights must be at least 2D [..., top_k]");

  const size_t H = static_cast<size_t>(grad.size(-1));
  const size_t recv_pr = grad.numel() / H;
  const size_t T_flat = grad_tokens.numel() / H;
  const size_t topk_n = static_cast<size_t>(grad_topk_weights.size(-1));
  NVTE_CHECK(static_cast<size_t>(g_recv_topk_weights.numel()) == recv_pr,
             "g_recv_topk_weights total size must equal grad recv_pr");
  NVTE_CHECK(static_cast<size_t>(grad_tokens.size(-1)) == H,
             "grad_tokens hidden dim must equal grad H");
  NVTE_CHECK(static_cast<size_t>(grad_topk_weights.numel()) == T_flat * topk_n,
             "grad_topk_weights numel (", grad_topk_weights.numel(),
             ") must equal T_flat * top_k (", T_flat * topk_n, ")");
  NVTE_CHECK(grad_tokens.scalar_type() == grad.scalar_type(), "grad_tokens dtype (",
             c10::toString(grad_tokens.scalar_type()), ") must match grad dtype (",
             c10::toString(grad.scalar_type()), ")");

  auto g_dtype = GetTransformerEngineDType(grad.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto grad_te = makeTransformerEngineTensor(grad.data_ptr(), Shape{recv_pr, H}, g_dtype);
  warn_if_not_symm(grad, "dispatch_bwd input (grad)");
  NVTECommWindow grad_win = maybe_make_window(grad);
  auto g_recv_w_te =
      makeTransformerEngineTensor(g_recv_topk_weights.data_ptr(), Shape{recv_pr}, DType::kFloat32);
  NVTECommWindow g_recv_topk_weights_win = maybe_make_window(g_recv_topk_weights);
  auto grad_tokens_te =
      makeTransformerEngineTensor(grad_tokens.data_ptr(), Shape{T_flat, H}, g_dtype);
  auto grad_topk_w_te = makeTransformerEngineTensor(grad_topk_weights.data_ptr(),
                                                    Shape{T_flat, topk_n}, DType::kFloat32);

  NVTEEpHandle handle{static_cast<uint64_t>(handle_id), handle_mem_te.data()};
  nvte_ep_dispatch_bwd(handle, grad_te.data(), grad_win, g_recv_w_te.data(),
                       g_recv_topk_weights_win, grad_tokens_te.data(), grad_topk_w_te.data(),
                       stream);
}

void ep_combine_bwd(at::Tensor handle_mem, int64_t handle_id, at::Tensor grad,
                    at::Tensor grad_expert_out) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(grad.dim() >= 2, "grad must be at least 2D [..., H]");
  NVTE_CHECK(grad_expert_out.dim() >= 2, "grad_expert_out must be at least 2D [..., recv_pr, H]");

  const size_t H = static_cast<size_t>(grad.size(-1));
  const size_t T_flat = grad.numel() / H;
  const size_t recv_pr = grad_expert_out.numel() / H;
  NVTE_CHECK(static_cast<size_t>(grad_expert_out.size(-1)) == H,
             "grad_expert_out hidden dim must match grad H");
  NVTE_CHECK(grad_expert_out.scalar_type() == grad.scalar_type(), "grad_expert_out dtype (",
             c10::toString(grad_expert_out.scalar_type()), ") must match grad dtype (",
             c10::toString(grad.scalar_type()), ")");

  auto g_dtype = GetTransformerEngineDType(grad.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto grad_te = makeTransformerEngineTensor(grad.data_ptr(), Shape{T_flat, H}, g_dtype);
  NVTECommWindow grad_win = maybe_make_window(grad);
  auto grad_eo_te =
      makeTransformerEngineTensor(grad_expert_out.data_ptr(), Shape{recv_pr, H}, g_dtype);
  NVTECommWindow grad_eo_win = maybe_make_window(grad_expert_out);

  NVTEEpHandle handle{static_cast<uint64_t>(handle_id), handle_mem_te.data()};
  nvte_ep_combine_bwd(handle, grad_te.data(), grad_win, grad_eo_te.data(), grad_eo_win, stream);
}

void register_ep_bindings(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("ep_initialize", &ep_initialize,
        "Initialize the EP backend; borrows torch's NCCL comm pointed to by ``comm_ptr``.",
        py::arg("comm_ptr"), py::arg("group_name"), py::arg("num_experts"),
        py::arg("max_tokens_per_rank"), py::arg("max_recv_tokens_per_rank"), py::arg("hidden_dim"),
        py::arg("max_num_sms") = 0, py::arg("allow_handle_mem_reloc") = false,
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_finalize", &ep_finalize, "Tear down the EP backend. Idempotent.",
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_set_zero_copy", &ep_set_zero_copy, "Toggle EP zero-copy symm-mem annotation.",
        py::arg("enabled"));
  m.def("ep_get_zero_copy", &ep_get_zero_copy, "Return the current EP zero-copy toggle state.");
  m.def("ep_register_layer", &ep_register_layer,
        "Register an EP layer; returns (handle_id, handle_mem_size_bytes).", py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment") = 0);
  m.def("ep_prepare", &ep_prepare, "EP prepare", py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch", &ep_dispatch, "EP dispatch", py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine", &ep_combine, "EP combine", py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch_bwd", &ep_dispatch_bwd, "EP dispatch backward",
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine_bwd", &ep_combine_bwd, "EP combine backward",
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace transformer_engine::pytorch

#endif  // NVTE_WITH_NCCL_EP
