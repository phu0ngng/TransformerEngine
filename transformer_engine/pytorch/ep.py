# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
import warnings
from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

import transformer_engine_torch as tex

from .cpu_offload import mark_not_offload
from .distributed import symm_mem_alloc, release_symm_mem_pool

if TYPE_CHECKING:
    from ..common.recipe import Recipe

__all__ = [
    "EpBuffer",
    "ep_bootstrap",
    "ep_finalize",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
    "release_symm_mem_pool",
    "is_symm_backed",
]


# ``symm_mem_alloc`` (imported from .distributed) allocates the symm-mem buffers
# used by the zero-copy IO path. Set ``ep_bootstrap(zero_copy=True)`` to opt in;
# the C++ backend then operates the EP group in zero-copy mode.


# Bootstrap


# NCCL EP requires NCCL >= 2.30.4 (matches the C++ backend's runtime check).
_MIN_NCCL_VERSION = (2, 30, 4)


def _check_nccl_runtime_version() -> None:
    """Raise with a clear message if the loaded libnccl is too old for NCCL EP."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
        v = ctypes.c_int(0)
        if lib.ncclGetVersion(ctypes.byref(v)) != 0:
            warnings.warn("ncclGetVersion failed; skipping NCCL EP version check.")
            return
    except OSError:  # libnccl not findable; let the C++ side error
        return
    n = v.value
    # NCCL packs as (major*10000 + minor*100 + patch) up to ~2.x; newer
    # builds use the same scheme. Decode defensively.
    major, minor, patch = n // 10000, (n // 100) % 100, n % 100
    if (major, minor, patch) < _MIN_NCCL_VERSION:
        min_str = ".".join(str(x) for x in _MIN_NCCL_VERSION)
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {min_str}, found {major}.{minor}.{patch} at runtime. "
            "Set LD_LIBRARY_PATH to a newer libnccl.so before launching."
        )


_BOOTSTRAPPED = False
_ATEXIT_REGISTERED = False
# EP group captured at bootstrap; EpBuffer uses it to allocate the symm-mem
# combine grad buffer in zero-copy mode.
_EP_GROUP: Optional[dist.ProcessGroup] = None


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown; swallows errors."""
    global _BOOTSTRAPPED, _EP_GROUP
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False
            _EP_GROUP = None


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    max_num_sms: int = 0,
    zero_copy: bool = False,
    max_token_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Initialize EP by borrowing ep_group's NCCL comm. Call once per process.

    max_token_dtype sets the widest token dtype this EP group will dispatch;
    it sizes NCCL EP staging buffers.

    ``zero_copy`` opts the EP group into the symm-mem zero-copy IO path; pass
    ``True`` only when payload tensors are allocated via ``symm_mem_alloc``.
    Defaults to ``False``.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED, _EP_GROUP
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")
    _check_nccl_runtime_version()
    if zero_copy:
        warnings.warn(
            "ep_bootstrap(zero_copy=True) is experimental; the symm-mem IO path "
            "and its alias contracts on EpBuffer slots are subject to change.",
            stacklevel=2,
        )

    # Materialize the PG's NCCL comm before borrowing its raw handle.
    dist.barrier(group=ep_group, device_ids=[torch.cuda.current_device()])
    comm_ptr = ep_group._get_backend(torch.device("cuda"))._comm_ptr()

    tex.ep_initialize(
        int(comm_ptr),
        str(ep_group.group_name),
        int(num_experts),
        int(max_tokens_per_rank),
        int(recv_capacity_per_rank),
        int(hidden_dim),
        int(max_num_sms),
        max_token_dtype,
        bool(zero_copy),
    )
    _BOOTSTRAPPED = True
    _EP_GROUP = ep_group
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def ep_finalize() -> None:
    """Optional explicit EP teardown; idempotent.

    An atexit handler covers normal interpreter shutdown, so most users do not
    need to call this. Call it explicitly only before
    ``dist.destroy_process_group()``, since the borrowed NCCL comm becomes
    invalid once the PG is destroyed.
    """
    global _BOOTSTRAPPED, _EP_GROUP
    if not _BOOTSTRAPPED:
        return
    try:
        tex.ep_finalize()
    finally:
        _BOOTSTRAPPED = False
        _EP_GROUP = None


def is_symm_backed(t: torch.Tensor) -> bool:
    """Whether ``t`` is symm-mem-backed on the EP group. Prefer torch's local ``is_symm_mem_tensor``
    when the build provides it (no collective, no exception); otherwise fall back to the rendezvous
    probe the C++ ep kernel uses (``maybe_make_window``): cached for an already-registered tensor,
    raises for a plain one."""
    from torch.distributed import _symmetric_memory as _symm

    if hasattr(_symm, "is_symm_mem_tensor"):
        return bool(_symm.is_symm_mem_tensor(t))
    if _EP_GROUP is None:
        raise RuntimeError(
            "is_symm_backed called before ensure_nccl_ep_bootstrapped(); no EP group registered."
        )
    try:
        _symm.rendezvous(t, _EP_GROUP.group_name)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


# Buffer


class EpBuffer:
    """Per-microbatch EP layer state: handle_mem, token_counts, and shape/dtype config.
    Use one EpBuffer per concurrently-in-flight call (e.g. per PP-1F1B microbatch).
    """

    __slots__ = (
        "handle_mem",
        "top_k",
        "alignment",
        "max_tokens_per_rank",
        "recv_capacity_per_rank",
        "hidden_dim",
        "num_local_experts",
        "payload_dtype",
        "device",
        "token_counts",
        "zero_copy",
        "dispatch_quant_recipe",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        dispatch_quant_recipe: Optional["Recipe"] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        self.top_k = int(top_k)
        self.alignment = alignment
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.recv_capacity_per_rank = int(recv_capacity_per_rank)
        self.hidden_dim = int(hidden_dim)
        self.num_local_experts = int(num_local_experts)
        self.payload_dtype = payload_dtype
        self.device = device
        self.zero_copy = bool(tex.ep_get_zero_copy())
        self.dispatch_quant_recipe = dispatch_quant_recipe

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        self.token_counts = torch.empty(self.num_local_experts, dtype=torch.int32, device=device)
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(self.handle_mem)


# torch.library custom ops (so they don't graph-break under torch.compile)

_LIB = "transformer_engine_ep"


@torch.library.custom_op(
    f"{_LIB}::prepare",
    mutates_args=("handle_mem", "token_counts"),
    device_types="cuda",
)
def _prepare_op(
    handle_mem: torch.Tensor,
    top_k: int,
    topk_idx: torch.Tensor,
    token_counts: torch.Tensor,
    alignment: int,
) -> None:
    tex.ep_prepare(handle_mem, topk_idx, token_counts, top_k, alignment)


@_prepare_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch",
    mutates_args=("recv_tokens", "recv_topk_weights", "recv_scale_inv"),
    device_types="cuda",
)
def _dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    tokens_scale_inv: Optional[torch.Tensor] = None,
    recv_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_dispatch(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        recv_tokens,
        recv_topk_weights,
        tokens_scale_inv,
        recv_scale_inv,
    )


@_dispatch_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine",
    mutates_args=("result",),
    device_types="cuda",
)
def _combine_op(
    handle_mem: torch.Tensor,
    expert_out: torch.Tensor,
    result: torch.Tensor,
) -> None:
    tex.ep_combine(handle_mem, expert_out, result)


@_combine_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch_bwd",
    mutates_args=("grad_tokens", "grad_topk_weights"),
    device_types="cuda",
)
def _dispatch_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch_bwd(handle_mem, grad, g_recv_topk_weights, grad_tokens, grad_topk_weights)


@_dispatch_bwd_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine_bwd",
    mutates_args=("grad_expert_out",),
    device_types="cuda",
)
def _combine_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    grad_expert_out: torch.Tensor,
) -> None:
    tex.ep_combine_bwd(handle_mem, grad, grad_expert_out)


@_combine_bwd_op.register_fake
def _(*_args, **_kw):
    return None


# Non-autograd primitives


def ep_prepare(buffer: "EpBuffer", topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``buffer.handle_mem`` and returns
    ``buffer.token_counts`` (int32, shape [num_local_experts]). topk_idx must
    be int32 or int64.
    """
    torch.ops.transformer_engine_ep.prepare(
        buffer.handle_mem, buffer.top_k, topk_idx, buffer.token_counts, buffer.alignment
    )
    return buffer.token_counts


def _ep_dispatch_raw(
    buffer: "EpBuffer",
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    """Raw dispatch; no autograd, no prepare. Caller must run ep_prepare first."""
    tex.ep_dispatch(
        buffer.handle_mem, topk_idx, tokens, topk_weights, recv_tokens, recv_topk_weights
    )


def _ep_combine_raw(buffer: "EpBuffer", expert_out: torch.Tensor, result: torch.Tensor) -> None:
    """Raw combine; no autograd. Caller pre-weights expert_out."""
    tex.ep_combine(buffer.handle_mem, expert_out, result)


# autograd.Function wrappers


class _EpDispatch(torch.autograd.Function):
    """Autograd prepare+dispatch; bwd uses user-supplied grad inputs as-is."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        top_k: int,
        alignment: int,
        recv_tokens: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        token_counts: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_scale_inv: Optional[torch.Tensor] = None,
        recv_scale_inv: Optional[torch.Tensor] = None,
    ):
        """Prepare + dispatch fwd. When scales are set (FP8; MXFP8 for now), ``tokens`` is the
        quantized tensor itself, kept as the autograd operand so grad reaches the pre-quant
        input through the quantizer."""
        from .quantized_tensor import QuantizedTensor

        is_scaled = tokens_scale_inv is not None
        tokens_data = tokens._rowwise_data if isinstance(tokens, QuantizedTensor) else tokens
        hidden = tokens_data.shape[-1]
        tokens_data = tokens_data.reshape(-1, hidden)
        # Reinterpret byte-backed FP8 data as the fp8 dtype so the backend sees a
        # scaled tensor.
        dispatch_tokens = tokens_data
        dispatch_recv = recv_tokens
        if is_scaled:
            fp8_view_dtype = (
                torch.float8_e5m2
                if tokens._fp8_dtype == tex.DType.kFloat8E5M2
                else torch.float8_e4m3fn
            )
            dispatch_tokens = tokens_data.view(fp8_view_dtype)
            dispatch_recv = recv_tokens.view(fp8_view_dtype)
        torch.ops.transformer_engine_ep.prepare(
            handle_mem, top_k, topk_idx, token_counts, alignment
        )
        torch.ops.transformer_engine_ep.dispatch(
            handle_mem,
            topk_idx,
            dispatch_tokens,
            topk_weights,
            dispatch_recv,
            recv_topk_weights,
            tokens_scale_inv,
            recv_scale_inv,
        )
        ctx.save_for_backward(handle_mem)
        ctx.tokens_shape = tokens.shape
        # Dispatch grad is high-precision (the quantizer's STE owns the fp8 boundary),
        # so grad_tokens is bf16 for scaled inputs.
        ctx.tokens_dtype = torch.bfloat16 if is_scaled else tokens.dtype
        ctx.topk_weights_shape = topk_weights.shape
        ctx.tokens_T_flat = tokens_data.shape[0]
        ctx.topk_T_flat = topk_weights.numel() // topk_weights.shape[-1]
        ctx.top_k = topk_weights.shape[-1]
        ctx.recv_capacity = recv_tokens.shape[0]
        ctx.hidden_dim = hidden
        ctx.mark_non_differentiable(token_counts)
        # Detach so the long-lived buffers aren't tracked as differentiable outputs;
        # autograd re-attaches grad_fn pointing back at this Function. For scaled
        # inputs the recv data + scales are wrapped into a single differentiable
        # MXFP8Tensor so downstream ops and autograd see a proper quantized tensor.
        if is_scaled:
            from .tensor.mxfp8_tensor import MXFP8Quantizer

            quantizer = MXFP8Quantizer(fp8_dtype=tokens._fp8_dtype, rowwise=True, columnwise=False)
            # View the recv buffer as its fp8 dtype (it may be byte-typed under zero-copy).
            recv_out = quantizer.create_tensor_from_data(
                recv_tokens.view(tokens._rowwise_data.dtype).detach(),
                recv_scale_inv.detach(),
                fake_dtype=tokens.dtype,
                fp8_dtype=tokens._fp8_dtype,
            )
        else:
            recv_out = recv_tokens.detach()
        return recv_out, recv_topk_weights.detach(), token_counts

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_token_counts):  # type: ignore[override]
        """Dispatch bwd; normalizes grad-input layout, otherwise passes through."""
        (handle_mem,) = ctx.saved_tensors
        device = handle_mem.device
        g_recv_tokens = g_recv_tokens.contiguous()
        g_recv_topk_weights = g_recv_topk_weights.contiguous()
        grad_tokens = torch.empty(
            ctx.tokens_T_flat, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
        )
        grad_topk_weights = torch.empty(
            ctx.topk_T_flat, ctx.top_k, dtype=torch.float32, device=device
        )
        torch.ops.transformer_engine_ep.dispatch_bwd(
            handle_mem,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
        return (
            None,  # handle_mem
            None,  # top_k
            None,  # alignment
            None,  # recv_tokens
            None,  # recv_topk_weights
            None,  # token_counts
            None,  # topk_idx
            grad_tokens.view(ctx.tokens_shape),
            grad_topk_weights.view(ctx.topk_weights_shape),
            None,  # tokens_scale_inv (scales; non-differentiable)
            None,  # recv_scale_inv (mutated buffer; non-differentiable)
        )


class _EpCombine(torch.autograd.Function):
    """Autograd combine.

    bwd scatters the expert_out grad into ``grad_out``. When the caller supplies it
    (mcore-managed mode) that buffer is used as-is; otherwise it is allocated on the
    fly here — from the symm-mem pool in zero-copy mode (one-sided target), or a plain
    tensor in normal mode (keeps allocation torch.compile / CUDA-graph safe and lets
    autograd own the grad's lifetime).

    ``grad_out`` is the backward's scatter target (an output it writes, never reads),
    so it is stashed as a plain ctx attribute rather than via save_for_backward, which
    would version-track a tensor we mutate.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        grad_out: Optional[torch.Tensor],
        expert_out: torch.Tensor,
    ):
        """Combine fwd; stashes the bwd grad target or expert_out shape to size it."""
        device = expert_out.device
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        torch.ops.transformer_engine_ep.combine(handle_mem, expert_out, result)
        ctx.save_for_backward(handle_mem)
        ctx.grad_out = grad_out
        if grad_out is None:
            ctx.expert_out_shape = expert_out.shape
            ctx.expert_out_dtype = expert_out.dtype
            ctx.device = device
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        """Combine bwd; scatters the result grad into the grad target."""
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        (handle_mem,) = ctx.saved_tensors
        grad_expert_out = ctx.grad_out
        if grad_expert_out is None:
            grad_expert_out = _alloc_io(
                ctx.expert_out_shape, ctx.expert_out_dtype, ctx.device, tex.ep_get_zero_copy()
            )
        torch.ops.transformer_engine_ep.combine_bwd(handle_mem, g_result, grad_expert_out)
        return (
            None,  # handle_mem
            None,  # num_local_tokens
            None,  # hidden_dim
            None,  # grad_out
            grad_expert_out,
        )


# Public high-level wrappers


# NCCL EP payloads are bfloat16, or a block-scaled QuantizedTensor (MXFP8).
def _require_bf16(name: str, t: torch.Tensor) -> None:
    if t.dtype is not torch.bfloat16:
        raise NotImplementedError(
            "NCCL EP currently supports only bfloat16 or MXFP8 payloads; got"
            f" {name}.dtype={t.dtype}."
        )


def _alloc_io(shape, dtype: torch.dtype, device, zero_copy: bool) -> torch.Tensor:
    """Allocate a dispatch/combine IO tensor the caller did not supply: from the symm-mem pool in
    zero-copy mode (auto-registered segment, lifecycle managed by torch refcount), else plain."""
    if zero_copy:
        t = symm_mem_alloc(shape, dtype, _EP_GROUP, device=device, use_pool=True)
        # symm-mem storage is non-resizable; exempt it from CPU activation offloading (which
        # releases via storage.resize_(0)). Matters for bf16 recv_tokens (the saved activation).
        mark_not_offload(t)
        return t
    return torch.empty(*shape, dtype=dtype, device=device)


def _check_topk_weights_f32(topk_weights: torch.Tensor) -> None:
    if topk_weights.dtype is not torch.float32:
        raise TypeError(
            f"topk_weights must be float32; got dtype={topk_weights.dtype}. "
            "Cast with topk_weights.float() before calling."
        )


def ep_dispatch(
    buffer: EpBuffer,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    recv_tokens: Optional[torch.Tensor] = None,
    recv_topk_weights: Optional[torch.Tensor] = None,
):
    """Prepare + dispatch with autograd. topk_idx must be int32 or int64.

    ``tokens`` is a bfloat16 tensor or an FP8 ``MXFP8Tensor`` with compact (unswizzled) scales;
    only MXFP8 is supported for now. A bfloat16 input is quantized before dispatch when the
    buffer's ``dispatch_quant_recipe`` is set (currently ``MXFP8BlockScaling``); an
    ``MXFP8Tensor`` input is dispatched as-is.
    ``recv_tokens`` / ``recv_topk_weights`` are the dispatch recv outputs: pass caller-owned
    buffers (mcore-managed mode; in zero-copy they must be symm-mem-backed) or leave them None
    to allocate on the fly (zero-copy: symm-mem pool; normal: plain). Returns (recv_tokens,
    recv_topk_weights, token_counts); token_counts is non-diff. For FP8, recv_tokens is an
    ``MXFP8Tensor``.
    """
    from .quantized_tensor import QuantizedTensor

    _check_topk_weights_f32(topk_weights)

    # Quantize a bfloat16 input when the buffer requests FP8 dispatch.
    if not isinstance(tokens, QuantizedTensor) and buffer.dispatch_quant_recipe is not None:
        from ..common.recipe import MXFP8BlockScaling

        if not isinstance(buffer.dispatch_quant_recipe, MXFP8BlockScaling):
            raise NotImplementedError(
                "EP dispatch supports MXFP8BlockScaling only; got "
                f"{type(buffer.dispatch_quant_recipe).__name__}."
            )
        _require_bf16("tokens", tokens)
        from .quantization import get_fp8_te_dtype
        from .tensor.mxfp8_tensor import MXFP8Quantizer

        fp8_dtype = get_fp8_te_dtype(buffer.dispatch_quant_recipe, fprop_tensor=True)
        tokens = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False).quantize(tokens)

    tokens_scale_inv = None
    recv_scale_inv = None
    if isinstance(tokens, QuantizedTensor):
        # FP8 dispatch: route e4m3 data + compact e8m0 scales; recv is repacked as an
        # MXFP8Tensor. recv data/scale/weight buffers are one-sided write targets, so
        # pool-allocate (symm-mem) under zero-copy, else plain tensors.
        from .constants import MXFP8_BLOCK_SCALING_SIZE
        from .tensor.mxfp8_tensor import MXFP8Tensor

        if not isinstance(tokens, MXFP8Tensor):
            raise NotImplementedError(
                f"NCCL EP dispatch supports bfloat16 or MXFP8Tensor; got {type(tokens).__name__}."
            )
        if tokens._with_gemm_swizzled_scales:
            raise NotImplementedError(
                "NCCL EP dispatch requires compact (unswizzled) MXFP8 scales; "
                "quantize with with_gemm_swizzled_scales=False."
            )
        data = tokens._rowwise_data
        tokens_scale_inv = tokens._rowwise_scale_inv
        if data is None or tokens_scale_inv is None:
            raise ValueError("MXFP8 tokens must carry rowwise data and scale_inv for EP dispatch.")

        hidden = data.shape[-1]
        t_flat = data.numel() // hidden
        cols = hidden // MXFP8_BLOCK_SCALING_SIZE
        # The backend forwards each token's scale row with a 16-byte-aligned store, so
        # the row (cols * dtype bytes) must be a multiple of 16.
        scale_row_bytes = cols * tokens_scale_inv.element_size()
        if scale_row_bytes % 16 != 0:
            raise ValueError(
                f"MXFP8 dispatch requires a 16-byte-aligned scale row; hidden={hidden} gives "
                f"{scale_row_bytes} bytes. Use a hidden size that is a multiple of "
                f"{16 * MXFP8_BLOCK_SCALING_SIZE}."
            )
        # Strip GEMM row/col padding to the logical [T, H/block] the backend expects.
        tokens_scale_inv = tokens_scale_inv.reshape(tokens_scale_inv.shape[0], -1)[
            :t_flat, :cols
        ].contiguous()

        recv_pr = buffer.recv_capacity_per_rank
        if recv_tokens is None:
            recv_tokens = _alloc_io((recv_pr, hidden), data.dtype, buffer.device, buffer.zero_copy)
        recv_scale_inv = _alloc_io(
            (recv_pr, cols), tokens_scale_inv.dtype, buffer.device, buffer.zero_copy
        )
        # Dispatch doesn't clear unrouted recv rows; zero them so padding dequantizes to 0
        # rather than an fp8/e8m0 NaN.
        recv_tokens.zero_()
        recv_scale_inv.zero_()
    else:
        _require_bf16("tokens", tokens)
        if recv_tokens is None:
            recv_tokens = _alloc_io(
                (buffer.recv_capacity_per_rank, buffer.hidden_dim),
                buffer.payload_dtype,
                buffer.device,
                buffer.zero_copy,
            )

    if recv_topk_weights is None:
        recv_topk_weights = _alloc_io(
            (buffer.recv_capacity_per_rank,), torch.float32, buffer.device, buffer.zero_copy
        )
    # Pass the (possibly quantized) tokens as the autograd operand so grad reaches the
    # pre-quant input; tokens_scale_inv/recv_scale_inv are None for bf16 dispatch.
    return _EpDispatch.apply(
        buffer.handle_mem,
        buffer.top_k,
        buffer.alignment,
        recv_tokens,
        recv_topk_weights,
        buffer.token_counts,
        topk_idx,
        tokens,
        topk_weights,
        tokens_scale_inv,
        recv_scale_inv,
    )


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    grad_out: Optional[torch.Tensor] = None,
):
    """Combine with autograd; caller pre-applies topk weighting.

    ``expert_out`` is the combine input (always caller-supplied; in zero-copy it must be symm-mem-
    backed). ``grad_out`` is the backward's grad target: pass a caller-owned buffer (mcore-managed
    mode) or leave it None to allocate on the fly in the backward (zero-copy: symm-mem pool; normal:
    plain). Result shape is (num_local_tokens, buffer.hidden_dim); defaults to
    buffer.max_tokens_per_rank rows.
    """
    _require_bf16("expert_out", expert_out)
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    return _EpCombine.apply(
        buffer.handle_mem,
        num_local_tokens,
        buffer.hidden_dim,
        grad_out,
        expert_out,
    )
