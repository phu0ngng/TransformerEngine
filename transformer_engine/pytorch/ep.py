# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
import contextlib
from typing import Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex


__all__ = [
    "EpHandle",
    "EpBuffer",
    "ep_bootstrap",
    "ep_finalize",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
]


# ── Symmetric-memory buffer allocator ────────────────────────────────────────


def symm_mem_alloc(
    shape,
    dtype: torch.dtype,
    ep_group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Allocate and rendezvous a symm-mem buffer on ``ep_group`` for the EP zero-copy path.

    Collective on ``ep_group``.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        from torch.distributed import _symmetric_memory as _symm_mem
    except ImportError as e:
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable; symm_mem_alloc "
            "requires PyTorch built with NCCL symm-mem support."
        ) from e
    if _symm_mem.get_backend(device) != "NCCL":
        _symm_mem.set_backend("NCCL")
    t = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(t, group=ep_group.group_name)
    return t


# ── Bootstrap ────────────────────────────────────────────────────────────────


_BOOTSTRAPPED = False
_ATEXIT_REGISTERED = False


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    max_num_sms: int = 0,
    allow_handle_mem_reloc: bool = False,
) -> None:
    """Initialize EP by borrowing ``ep_group``'s NCCL comm. Call once per process.

    Set ``allow_handle_mem_reloc=True`` only if ``handle.handle_mem`` may move
    between fwd and bwd of the same layer (off by default).
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")

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
        bool(allow_handle_mem_reloc),
    )
    _BOOTSTRAPPED = True
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def ep_finalize() -> None:
    """Explicit teardown. Idempotent. Call before destroying the process group."""
    _atexit_finalize()


# ── Handle ───────────────────────────────────────────────────────────────────


class EpHandle:
    """Routing context for one EP layer. Construct once at module init; reuse per step.

    Single-use per step: do not share across concurrently in-flight
    ``ep_dispatch`` / ``ep_combine`` calls (e.g. PP-1F1B microbatches).
    """

    __slots__ = (
        "handle_mem",
        "handle_id",
        "top_k",
        "alignment",
        "max_tokens_per_rank",
        "recv_capacity_per_rank",
        "hidden_dim",
        "num_local_experts",
        "payload_dtype",
        "device",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        alignment: int = 0,
        device: Optional[torch.device] = None,
        payload_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.top_k = int(top_k)
        self.alignment = int(alignment)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.recv_capacity_per_rank = int(recv_capacity_per_rank)
        self.hidden_dim = int(hidden_dim)
        self.num_local_experts = int(num_local_experts)
        self.payload_dtype = payload_dtype
        self.device = device
        handle_id, size_bytes = tex.ep_register_layer(self.top_k, self.alignment)
        self.handle_id = int(handle_id)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)


# ── Buffer ───────────────────────────────────────────────────────────────────


class EpBuffer:
    """Symm-mem-backed payload buffers (``recv_tokens``, ``combine_in``) for one EP layer.

    Construct once at layer init (collective rendezvous on ``ep_group``).
    ``use_symm_mem=False`` falls back to plain HBM for debug runs.

    Multi-stream usage: call :meth:`record_stream` from every stream that
    touches the buffer outside its allocation stream, otherwise the caching
    allocator can reclaim memory that peers' symm-mem windows still point at.
    """

    __slots__ = ("recv_tokens", "combine_in")

    def __init__(
        self,
        handle: EpHandle,
        ep_group: Optional[dist.ProcessGroup] = None,
        *,
        use_symm_mem: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        shape = (handle.recv_capacity_per_rank, handle.hidden_dim)
        if use_symm_mem:
            if ep_group is None:
                raise ValueError("EpBuffer(use_symm_mem=True) requires ep_group.")
            self.recv_tokens = symm_mem_alloc(shape, handle.payload_dtype, ep_group, device=device)
            self.combine_in = symm_mem_alloc(shape, handle.payload_dtype, ep_group, device=device)
        else:
            self.recv_tokens = torch.empty(shape, dtype=handle.payload_dtype, device=device)
            self.combine_in = torch.empty(shape, dtype=handle.payload_dtype, device=device)

    @classmethod
    def from_external(
        cls,
        handle: EpHandle,
        *,
        recv_tokens: torch.Tensor,
        combine_in: torch.Tensor,
    ) -> "EpBuffer":
        """Construct from caller-allocated buffers (e.g. slices of a shared symm-mem pool).

        Both tensors must have shape ``(handle.recv_capacity_per_rank, handle.hidden_dim)``
        and dtype ``handle.payload_dtype``.
        """
        expected = (handle.recv_capacity_per_rank, handle.hidden_dim)
        if tuple(recv_tokens.shape) != expected:
            raise ValueError(f"recv_tokens shape {tuple(recv_tokens.shape)} != expected {expected}")
        if tuple(combine_in.shape) != expected:
            raise ValueError(f"combine_in shape {tuple(combine_in.shape)} != expected {expected}")
        if recv_tokens.dtype != handle.payload_dtype or combine_in.dtype != handle.payload_dtype:
            raise ValueError(
                f"buffer dtype must match handle.payload_dtype ({handle.payload_dtype})"
            )
        inst = cls.__new__(cls)
        inst.recv_tokens = recv_tokens
        inst.combine_in = combine_in
        return inst

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record ``stream`` as a user of both owned tensors so the caching allocator
        defers reclaim until ``stream`` has caught up."""
        self.recv_tokens.record_stream(stream)
        self.combine_in.record_stream(stream)


# ── torch.library custom ops (so they don't graph-break under torch.compile) ─

_LIB = "transformer_engine_ep"


@torch.library.custom_op(
    f"{_LIB}::prepare",
    mutates_args=("handle_mem", "token_counts"),
    device_types="cuda",
)
def _prepare_op(
    handle_mem: torch.Tensor,
    handle_id: int,
    topk_idx: torch.Tensor,
    token_counts: torch.Tensor,
    alignment: int,
) -> None:
    tex.ep_prepare(handle_mem, handle_id, topk_idx, token_counts, alignment)


@_prepare_op.register_fake
def _(*args, **kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch",
    mutates_args=("recv_tokens", "recv_topk_weights"),
    device_types="cuda",
)
def _dispatch_op(
    handle_mem: torch.Tensor,
    handle_id: int,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch(
        handle_mem,
        handle_id,
        topk_idx,
        tokens,
        topk_weights,
        recv_tokens,
        recv_topk_weights,
    )


@_dispatch_op.register_fake
def _(*args, **kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine",
    mutates_args=("result",),
    device_types="cuda",
)
def _combine_op(
    handle_mem: torch.Tensor,
    handle_id: int,
    expert_out: torch.Tensor,
    result: torch.Tensor,
) -> None:
    tex.ep_combine(handle_mem, handle_id, expert_out, result)


@_combine_op.register_fake
def _(*args, **kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch_bwd",
    mutates_args=("grad_tokens", "grad_topk_weights"),
    device_types="cuda",
)
def _dispatch_bwd_op(
    handle_mem: torch.Tensor,
    handle_id: int,
    grad: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch_bwd(
        handle_mem, handle_id, grad, g_recv_topk_weights, grad_tokens, grad_topk_weights
    )


@_dispatch_bwd_op.register_fake
def _(*args, **kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine_bwd",
    mutates_args=("grad_expert_out",),
    device_types="cuda",
)
def _combine_bwd_op(
    handle_mem: torch.Tensor,
    handle_id: int,
    grad: torch.Tensor,
    grad_expert_out: torch.Tensor,
) -> None:
    tex.ep_combine_bwd(handle_mem, handle_id, grad, grad_expert_out)


@_combine_bwd_op.register_fake
def _(*args, **kw):
    return None


# ── Non-autograd primitives ──────────────────────────────────────────────────


def ep_prepare(handle: EpHandle, topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``handle.handle_mem`` and returns ``token_counts``
    (int32, shape ``[num_local_experts]``).

    ``topk_idx`` must be int64.
    """
    token_counts = torch.empty(
        handle.num_local_experts, dtype=torch.int32, device=handle.device
    )
    tex.ep_prepare(
        handle.handle_mem, handle.handle_id, topk_idx, token_counts, handle.alignment
    )
    return token_counts


def _ep_dispatch_raw(
    handle: EpHandle,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    """Raw dispatch — no autograd, no prepare. Caller must run ``ep_prepare`` first."""
    tex.ep_dispatch(
        handle.handle_mem,
        handle.handle_id,
        topk_idx,
        tokens,
        topk_weights,
        recv_tokens,
        recv_topk_weights,
    )


def _ep_combine_raw(handle: EpHandle, expert_out: torch.Tensor, result: torch.Tensor) -> None:
    """Raw combine — no autograd, no weighting. Caller pre-weights ``expert_out``."""
    tex.ep_combine(handle.handle_mem, handle.handle_id, expert_out, result)


# ── autograd.Function wrappers ───────────────────────────────────────────────


class _EpDispatch(torch.autograd.Function):
    """Autograd-aware prepare + dispatch. Fwd and bwd share ``handle_mem``;
    do not re-run ``ep_prepare`` on this handle between them."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        handle_id: int,
        alignment: int,
        recv_tokens: torch.Tensor,
        num_local_experts: int,
        zero_copy: bool,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        device = tokens.device
        recv_capacity = recv_tokens.shape[0]
        token_counts = torch.empty(num_local_experts, dtype=torch.int32, device=device)
        recv_topk_weights = torch.empty(recv_capacity, dtype=torch.float32, device=device)
        with _zero_copy_scope(zero_copy):
            torch.ops.transformer_engine_ep.prepare(
                handle_mem, handle_id, topk_idx, token_counts, alignment
            )
            torch.ops.transformer_engine_ep.dispatch(
                handle_mem,
                handle_id,
                topk_idx,
                tokens,
                topk_weights,
                recv_tokens,
                recv_topk_weights,
            )
        ctx.handle_mem = handle_mem
        ctx.handle_id = handle_id
        ctx.zero_copy = zero_copy
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype
        ctx.topk_weights_shape = topk_weights.shape
        ctx.topk_weights_dtype = topk_weights.dtype
        ctx.recv_capacity = recv_capacity
        ctx.hidden_dim = tokens.shape[-1]
        ctx.mark_non_differentiable(token_counts)
        # Detach so the long-lived buffer isn't tracked as a differentiable output.
        return recv_tokens.detach(), recv_topk_weights, token_counts

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_token_counts):  # type: ignore[override]
        device = ctx.handle_mem.device
        if g_recv_tokens is None:
            g_recv_tokens = torch.zeros(
                ctx.recv_capacity, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
            )
        if g_recv_topk_weights is None:
            g_recv_topk_weights = torch.zeros(ctx.recv_capacity, dtype=torch.float32, device=device)
        grad_tokens = torch.empty(ctx.tokens_shape, dtype=ctx.tokens_dtype, device=device)
        grad_topk_weights = torch.empty(
            ctx.topk_weights_shape, dtype=ctx.topk_weights_dtype, device=device
        )
        if not g_recv_tokens.is_contiguous():
            g_recv_tokens = g_recv_tokens.contiguous()
        if not g_recv_topk_weights.is_contiguous():
            g_recv_topk_weights = g_recv_topk_weights.contiguous()
        with _zero_copy_scope(ctx.zero_copy):
            torch.ops.transformer_engine_ep.dispatch_bwd(
                ctx.handle_mem,
                ctx.handle_id,
                g_recv_tokens,
                g_recv_topk_weights,
                grad_tokens,
                grad_topk_weights,
            )
        return (
            None,  # handle_mem
            None,  # handle_id
            None,  # alignment
            None,  # recv_tokens
            None,  # num_local_experts
            None,  # zero_copy
            None,  # topk_idx
            grad_tokens,
            grad_topk_weights,
        )


@torch.compile(dynamic=False, fullgraph=True)
def _combine_bwd_post(grad_combine_in, expert_out, recv_topk_weights):
    """Fused post-NCCL-combine_bwd: shares grad_combine_in reads across the two muls."""
    w = recv_topk_weights.unsqueeze(-1).to(expert_out.dtype)
    grad_expert_out = grad_combine_in * w
    grad_recv_topk_weights = (
        (grad_combine_in * expert_out).sum(-1, dtype=torch.float32).to(recv_topk_weights.dtype)
    )
    return grad_expert_out, grad_recv_topk_weights


class _EpCombine(torch.autograd.Function):
    """Autograd-aware weight + combine. Fwd and bwd share ``handle_mem``;
    do not re-run ``ep_prepare`` on this handle between them."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        handle_id: int,
        combine_in: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        zero_copy: bool,
        expert_out: torch.Tensor,
        recv_topk_weights: torch.Tensor,
    ):
        device = expert_out.device
        # Weight in payload dtype: single fused broadcast multiply into combine_in.
        w = recv_topk_weights.unsqueeze(-1).to(expert_out.dtype)
        torch.mul(expert_out, w, out=combine_in)
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        with _zero_copy_scope(zero_copy):
            torch.ops.transformer_engine_ep.combine(handle_mem, handle_id, combine_in, result)
        ctx.save_for_backward(expert_out, recv_topk_weights)
        ctx.handle_mem = handle_mem
        ctx.handle_id = handle_id
        ctx.zero_copy = zero_copy
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        expert_out, recv_topk_weights = ctx.saved_tensors
        grad_combine_in = torch.empty_like(expert_out)
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        with _zero_copy_scope(ctx.zero_copy):
            torch.ops.transformer_engine_ep.combine_bwd(
                ctx.handle_mem, ctx.handle_id, g_result, grad_combine_in
            )
        grad_expert_out, grad_recv_topk_weights = _combine_bwd_post(
            grad_combine_in, expert_out, recv_topk_weights
        )
        return (
            None,  # handle_mem
            None,  # handle_id
            None,  # combine_in
            None,  # num_local_tokens
            None,  # hidden_dim
            None,  # zero_copy
            grad_expert_out,
            grad_recv_topk_weights,
        )


# ── Public high-level wrappers ───────────────────────────────────────────────


# FP8 dispatch is not yet supported by the common backend.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# Python-side mirror of the C++ g_zero_copy_enabled atomic. Lets _zero_copy_scope
# skip a pybind getter on every per-step op (the common case is enabled==prev).
_ZC_ENABLED = [True]


@contextlib.contextmanager
def _zero_copy_scope(enabled: bool):
    """Set the symm-mem zero-copy toggle for the scope, saving + restoring the prior value.

    No-op under ``torch.compile`` (pybind getter/setter aren't dynamo-traceable);
    callers must pre-set the global toggle before entering the compiled region.
    """
    if torch.compiler.is_compiling():
        yield
        return
    prev = _ZC_ENABLED[0]
    if prev == enabled:
        yield
        return
    tex.ep_set_zero_copy(enabled)
    _ZC_ENABLED[0] = enabled
    try:
        yield
    finally:
        tex.ep_set_zero_copy(prev)
        _ZC_ENABLED[0] = prev


def _reject_fp8(*tensors: torch.Tensor) -> None:
    if torch.compiler.is_compiling():
        return
    for t in tensors:
        if t.dtype in _FP8_DTYPES:
            raise NotImplementedError(
                f"FP8 dispatch/combine not supported (got dtype={t.dtype}); "
                "quantize outside the EP boundary."
            )


def ep_dispatch(
    handle: EpHandle,
    buffer: EpBuffer,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    zero_copy: bool = True,
):
    """Run prepare + dispatch with autograd. ``topk_idx`` must be int64.

    Returns ``(recv_tokens, recv_topk_weights, token_counts)``;
    ``token_counts`` is non-differentiable.
    """
    _reject_fp8(tokens, buffer.recv_tokens)
    return _EpDispatch.apply(
        handle.handle_mem,
        handle.handle_id,
        handle.alignment,
        buffer.recv_tokens,
        handle.num_local_experts,
        zero_copy,
        topk_idx,
        tokens,
        topk_weights,
    )


def ep_combine(
    handle: EpHandle,
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    zero_copy: bool = True,
):
    """Apply per-slot weighting then combine, with autograd.

    Result shape is ``(num_local_tokens, handle.hidden_dim)``; defaults to
    ``handle.max_tokens_per_rank`` rows.
    """
    _reject_fp8(expert_out, buffer.combine_in)
    if num_local_tokens is None:
        num_local_tokens = handle.max_tokens_per_rank
    return _EpCombine.apply(
        handle.handle_mem,
        handle.handle_id,
        buffer.combine_in,
        num_local_tokens,
        handle.hidden_dim,
        zero_copy,
        expert_out,
        recv_topk_weights,
    )
