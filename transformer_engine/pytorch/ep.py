# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
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


# -- Symmetric-memory buffer allocator ----------------------------------------


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


# -- Bootstrap ----------------------------------------------------------------


_BOOTSTRAPPED = False
_ATEXIT_REGISTERED = False
# Captured at ep_bootstrap and consumed implicitly by EpBuffer auto-alloc.
_ZERO_COPY: bool = False
_EP_GROUP: Optional[dist.ProcessGroup] = None


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown."""
    global _BOOTSTRAPPED, _ZERO_COPY, _EP_GROUP
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False
            _ZERO_COPY = False
            _EP_GROUP = None


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    max_num_sms: int = 0,
    allow_handle_mem_reloc: bool = False,
    zero_copy: bool = True,
) -> None:
    """Initialize EP by borrowing ``ep_group``'s NCCL comm. Call once per process.

    ``zero_copy`` is one-shot and also drives ``EpBuffer`` auto-alloc
    (symm-mem when True, plain HBM otherwise). Set ``allow_handle_mem_reloc=True``
    only if ``handle.handle_mem`` may move between fwd and bwd of the same layer.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED, _ZERO_COPY, _EP_GROUP
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
    tex.ep_set_zero_copy(bool(zero_copy))
    _BOOTSTRAPPED = True
    _ZERO_COPY = bool(zero_copy)
    _EP_GROUP = ep_group
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def ep_finalize() -> None:
    """Explicit teardown. Idempotent. Call before destroying the process group."""
    _atexit_finalize()


# -- Handle -------------------------------------------------------------------


class EpHandle:
    """Routing context for one EP layer. Construct once per concurrently-live
    microbatch (e.g. one per in-flight PP-1F1B step): a second forward on the
    same handle before the first's backward overwrites ``handle_mem`` and
    corrupts the earlier backward.
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


# -- Buffer -------------------------------------------------------------------


class EpBuffer:
    """Persistent payload + scratch buffers for one EP layer.

    Big recv-side slots (``recv_tokens`` = dispatch fwd output;
    ``recv_tokens_grad`` = combine bwd output) are opt-in: pass caller-allocated
    tensors to alias storage (e.g. the same tensor for both, or slabs from a
    shared symm-mem pool), or let the constructor auto-allocate. Auto-alloc
    uses symm-mem when ``ep_bootstrap(zero_copy=True)``, plain HBM otherwise.
    Per-rank scratch (``recv_topk_weights``, ``token_counts``,
    ``grad_topk_weights``) is always plain HBM.

    Use one ``EpBuffer`` per concurrently-in-flight call on the layer (one
    per PP-1F1B microbatch); sharing between an outstanding fwd and a later
    call overwrites tensors the earlier bwd still reads. Call
    :meth:`record_stream` from streams other than the allocation stream.
    """

    __slots__ = (
        "recv_tokens",
        "recv_tokens_grad",
        "recv_topk_weights",
        "token_counts",
        "grad_topk_weights",
    )

    def __init__(
        self,
        handle: EpHandle,
        *,
        recv_tokens: Optional[torch.Tensor] = None,
        recv_tokens_grad: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        recv_shape = (handle.recv_capacity_per_rank, handle.hidden_dim)
        self.recv_tokens = self._init_big_slot(
            recv_tokens, recv_shape, handle.payload_dtype, device, "recv_tokens"
        )
        self.recv_tokens_grad = self._init_big_slot(
            recv_tokens_grad, recv_shape, handle.payload_dtype, device, "recv_tokens_grad"
        )
        # Per-rank scratch - never cross-rank, plain HBM.
        self.recv_topk_weights = torch.empty(
            handle.recv_capacity_per_rank, dtype=torch.float32, device=device
        )
        self.token_counts = torch.empty(handle.num_local_experts, dtype=torch.int32, device=device)
        self.grad_topk_weights = torch.empty(
            (handle.max_tokens_per_rank, handle.top_k), dtype=torch.float32, device=device
        )

    @staticmethod
    def _init_big_slot(
        provided: Optional[torch.Tensor],
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if provided is None:
            if _ZERO_COPY:
                if _EP_GROUP is None:
                    raise RuntimeError(
                        f"EpBuffer auto-alloc of {name} as symm-mem requires ep_bootstrap "
                        "to have run; call ep_bootstrap first."
                    )
                return symm_mem_alloc(shape, dtype, _EP_GROUP, device=device)
            return torch.empty(shape, dtype=dtype, device=device)
        if tuple(provided.shape) != shape:
            raise ValueError(f"{name} shape {tuple(provided.shape)} != expected {shape}")
        if provided.dtype != dtype:
            raise ValueError(f"{name} dtype {provided.dtype} != expected {dtype}")
        if provided.device != device:
            raise ValueError(f"{name} device {provided.device} != expected {device}")
        return provided

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record ``stream`` as a user of all owned tensors so the caching allocator
        defers reclaim until ``stream`` has caught up."""
        for t in (
            self.recv_tokens,
            self.recv_tokens_grad,
            self.recv_topk_weights,
            self.token_counts,
            self.grad_topk_weights,
        ):
            t.record_stream(stream)


# -- torch.library custom ops (so they don't graph-break under torch.compile) -

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


# -- Non-autograd primitives --------------------------------------------------


def ep_prepare(handle: EpHandle, topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``handle.handle_mem`` and returns ``token_counts``
    (int32, shape ``[num_local_experts]``).

    ``topk_idx`` must be int64.
    """
    token_counts = torch.empty(handle.num_local_experts, dtype=torch.int32, device=handle.device)
    tex.ep_prepare(handle.handle_mem, handle.handle_id, topk_idx, token_counts, handle.alignment)
    return token_counts


def _ep_dispatch_raw(
    handle: EpHandle,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    """Raw dispatch - no autograd, no prepare. Caller must run ``ep_prepare`` first."""
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
    """Raw combine - no autograd, no weighting. Caller pre-weights ``expert_out``."""
    tex.ep_combine(handle.handle_mem, handle.handle_id, expert_out, result)


# -- autograd.Function wrappers -----------------------------------------------


class _EpDispatch(torch.autograd.Function):
    """Autograd-aware prepare + dispatch. Fwd/bwd share ``handle_mem`` and
    the ``EpBuffer`` slots; do not re-run ``ep_prepare`` between them and do
    not share ``EpBuffer`` with another in-flight call (see :class:`EpBuffer`).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        handle_id: int,
        alignment: int,
        recv_tokens: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        token_counts: torch.Tensor,
        grad_topk_weights_buf: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
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
        ctx.grad_topk_weights_buf = grad_topk_weights_buf
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype
        ctx.topk_weights_shape = topk_weights.shape
        ctx.topk_weights_dtype = topk_weights.dtype
        ctx.tokens_T_flat = tokens.numel() // tokens.shape[-1]
        ctx.topk_T_flat = topk_weights.numel() // topk_weights.shape[-1]
        ctx.recv_capacity = recv_tokens.shape[0]
        ctx.hidden_dim = tokens.shape[-1]
        ctx.mark_non_differentiable(token_counts)
        # Detach so the long-lived buffers aren't tracked as differentiable outputs;
        # autograd re-attaches grad_fn pointing back at this Function.
        return recv_tokens.detach(), recv_topk_weights.detach(), token_counts

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_token_counts):  # type: ignore[override]
        device = ctx.handle_mem.device
        if g_recv_tokens is None:
            g_recv_tokens = torch.zeros(
                ctx.recv_capacity, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
            )
        if g_recv_topk_weights is None:
            g_recv_topk_weights = torch.zeros(ctx.recv_capacity, dtype=torch.float32, device=device)
        if not g_recv_tokens.is_contiguous():
            g_recv_tokens = g_recv_tokens.contiguous()
        if not g_recv_topk_weights.is_contiguous():
            g_recv_topk_weights = g_recv_topk_weights.contiguous()
        # Send-side bwd output: alloc fresh plain HBM each call (not a persistent slot).
        grad_tokens = torch.empty(
            ctx.tokens_T_flat, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
        )
        grad_topk_weights = ctx.grad_topk_weights_buf.narrow(0, 0, ctx.topk_T_flat)
        torch.ops.transformer_engine_ep.dispatch_bwd(
            ctx.handle_mem,
            ctx.handle_id,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
        # Reshape back to the original input shape so autograd's grad slot matches.
        grad_tokens_out = grad_tokens.view(ctx.tokens_shape)
        grad_topk_weights_out = grad_topk_weights.view(ctx.topk_weights_shape)
        return (
            None,  # handle_mem
            None,  # handle_id
            None,  # alignment
            None,  # recv_tokens
            None,  # recv_topk_weights
            None,  # token_counts
            None,  # grad_topk_weights_buf
            None,  # topk_idx
            grad_tokens_out,
            grad_topk_weights_out,
        )


class _EpCombine(torch.autograd.Function):
    """Autograd-aware combine. bwd writes grad_expert_out into
    ``buffer.recv_tokens_grad`` (the slot passed at fwd time; aliasable to
    ``recv_tokens`` since their lifecycles don't overlap). fwd/bwd share
    ``handle_mem`` — do not re-run ``ep_prepare`` between them. Callers
    pre-multiply the topk weights into ``expert_out`` (see ep_combine).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        handle_id: int,
        recv_tokens_grad: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        expert_out: torch.Tensor,
    ):
        device = expert_out.device
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        # No staging copy: the combine collective reads expert_out directly.
        # Callers fold the topk weights into expert_out beforehand.
        torch.ops.transformer_engine_ep.combine(handle_mem, handle_id, expert_out, result)
        ctx.handle_mem = handle_mem
        ctx.handle_id = handle_id
        ctx.recv_tokens_grad = recv_tokens_grad
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        grad_expert_out = ctx.recv_tokens_grad
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        torch.ops.transformer_engine_ep.combine_bwd(
            ctx.handle_mem, ctx.handle_id, g_result, grad_expert_out
        )
        return (
            None,  # handle_mem
            None,  # handle_id
            None,  # recv_tokens_grad
            None,  # num_local_tokens
            None,  # hidden_dim
            grad_expert_out,  # grad wrt expert_out
        )


# -- Public high-level wrappers -----------------------------------------------


# FP8 dispatch is not yet supported by the common backend.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


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
):
    """Run prepare + dispatch with autograd. ``topk_idx`` must be int64.

    Returns ``(recv_tokens, recv_topk_weights, token_counts)`` - views into
    ``buffer``'s persistent slots; consume them before the next ``ep_dispatch``
    on the same buffer or they get overwritten. ``token_counts`` is
    non-differentiable.
    """
    _reject_fp8(tokens, buffer.recv_tokens)
    return _EpDispatch.apply(
        handle.handle_mem,
        handle.handle_id,
        handle.alignment,
        buffer.recv_tokens,
        buffer.recv_topk_weights,
        buffer.token_counts,
        buffer.grad_topk_weights,
        topk_idx,
        tokens,
        topk_weights,
    )


def ep_combine(
    handle: EpHandle,
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
):
    """Run combine with autograd. Callers must fold the topk weights into
    ``expert_out`` first (``expert_out = expert_out * topk_weight``); combine
    applies no weighting and passes ``expert_out`` straight to the collective.

    Result shape is ``(num_local_tokens, handle.hidden_dim)``, defaulting to
    ``handle.max_tokens_per_rank`` rows. bwd writes grad_expert_out into
    ``buffer.recv_tokens_grad``.
    """
    _reject_fp8(expert_out, buffer.recv_tokens_grad)
    if num_local_tokens is None:
        num_local_tokens = handle.max_tokens_per_rank
    return _EpCombine.apply(
        handle.handle_mem,
        handle.handle_id,
        buffer.recv_tokens_grad,
        num_local_tokens,
        handle.hidden_dim,
        expert_out,
    )
