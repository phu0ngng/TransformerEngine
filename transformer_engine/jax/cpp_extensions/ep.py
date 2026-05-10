# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for Expert Parallelism (EP).

Sharding note: every primitive's `partition` rule below is a placeholder.
It echoes the input shardings unchanged and declares replicated outputs
(`PartitionSpec(None)` / `PartitionSpec(None, None)`). The rules are
correct for the current single-process / no-mesh launch but are NOT a
production sharding plan. Once EpConfig-driven sharding (token_counts
reshape→[ep_size, nle], per-axis replication of handle_mem, etc.) is
designed, these rules need to be revisited end-to-end. Passing
non-replicated inputs today is unsupported and may produce silently
wrong NCCL EP routing.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax
from .base import BasePrimitive, register_primitive

__all__ = [
    "EpConfig",
    "set_ep_config",
    "get_ep_config",
    "get_ep_num_local_experts",
    "ep_prepare",
    "ep_dispatch_fwd",
    "ep_combine_fwd",
    "ep_dispatch_bwd",
    "ep_combine_bwd",
    "EpPreparePrimitive",
    "EpDispatchPrimitive",
    "EpCombinePrimitive",
    "EpDispatchBwdPrimitive",
    "EpCombineBwdPrimitive",
]


# ── Module-level EP config ──────────────────────────────────────────────────
#
# `EpConfig` mirrors the bootstrap surface (NVTEEpGroupConfig + world/rank).
# The C++ `EPBackend` is the runtime source of truth; this Python copy is
# read by JAX abstract-eval / sharding helpers that need the static shape
# metadata (`num_local_experts` for token_counts, `ep_size` for future
# token_counts reshape→[ep_size, nle] sharding patterns, etc.).
#
# TE-EP supports a single EP group per process — this config, the C++
# Meyers singleton, and `initialize_ep_communicator` all share the same
# single-group assumption. Calling `ep_bootstrap` twice in the same
# process is not supported.


@dataclass(frozen=True)
class EpConfig:
    """Immutable Python-side view of the EP bootstrap config.

    Today only `num_local_experts` is consumed by JAX abstract-eval (sizes
    `token_counts`); `ep_size`, `num_experts`, and the per-rank token bounds
    are recorded for the planned EpConfig-driven sharding rules
    (`token_counts` reshape→[ep_size, nle]) and for any future
    Python-side replay of the routing meta. `world_size`/`rank` round-trip
    the bootstrap signature and are reserved for multi-EP-group plans
    (not in scope today). The per-handle EM zone alignment knob is plumbed
    through `ep_prepare(...)` (NOT this bootstrap-time struct) since it is
    a per-prepare property of the NCCL EP handle.
    """

    world_size: int
    rank: int
    ep_size: int
    num_experts: int
    num_local_experts: int
    max_tokens_per_rank: int
    max_recv_tokens_per_rank: int
    hidden_dim: int


_ep_config: EpConfig = None


def set_ep_config(config: EpConfig) -> None:
    """Cache the EP config for use by abstract-eval / sharding helpers.

    Called once by `ep_bootstrap`. Must not be called twice — see
    `EpConfig` docstring for the single-group-per-process limitation.
    """
    global _ep_config
    _ep_config = config


def get_ep_config() -> EpConfig:
    if _ep_config is None:
        raise RuntimeError("EpConfig has not been set. Did you call ep_bootstrap()?")
    return _ep_config


def get_ep_num_local_experts() -> int:
    """Convenience accessor for `get_ep_config().num_local_experts`."""
    return get_ep_config().num_local_experts


# ── ep_prepare ──────────────────────────────────────────────────────────────────
# Inputs:  topk_idx [..., top_k] int64  (N-D; flattened in C++)
# Outputs: token_counts [num_local_experts] int32
#          handle_mem [handle_mem_size] uint8


class EpPreparePrimitive(BasePrimitive):
    name = "te_ep_prepare_ffi"
    multiple_results = True
    impl_static_args = (1,)  # dispatch_output_per_expert_alignment
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(topk_idx_aval, *, dispatch_output_per_expert_alignment):
        num_local_experts = get_ep_num_local_experts()
        assert (
            len(topk_idx_aval.shape) >= 2
        ), f"topk_idx must be at least 2D [..., top_k], got shape {topk_idx_aval.shape}"
        top_k = int(topk_idx_aval.shape[-1])
        handle_mem_size = transformer_engine_jax.get_ep_handle_mem_size(
            top_k,
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )
        token_counts_aval = jax.core.ShapedArray((num_local_experts,), jnp.int32)
        handle_mem_aval = jax.core.ShapedArray((handle_mem_size,), jnp.uint8)
        return token_counts_aval, handle_mem_aval

    @staticmethod
    def lowering(ctx, topk_idx, *, dispatch_output_per_expert_alignment):
        return ffi.ffi_lowering(EpPreparePrimitive.name)(
            ctx,
            topk_idx,
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )

    @staticmethod
    def impl(topk_idx, dispatch_output_per_expert_alignment):
        assert EpPreparePrimitive.inner_primitive is not None
        return EpPreparePrimitive.inner_primitive.bind(
            topk_idx,
            dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, dispatch_output_per_expert_alignment):
        raise NotImplementedError("EpPreparePrimitive does not support vmap")

    @staticmethod
    def partition(dispatch_output_per_expert_alignment, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = (arg_infos[0].sharding,)
        tc_sharding = NamedSharding(mesh, PartitionSpec(None))
        hm_sharding = NamedSharding(mesh, PartitionSpec(None))

        def sharded_impl(topk_idx):
            return EpPreparePrimitive.impl(topk_idx, dispatch_output_per_expert_alignment)

        return mesh, sharded_impl, (tc_sharding, hm_sharding), arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "T topk -> nle, hm"


register_primitive(EpPreparePrimitive)


# ── ep_dispatch ─────────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, topk_idx [..., top_k] int64,
#          tokens [..., H], topk_weights [..., top_k] float32
# Outputs: recv_tokens       [recv_capacity, H] (token dtype, always 2D)
#          recv_topk_weights [recv_capacity] float32 (always 1D, 1 weight per slot)


class EpDispatchPrimitive(BasePrimitive):
    name = "te_ep_dispatch_ffi"
    multiple_results = True  # (recv_tokens, recv_topk_weights)
    impl_static_args = (4, 5)  # recv_capacity, top_k
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval, topk_idx_aval, tokens_aval, topk_weights_aval, *, recv_capacity, top_k
    ):
        del handle_mem_aval, topk_idx_aval, topk_weights_aval, top_k
        assert (
            len(tokens_aval.shape) >= 2
        ), f"tokens must be at least 2D [..., H], got shape {tokens_aval.shape}"
        tok_dtype = dtypes.canonicalize_dtype(tokens_aval.dtype)
        hidden_dim = tokens_aval.shape[-1]
        recv_tokens_aval = jax.core.ShapedArray((recv_capacity, hidden_dim), tok_dtype)
        recv_topk_weights_aval = jax.core.ShapedArray((recv_capacity,), jnp.float32)
        return recv_tokens_aval, recv_topk_weights_aval

    @staticmethod
    def lowering(ctx, handle_mem, topk_idx, tokens, topk_weights, *, recv_capacity, top_k):
        return ffi.ffi_lowering(EpDispatchPrimitive.name)(
            ctx,
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            recv_capacity=recv_capacity,
            top_k=top_k,
        )

    @staticmethod
    def impl(handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k):
        assert EpDispatchPrimitive.inner_primitive is not None
        return EpDispatchPrimitive.inner_primitive.bind(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            recv_capacity=recv_capacity,
            top_k=top_k,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, recv_capacity, top_k):
        raise NotImplementedError("EpDispatchPrimitive does not support vmap")

    @staticmethod
    def partition(recv_capacity, top_k, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_shardings = (
            NamedSharding(mesh, PartitionSpec(None, None)),
            NamedSharding(mesh, PartitionSpec(None)),
        )

        def sharded_impl(handle_mem, topk_idx, tokens, topk_weights):
            return EpDispatchPrimitive.impl(
                handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, T topk_in, T H, T topk -> recv H, recv"


register_primitive(EpDispatchPrimitive)


# ── ep_combine ──────────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, expert_out [recv_capacity, H]   (ALWAYS 2D)
# Outputs: result [..., H]    (N-D; out_leading_shape passed by caller)
#
# NOTE: `expert_out` here is the *post-hadamard, masked* buffer, i.e. the
# caller's `expert_out * recv_topk_weights * mask` — NOT raw FFN output.
# The public-API `ep_combine` in transformer_engine/jax/ep.py applies the
# hadamard before calling this primitive. NCCL EP combine itself is an
# unweighted scatter-sum.
#
# Asymmetry vs ep_dispatch: dispatch accepts N-D `tokens` and flattens
# leading dims internally; combine REQUIRES 2D `expert_out`. The recv-side
# EM-grouped layout is intrinsically 2D and the FFN output keeps that shape.
#
# `recv_capacity` is implicit — read from expert_out.shape[0].
# `out_leading_shape` is a static tuple (e.g. (T,) or (B, S)) that determines
# the result's leading dims; the C++ FFI flattens it to a single row dim.


def _normalize_leading_shape(s):
    return s if isinstance(s, tuple) else (int(s),)


def _prod(seq):
    p = 1
    for x in seq:
        p *= int(x)
    return p


class EpCombinePrimitive(BasePrimitive):
    name = "te_ep_combine_ffi"
    multiple_results = False
    impl_static_args = (2,)  # out_leading_shape
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, expert_out_aval, *, out_leading_shape):
        del handle_mem_aval
        assert (
            len(expert_out_aval.shape) == 2
        ), f"expert_out must be 2D [recv_capacity, H], got shape {expert_out_aval.shape}"
        eo_dtype = dtypes.canonicalize_dtype(expert_out_aval.dtype)
        hidden_dim = expert_out_aval.shape[1]
        out_shape = tuple(out_leading_shape) + (hidden_dim,)
        return jax.core.ShapedArray(out_shape, eo_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, expert_out, *, out_leading_shape):
        return ffi.ffi_lowering(EpCombinePrimitive.name)(
            ctx,
            handle_mem,
            expert_out,
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(handle_mem, expert_out, out_leading_shape):
        assert EpCombinePrimitive.inner_primitive is not None
        return EpCombinePrimitive.inner_primitive.bind(
            handle_mem,
            expert_out,
            out_leading_shape=out_leading_shape,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_leading_shape):
        raise NotImplementedError("EpCombinePrimitive does not support vmap")

    @staticmethod
    def partition(out_leading_shape, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_spec = (None,) * (len(out_leading_shape) + 1)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        def sharded_impl(handle_mem, expert_out):
            return EpCombinePrimitive.impl(handle_mem, expert_out, out_leading_shape)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, recv H -> T H"


register_primitive(EpCombinePrimitive)


# ── ep_dispatch_bwd ─────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, grad [recv_capacity, H] (always 2D)
# Outputs: grad_tokens [..., H]   (N-D; out_leading_shape passed by caller)


class EpDispatchBwdPrimitive(BasePrimitive):
    name = "te_ep_dispatch_bwd_ffi"
    multiple_results = False
    impl_static_args = (2,)  # out_leading_shape
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, grad_aval, *, out_leading_shape):
        del handle_mem_aval
        assert (
            len(grad_aval.shape) == 2
        ), f"grad must be 2D [recv_capacity, H], got shape {grad_aval.shape}"
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[1]
        out_shape = tuple(out_leading_shape) + (hidden_dim,)
        return jax.core.ShapedArray(out_shape, g_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, grad, *, out_leading_shape):
        return ffi.ffi_lowering(EpDispatchBwdPrimitive.name)(
            ctx,
            handle_mem,
            grad,
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(handle_mem, grad, out_leading_shape):
        assert EpDispatchBwdPrimitive.inner_primitive is not None
        return EpDispatchBwdPrimitive.inner_primitive.bind(
            handle_mem,
            grad,
            out_leading_shape=out_leading_shape,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_leading_shape):
        raise NotImplementedError("EpDispatchBwdPrimitive does not support vmap")

    @staticmethod
    def partition(out_leading_shape, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_spec = (None,) * (len(out_leading_shape) + 1)
        out_sharding = NamedSharding(mesh, PartitionSpec(*out_spec))

        def sharded_impl(handle_mem, grad):
            return EpDispatchBwdPrimitive.impl(handle_mem, grad, out_leading_shape)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, recv H -> T H"


register_primitive(EpDispatchBwdPrimitive)


# ── ep_combine_bwd ──────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, grad [..., H]   (N-D, flattened in C++)
# Outputs: grad_expert_out [recv_capacity, H]    (always 2D)


class EpCombineBwdPrimitive(BasePrimitive):
    name = "te_ep_combine_bwd_ffi"
    multiple_results = False
    impl_static_args = (2,)  # recv_capacity
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, grad_aval, *, recv_capacity):
        del handle_mem_aval
        assert (
            len(grad_aval.shape) >= 2
        ), f"grad must be at least 2D [..., H], got shape {grad_aval.shape}"
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[-1]
        return jax.core.ShapedArray((recv_capacity, hidden_dim), g_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, grad, *, recv_capacity):
        return ffi.ffi_lowering(EpCombineBwdPrimitive.name)(
            ctx,
            handle_mem,
            grad,
            recv_capacity=recv_capacity,
        )

    @staticmethod
    def impl(handle_mem, grad, recv_capacity):
        assert EpCombineBwdPrimitive.inner_primitive is not None
        return EpCombineBwdPrimitive.inner_primitive.bind(
            handle_mem,
            grad,
            recv_capacity=recv_capacity,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, recv_capacity):
        raise NotImplementedError("EpCombineBwdPrimitive does not support vmap")

    @staticmethod
    def partition(recv_capacity, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(None, None))

        def sharded_impl(handle_mem, grad):
            return EpCombineBwdPrimitive.impl(handle_mem, grad, recv_capacity)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, T H -> recv H"


register_primitive(EpCombineBwdPrimitive)


# ── Public API wrappers ─────────────────────────────────────────────────────────


def ep_prepare(topk_idx, dispatch_output_per_expert_alignment=0):
    """Routing prep: AllGather + metadata exchange.

    Args:
      topk_idx: int64 [..., top_k] routing indices.
      dispatch_output_per_expert_alignment: per-handle EM zone alignment in
        tokens (pow2; 0/1 = no padding). Threaded through to NCCL EP's
        ncclEpInitHandle/ncclEpUpdateHandle as the per-handle padding knob.

    Returns (token_counts [num_local_experts] int32, handle_mem [N] uint8).
    """
    return EpPreparePrimitive.outer_primitive.bind(
        topk_idx,
        dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
    )


def ep_dispatch_fwd(handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k):
    """Scatter tokens and weights to expert ranks.

    Returns (recv_tokens [recv_capacity, H], recv_topk_weights [recv_capacity] f32).
    """
    return EpDispatchPrimitive.outer_primitive.bind(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        recv_capacity=recv_capacity,
        top_k=top_k,
    )


def ep_combine_fwd(handle_mem, expert_out, num_local_tokens):
    """Gather expert outputs back to home ranks.

    The hadamard product with `recv_topk_weights` happens in JAX before this
    call (see transformer_engine.jax.ep.ep_combine). recv_capacity is implicit
    in expert_out.shape[0].

    Args:
      num_local_tokens: int (legacy 2D output [T, H]) OR tuple of ints (N-D
        output [..., H]). Accepts both for backward-compat.
    """
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpCombinePrimitive.outer_primitive.bind(
        handle_mem,
        expert_out,
        out_leading_shape=out_leading,
    )


def ep_dispatch_bwd(handle_mem, grad, num_local_tokens):
    """Backward of dispatch (combine direction).

    Args:
      num_local_tokens: int (legacy 2D output) or tuple (N-D leading dims).
    """
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpDispatchBwdPrimitive.outer_primitive.bind(
        handle_mem,
        grad,
        out_leading_shape=out_leading,
    )


def ep_combine_bwd(handle_mem, grad, recv_capacity):
    """Backward of combine (dispatch direction).

    Returns grad_expert_out [recv_capacity, H].
    """
    return EpCombineBwdPrimitive.outer_primitive.bind(
        handle_mem,
        grad,
        recv_capacity=recv_capacity,
    )
