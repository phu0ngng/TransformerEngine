# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for Expert Parallelism (EP).

Sharding model (SPRINT7):
  - Outputs of EpPrepare/EpDispatch carry a leading `ep_size` dim and are
    sharded on `MeshResource.ep_resource`. Per-shard view at the FFI is the
    inner shape — each rank owns its own slice of the routing state.
  - EpDispatch requires `tokens` to be sharded on the leading dim by
    `dp_resource` or `fsdp_resource` only; hidden dim is replicated.
  - EpCombine consumes the 3D recv layout `[ep_size, recv_capacity_per_rank,
    H]` and resolves the output sharding via the public-API
    `out_sharding` kwarg (FSDP preferred if both DP and FSDP are set,
    otherwise the one that is set; raises if neither).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax
from .base import BasePrimitive, register_primitive
from ..sharding import global_mesh_resource

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
# Inputs:  topk_idx [..., top_k] int32 or int64 (N-D; flattened in C++; int32 is
#          upcast to int64 on-stream by the FFI).
# Outputs: token_counts [ep_size, num_local_experts] int32 (leading dim sharded on ep_resource)
#          handle_mem   [ep_size, handle_mem_size]    uint8 (leading dim sharded on ep_resource)
# Per-shard view at the FFI: token_counts [num_local_experts], handle_mem [N].


class EpPreparePrimitive(BasePrimitive):
    name = "te_ep_prepare_ffi"
    multiple_results = True
    impl_static_args = (1, 2)  # dispatch_output_per_expert_alignment, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(topk_idx_aval, *, dispatch_output_per_expert_alignment, is_outer):
        # is_outer=True (mesh-global view): token_counts [ep_size * NLE],
        # handle_mem [ep_size * N]. is_outer=False (per-shard, inside
        # custom_partitioning's sharded_impl): [NLE], [N].
        cfg = get_ep_config()
        ep_size = cfg.ep_size
        num_local_experts = cfg.num_local_experts
        assert (
            len(topk_idx_aval.shape) >= 2
        ), f"topk_idx must be at least 2D [..., top_k], got shape {topk_idx_aval.shape}"
        top_k = int(topk_idx_aval.shape[-1])
        handle_mem_size = transformer_engine_jax.get_ep_handle_mem_size(
            top_k,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
        )
        factor = ep_size if is_outer else 1
        token_counts_aval = jax.core.ShapedArray((factor * num_local_experts,), jnp.int32)
        handle_mem_aval = jax.core.ShapedArray((factor * handle_mem_size,), jnp.uint8)
        return token_counts_aval, handle_mem_aval

    @staticmethod
    def outer_abstract(topk_idx_aval, *, dispatch_output_per_expert_alignment, is_outer):
        del is_outer
        return EpPreparePrimitive.abstract(
            topk_idx_aval,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            is_outer=True,
        )

    @staticmethod
    def lowering(ctx, topk_idx, *, dispatch_output_per_expert_alignment, is_outer):
        del is_outer
        return ffi.ffi_lowering(EpPreparePrimitive.name)(
            ctx,
            topk_idx,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
        )

    @staticmethod
    def impl(topk_idx, dispatch_output_per_expert_alignment, is_outer):
        del is_outer
        assert EpPreparePrimitive.inner_primitive is not None
        return EpPreparePrimitive.inner_primitive.bind(
            topk_idx,
            dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
            is_outer=False,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, dispatch_output_per_expert_alignment, is_outer):
        raise NotImplementedError("EpPreparePrimitive does not support vmap")

    @staticmethod
    def partition(dispatch_output_per_expert_alignment, is_outer, mesh, arg_infos, result_infos):
        del is_outer
        del result_infos
        gsr = global_mesh_resource()
        ep_axis = gsr.ep_resource
        # topk_idx leading dim must be dp/fsdp-sharded; trailing dims replicated.
        idx_spec = arg_infos[0].sharding.spec
        allowed = {gsr.dp_resource, gsr.fsdp_resource}
        if (len(idx_spec) > 0 and idx_spec[0] not in allowed) or any(
            ax is not None for ax in idx_spec[1:]
        ):
            raise NotImplementedError(
                "EpPrepare: topk_idx leading dim must be sharded on dp_resource or"
                f" fsdp_resource (allowed={allowed}) and trailing dims replicated; got"
                f" spec={idx_spec}."
            )
        arg_shardings = (arg_infos[0].sharding,)
        tc_sharding = NamedSharding(mesh, PartitionSpec(ep_axis))
        hm_sharding = NamedSharding(mesh, PartitionSpec(ep_axis))

        def sharded_impl(topk_idx):
            return EpPreparePrimitive.impl(
                topk_idx, dispatch_output_per_expert_alignment, True
            )

        return mesh, sharded_impl, [tc_sharding, hm_sharding], arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "T topk -> ep_nle, ep_hm"


register_primitive(EpPreparePrimitive)


# ── ep_dispatch ─────────────────────────────────────────────────────────────────
# Inputs:  handle_mem [ep_size, N] uint8 (sharded on ep_resource),
#          topk_idx [..., top_k] int32 or int64 (leading dim sharded on dp/fsdp),
#          tokens [..., H] (leading dim sharded on dp/fsdp; hidden replicated),
#          topk_weights [..., top_k] float32 (matches topk_idx sharding)
# Outputs: recv_tokens       [ep_size, recv_capacity_per_rank, H] (sharded on ep_resource)
#          recv_topk_weights [ep_size, recv_capacity_per_rank]    (sharded on ep_resource)
# Per-shard view at the FFI: recv_tokens [recv_capacity_per_rank, H],
#                            recv_topk_weights [recv_capacity_per_rank].


class EpDispatchPrimitive(BasePrimitive):
    name = "te_ep_dispatch_ffi"
    multiple_results = True  # (recv_tokens, recv_topk_weights)
    impl_static_args = (4, 5, 6)  # recv_capacity, top_k, is_outer
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        handle_mem_aval,
        topk_idx_aval,
        tokens_aval,
        topk_weights_aval,
        *,
        recv_capacity,
        top_k,
        is_outer,
    ):
        del handle_mem_aval, topk_idx_aval, topk_weights_aval, top_k
        assert (
            len(tokens_aval.shape) >= 2
        ), f"tokens must be at least 2D [..., H], got shape {tokens_aval.shape}"
        # is_outer=True (global): recv_tokens [recv_capacity, H], recv_topk_weights
        # [recv_capacity]. is_outer=False (per-shard inside sharded_impl):
        # [recv_capacity/ep_size, H], [recv_capacity/ep_size].
        ep_size = get_ep_config().ep_size
        assert (
            recv_capacity % ep_size == 0
        ), f"recv_capacity ({recv_capacity}) must be divisible by ep_size ({ep_size})"
        leading = recv_capacity if is_outer else recv_capacity // ep_size
        tok_dtype = dtypes.canonicalize_dtype(tokens_aval.dtype)
        hidden_dim = tokens_aval.shape[-1]
        recv_tokens_aval = jax.core.ShapedArray((leading, hidden_dim), tok_dtype)
        recv_topk_weights_aval = jax.core.ShapedArray((leading,), jnp.float32)
        return recv_tokens_aval, recv_topk_weights_aval

    @staticmethod
    def outer_abstract(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["is_outer"] = True
        return EpDispatchPrimitive.abstract(*args, **kwargs)

    @staticmethod
    def lowering(
        ctx, handle_mem, topk_idx, tokens, topk_weights, *, recv_capacity, top_k, is_outer
    ):
        del is_outer
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
    def impl(handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k, is_outer):
        del is_outer
        assert EpDispatchPrimitive.inner_primitive is not None
        return EpDispatchPrimitive.inner_primitive.bind(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            recv_capacity=recv_capacity,
            top_k=top_k,
            is_outer=False,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, recv_capacity, top_k, is_outer):
        raise NotImplementedError("EpDispatchPrimitive does not support vmap")

    @staticmethod
    def partition(recv_capacity, top_k, is_outer, mesh, arg_infos, result_infos):
        del is_outer
        del result_infos
        gsr = global_mesh_resource()
        ep_axis = gsr.ep_resource
        # tokens (arg 2): leading dim must be sharded on dp/fsdp only; hidden replicated.
        tokens_spec = arg_infos[2].sharding.spec
        leading_axis = tokens_spec[0] if len(tokens_spec) > 0 else None
        allowed = {gsr.dp_resource, gsr.fsdp_resource}
        if leading_axis not in allowed:
            raise NotImplementedError(
                "EpDispatch: tokens leading dim must be sharded on dp_resource or"
                f" fsdp_resource (allowed={allowed}); got '{leading_axis}'."
            )
        for ax in tokens_spec[1:]:
            if ax is not None:
                raise NotImplementedError(
                    "EpDispatch: tokens non-leading dims must be replicated; got"
                    f" spec={tokens_spec}"
                )
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_shardings = [
            NamedSharding(mesh, PartitionSpec(ep_axis, None)),
            NamedSharding(mesh, PartitionSpec(ep_axis)),
        ]

        def sharded_impl(handle_mem, topk_idx, tokens, topk_weights):
            return EpDispatchPrimitive.impl(
                handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k, True
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "ep_hm, T topk_in, T H, T topk -> ep_recv_pr H, ep_recv_pr"


register_primitive(EpDispatchPrimitive)


# ── ep_combine ──────────────────────────────────────────────────────────────────
# Inputs:  handle_mem [ep_size, N] uint8 (sharded on ep_resource),
#          expert_out [ep_size, recv_capacity_per_rank, H] (sharded on ep_resource)
# Outputs: result [..., H]    (N-D; out_leading_shape passed by caller; leading
#                              dim sharded per out_partition_spec / MeshResource)
#
# NOTE: `expert_out` here is the *post-hadamard, masked* buffer, i.e. the
# caller's `expert_out * recv_topk_weights * mask`. The public-API `ep_combine`
# in transformer_engine/jax/ep.py applies the hadamard before calling this
# primitive.
#
# Per-shard view at the FFI: handle_mem [N], expert_out
# [recv_capacity_per_rank, H]. `recv_capacity_per_rank` is read from
# expert_out.shape[1].


def _normalize_leading_shape(s):
    return s if isinstance(s, tuple) else (int(s),)


def _prod(seq):
    p = 1
    for x in seq:
        p *= int(x)
    return p


def _resolve_out_partition_spec(out_partition_spec, num_leading):
    """Resolve combine output PartitionSpec axes.

    out_partition_spec: tuple of length num_leading + 1 (leading axes + hidden) or None.
      When None, leading axis is fsdp_resource if both dp/fsdp set; else whichever
      one is set; else raises. Trailing leading axes and hidden are replicated.
    """
    if out_partition_spec is not None:
        assert len(out_partition_spec) == num_leading + 1, (
            f"out_partition_spec length {len(out_partition_spec)} must equal num_leading"
            f" + 1 ({num_leading + 1})"
        )
        return tuple(out_partition_spec)
    gsr = global_mesh_resource()
    if gsr.fsdp_resource is not None and gsr.dp_resource is not None:
        leading = gsr.fsdp_resource
    elif gsr.fsdp_resource is not None:
        leading = gsr.fsdp_resource
    elif gsr.dp_resource is not None:
        leading = gsr.dp_resource
    else:
        raise ValueError(
            "ep_combine: neither dp_resource nor fsdp_resource is set on the active"
            " MeshResource; pass out_sharding=... explicitly."
        )
    return (leading,) + (None,) * num_leading  # leading axis + replicated rest


class EpCombinePrimitive(BasePrimitive):
    name = "te_ep_combine_ffi"
    multiple_results = False
    impl_static_args = (2, 3)  # out_leading_shape, out_partition_spec
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, expert_out_aval, *, out_leading_shape, out_partition_spec):
        del handle_mem_aval, out_partition_spec
        assert (
            len(expert_out_aval.shape) == 2
        ), f"expert_out must be 2D [recv_capacity, H], got shape {expert_out_aval.shape}"
        eo_dtype = dtypes.canonicalize_dtype(expert_out_aval.dtype)
        hidden_dim = expert_out_aval.shape[1]
        out_shape = tuple(out_leading_shape) + (hidden_dim,)
        return jax.core.ShapedArray(out_shape, eo_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, expert_out, *, out_leading_shape, out_partition_spec):
        del out_partition_spec
        return ffi.ffi_lowering(EpCombinePrimitive.name)(
            ctx,
            handle_mem,
            expert_out,
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(handle_mem, expert_out, out_leading_shape, out_partition_spec):
        assert EpCombinePrimitive.inner_primitive is not None
        return EpCombinePrimitive.inner_primitive.bind(
            handle_mem,
            expert_out,
            out_leading_shape=out_leading_shape,
            out_partition_spec=out_partition_spec,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_leading_shape, out_partition_spec):
        raise NotImplementedError("EpCombinePrimitive does not support vmap")

    @staticmethod
    def partition(out_leading_shape, out_partition_spec, mesh, arg_infos, result_infos):
        del result_infos
        gsr = global_mesh_resource()
        ep_axis = gsr.ep_resource
        # expert_out leading dim must be ep-sharded; trailing dims replicated.
        eo_spec = arg_infos[1].sharding.spec
        if (len(eo_spec) > 0 and eo_spec[0] != ep_axis) or any(
            ax is not None for ax in eo_spec[1:]
        ):
            raise NotImplementedError(
                "EpCombine: expert_out must be sharded on ep_resource on the leading"
                f" dim and replicated elsewhere; got spec={eo_spec}."
            )
        resolved = _resolve_out_partition_spec(out_partition_spec, len(out_leading_shape))
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*resolved))

        def sharded_impl(handle_mem, expert_out):
            return EpCombinePrimitive.impl(
                handle_mem, expert_out, out_leading_shape, out_partition_spec
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "ep_hm, ep_recv_pr H -> dp H"


register_primitive(EpCombinePrimitive)


# ── ep_dispatch_bwd ─────────────────────────────────────────────────────────────
# Inputs:  handle_mem [ep_size, N] uint8 (sharded on ep_resource),
#          grad [ep_size, recv_capacity_per_rank, H] (sharded on ep_resource)
# Outputs: grad_tokens [..., H]   (N-D; leading dim sharded per out_partition_spec)
# Per-shard view at FFI: handle_mem [N], grad [recv_capacity_per_rank, H].


class EpDispatchBwdPrimitive(BasePrimitive):
    name = "te_ep_dispatch_bwd_ffi"
    multiple_results = False
    impl_static_args = (2, 3)  # out_leading_shape, out_partition_spec
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, grad_aval, *, out_leading_shape, out_partition_spec):
        del handle_mem_aval, out_partition_spec
        assert (
            len(grad_aval.shape) == 2
        ), f"grad must be 2D [recv_capacity, H], got shape {grad_aval.shape}"
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[1]
        out_shape = tuple(out_leading_shape) + (hidden_dim,)
        return jax.core.ShapedArray(out_shape, g_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, grad, *, out_leading_shape, out_partition_spec):
        del out_partition_spec
        return ffi.ffi_lowering(EpDispatchBwdPrimitive.name)(
            ctx,
            handle_mem,
            grad,
            num_local_tokens=_prod(out_leading_shape),
        )

    @staticmethod
    def impl(handle_mem, grad, out_leading_shape, out_partition_spec):
        assert EpDispatchBwdPrimitive.inner_primitive is not None
        return EpDispatchBwdPrimitive.inner_primitive.bind(
            handle_mem,
            grad,
            out_leading_shape=out_leading_shape,
            out_partition_spec=out_partition_spec,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, out_leading_shape, out_partition_spec):
        raise NotImplementedError("EpDispatchBwdPrimitive does not support vmap")

    @staticmethod
    def partition(out_leading_shape, out_partition_spec, mesh, arg_infos, result_infos):
        del result_infos
        gsr = global_mesh_resource()
        ep_axis = gsr.ep_resource
        g_spec = arg_infos[1].sharding.spec
        if (len(g_spec) > 0 and g_spec[0] != ep_axis) or any(ax is not None for ax in g_spec[1:]):
            raise NotImplementedError(
                "EpDispatchBwd: grad must be sharded on ep_resource on the leading dim"
                f" and replicated elsewhere; got spec={g_spec}."
            )
        resolved = _resolve_out_partition_spec(out_partition_spec, len(out_leading_shape))
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(*resolved))

        def sharded_impl(handle_mem, grad):
            return EpDispatchBwdPrimitive.impl(
                handle_mem, grad, out_leading_shape, out_partition_spec
            )

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "ep_hm, ep_recv_pr H -> dp H"


register_primitive(EpDispatchBwdPrimitive)


# ── ep_combine_bwd ──────────────────────────────────────────────────────────────
# Inputs:  handle_mem [ep_size, N] uint8 (sharded on ep_resource),
#          grad [..., H]   (N-D; flattened in C++; leading dim sharded dp/fsdp)
# Outputs: grad_expert_out [ep_size, recv_capacity_per_rank, H] (sharded on ep_resource)
# Per-shard view at FFI: handle_mem [N], grad_expert_out [recv_capacity_per_rank, H].


class EpCombineBwdPrimitive(BasePrimitive):
    name = "te_ep_combine_bwd_ffi"
    multiple_results = False
    impl_static_args = (2,)  # recv_capacity (GLOBAL = ep_size * recv_capacity_per_rank)
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
        ep_size = get_ep_config().ep_size
        assert (
            recv_capacity % ep_size == 0
        ), f"recv_capacity ({recv_capacity}) must be divisible by ep_size ({ep_size})"
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
        gsr = global_mesh_resource()
        ep_axis = gsr.ep_resource
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(ep_axis, None))

        def sharded_impl(handle_mem, grad):
            return EpCombineBwdPrimitive.impl(handle_mem, grad, recv_capacity)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "ep_hm, T H -> ep_recv_pr H"


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
        dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
        is_outer=True,
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
        is_outer=True,
    )


def ep_combine_fwd(handle_mem, expert_out, num_local_tokens, out_partition_spec=None):
    """Gather expert outputs back to home ranks.

    The hadamard product with `recv_topk_weights` happens in JAX before this
    call (see transformer_engine.jax.ep.ep_combine). `expert_out` is
    [ep_size, recv_capacity_per_rank, H] (sharded on ep_resource).

    Args:
      num_local_tokens: int → 2D output [T, H]. Tuple of ints → N-D output.
      out_partition_spec: optional tuple of axis names (length = num_leading + 1)
        used to build the output `PartitionSpec`. When None, the leading axis is
        resolved from MeshResource (FSDP preferred if both DP and FSDP are set;
        else whichever is set; else raises).
    """
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpCombinePrimitive.outer_primitive.bind(
        handle_mem,
        expert_out,
        out_leading_shape=out_leading,
        out_partition_spec=out_partition_spec,
    )


def ep_dispatch_bwd(handle_mem, grad, num_local_tokens, out_partition_spec=None):
    """Backward of dispatch (combine direction). Same out_partition_spec semantics as ep_combine_fwd."""
    out_leading = _normalize_leading_shape(num_local_tokens)
    return EpDispatchBwdPrimitive.outer_primitive.bind(
        handle_mem,
        grad,
        out_leading_shape=out_leading,
        out_partition_spec=out_partition_spec,
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
