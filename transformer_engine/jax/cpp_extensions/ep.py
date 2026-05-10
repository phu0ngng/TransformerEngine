# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX/TE custom ops for Expert Parallelism (EP)"""

import jax
import jax.numpy as jnp
from jax import dtypes, ffi
from jax.sharding import NamedSharding, PartitionSpec

import transformer_engine_jax
from .base import BasePrimitive, register_primitive

__all__ = [
    "set_ep_num_local_experts",
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


# ── Module-level cache of num_local_experts ─────────────────────────────────
#
# `num_local_experts` is no longer threaded through `ep_prepare` — it is
# already cached on the C++ side at nvte_ep_initialize() time (derived from
# group_config.num_experts / group_config.ep_size). The high-level
# ep_bootstrap() in transformer_engine.jax.ep stores it here so the
# JAX abstract-eval rule can use it to size the token_counts output without
# adding a parameter back to the primitive.

_ep_num_local_experts: int = 0


def set_ep_num_local_experts(n: int) -> None:
    """Cache the per-rank local-expert count for use by EpPreparePrimitive.abstract.

    NOTE: TE-EP supports a single EP group per process. This cache, the C++
    `EPBackend` Meyers singleton, and `transformer_engine_jax.initialize_ep_communicator`
    all share the same single-group assumption — calling `ep_bootstrap` twice in
    the same process is not supported.
    """
    global _ep_num_local_experts
    _ep_num_local_experts = int(n)


def get_ep_num_local_experts() -> int:
    if _ep_num_local_experts <= 0:
        raise RuntimeError(
            "ep_num_local_experts has not been set. Did you call ep_bootstrap()?"
        )
    return _ep_num_local_experts


# ── ep_prepare ──────────────────────────────────────────────────────────────────
# Inputs:  topk_idx [T, top_k] int64
# Outputs: token_counts [num_local_experts] int32
#          handle_mem [handle_mem_size] uint8


class EpPreparePrimitive(BasePrimitive):
    name = "te_ep_prepare_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(topk_idx_aval):
        num_local_experts = get_ep_num_local_experts()
        top_k = int(topk_idx_aval.shape[1])
        handle_mem_size = transformer_engine_jax.get_ep_handle_mem_size(top_k)
        token_counts_aval = jax.core.ShapedArray((num_local_experts,), jnp.int32)
        handle_mem_aval = jax.core.ShapedArray((handle_mem_size,), jnp.uint8)
        return token_counts_aval, handle_mem_aval

    @staticmethod
    def lowering(ctx, topk_idx):
        return ffi.ffi_lowering(EpPreparePrimitive.name)(ctx, topk_idx)

    @staticmethod
    def impl(topk_idx):
        assert EpPreparePrimitive.inner_primitive is not None
        return EpPreparePrimitive.inner_primitive.bind(topk_idx)

    @staticmethod
    def batcher(batched_args, batch_dims):
        raise NotImplementedError("EpPreparePrimitive does not support vmap")

    @staticmethod
    def partition(mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = (arg_infos[0].sharding,)
        tc_sharding = NamedSharding(mesh, PartitionSpec(None))
        hm_sharding = NamedSharding(mesh, PartitionSpec(None))

        def sharded_impl(topk_idx):
            return EpPreparePrimitive.impl(topk_idx)

        return mesh, sharded_impl, (tc_sharding, hm_sharding), arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "T topk -> nle, hm"


register_primitive(EpPreparePrimitive)


# ── ep_dispatch ─────────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, topk_idx [T, top_k] int64,
#          tokens [T, H], topk_weights [T, top_k] float32
# Outputs: recv_tokens       [recv_capacity, H] (token dtype)
#          recv_topk_weights [recv_capacity] float32  (HT+EM: 1 weight per slot)


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
        tok_dtype = dtypes.canonicalize_dtype(tokens_aval.dtype)
        hidden_dim = tokens_aval.shape[1]
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
# Inputs:  handle_mem [N] uint8, expert_out [recv_capacity, H]
# Outputs: result [num_local_tokens, H]
#
# NOTE: `expert_out` here is the *post-hadamard, masked* buffer, i.e. the
# caller's `expert_out * recv_topk_weights * mask` — NOT raw FFN output.
# The public-API `ep_combine` in transformer_engine/jax/ep.py applies the
# hadamard before calling this primitive. NCCL EP combine itself is an
# unweighted scatter-sum.
#
# `recv_capacity` is implicit — read from expert_out.shape[0].


class EpCombinePrimitive(BasePrimitive):
    name = "te_ep_combine_ffi"
    multiple_results = False
    impl_static_args = (2,)  # num_local_tokens
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, expert_out_aval, *, num_local_tokens):
        del handle_mem_aval
        eo_dtype = dtypes.canonicalize_dtype(expert_out_aval.dtype)
        hidden_dim = expert_out_aval.shape[1]
        return jax.core.ShapedArray((num_local_tokens, hidden_dim), eo_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, expert_out, *, num_local_tokens):
        return ffi.ffi_lowering(EpCombinePrimitive.name)(
            ctx, handle_mem, expert_out, num_local_tokens=num_local_tokens,
        )

    @staticmethod
    def impl(handle_mem, expert_out, num_local_tokens):
        assert EpCombinePrimitive.inner_primitive is not None
        return EpCombinePrimitive.inner_primitive.bind(
            handle_mem, expert_out, num_local_tokens=num_local_tokens,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, num_local_tokens):
        raise NotImplementedError("EpCombinePrimitive does not support vmap")

    @staticmethod
    def partition(num_local_tokens, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(None, None))

        def sharded_impl(handle_mem, expert_out):
            return EpCombinePrimitive.impl(handle_mem, expert_out, num_local_tokens)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, recv H -> T H"


register_primitive(EpCombinePrimitive)


# ── ep_dispatch_bwd ─────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, grad [recv_capacity, H]
# Outputs: grad_tokens [num_local_tokens, H]


class EpDispatchBwdPrimitive(BasePrimitive):
    name = "te_ep_dispatch_bwd_ffi"
    multiple_results = False
    impl_static_args = (2,)  # num_local_tokens
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, grad_aval, *, num_local_tokens):
        del handle_mem_aval
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[1]
        return jax.core.ShapedArray((num_local_tokens, hidden_dim), g_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, grad, *, num_local_tokens):
        return ffi.ffi_lowering(EpDispatchBwdPrimitive.name)(
            ctx, handle_mem, grad, num_local_tokens=num_local_tokens,
        )

    @staticmethod
    def impl(handle_mem, grad, num_local_tokens):
        assert EpDispatchBwdPrimitive.inner_primitive is not None
        return EpDispatchBwdPrimitive.inner_primitive.bind(
            handle_mem, grad, num_local_tokens=num_local_tokens,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, num_local_tokens):
        raise NotImplementedError("EpDispatchBwdPrimitive does not support vmap")

    @staticmethod
    def partition(num_local_tokens, mesh, arg_infos, result_infos):
        del result_infos
        arg_shardings = tuple(a.sharding for a in arg_infos)
        out_sharding = NamedSharding(mesh, PartitionSpec(None, None))

        def sharded_impl(handle_mem, grad):
            return EpDispatchBwdPrimitive.impl(handle_mem, grad, num_local_tokens)

        return mesh, sharded_impl, out_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args):
        del args
        return "hm, recv H -> T H"


register_primitive(EpDispatchBwdPrimitive)


# ── ep_combine_bwd ──────────────────────────────────────────────────────────────
# Inputs:  handle_mem [N] uint8, grad [num_local_tokens, H]
# Outputs: grad_expert_out [recv_capacity, H]


class EpCombineBwdPrimitive(BasePrimitive):
    name = "te_ep_combine_bwd_ffi"
    multiple_results = False
    impl_static_args = (2,)  # recv_capacity
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(handle_mem_aval, grad_aval, *, recv_capacity):
        del handle_mem_aval
        g_dtype = dtypes.canonicalize_dtype(grad_aval.dtype)
        hidden_dim = grad_aval.shape[1]
        return jax.core.ShapedArray((recv_capacity, hidden_dim), g_dtype)

    @staticmethod
    def lowering(ctx, handle_mem, grad, *, recv_capacity):
        return ffi.ffi_lowering(EpCombineBwdPrimitive.name)(
            ctx, handle_mem, grad, recv_capacity=recv_capacity,
        )

    @staticmethod
    def impl(handle_mem, grad, recv_capacity):
        assert EpCombineBwdPrimitive.inner_primitive is not None
        return EpCombineBwdPrimitive.inner_primitive.bind(
            handle_mem, grad, recv_capacity=recv_capacity,
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


def ep_prepare(topk_idx):
    """Routing prep: AllGather + metadata exchange.

    Returns (token_counts [num_local_experts] int32, handle_mem [N] uint8).
    """
    return EpPreparePrimitive.outer_primitive.bind(topk_idx)


def ep_dispatch_fwd(handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k):
    """Scatter tokens and weights to expert ranks.

    Returns (recv_tokens [recv_capacity, H], recv_topk_weights [recv_capacity] f32).
    """
    return EpDispatchPrimitive.outer_primitive.bind(
        handle_mem, topk_idx, tokens, topk_weights,
        recv_capacity=recv_capacity, top_k=top_k,
    )


def ep_combine_fwd(handle_mem, expert_out, num_local_tokens):
    """Gather expert outputs back to home ranks.

    The hadamard product with `recv_topk_weights` happens in JAX before this
    call (see transformer_engine.jax.ep.ep_combine). recv_capacity is implicit
    in expert_out.shape[0].

    Returns result [num_local_tokens, H].
    """
    return EpCombinePrimitive.outer_primitive.bind(
        handle_mem, expert_out, num_local_tokens=num_local_tokens,
    )


def ep_dispatch_bwd(handle_mem, grad, num_local_tokens):
    """Backward of dispatch (combine direction).

    Returns grad_tokens [num_local_tokens, H].
    """
    return EpDispatchBwdPrimitive.outer_primitive.bind(
        handle_mem, grad, num_local_tokens=num_local_tokens,
    )


def ep_combine_bwd(handle_mem, grad, recv_capacity):
    """Backward of combine (dispatch direction).

    Returns grad_expert_out [recv_capacity, H].
    """
    return EpCombineBwdPrimitive.outer_primitive.bind(
        handle_mem, grad, recv_capacity=recv_capacity,
    )
