# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""High-level JAX Expert Parallelism (EP) API with custom VJP."""

import ctypes
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as jmu

import transformer_engine_jax
import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax.sharding import global_mesh_resource, get_mesh_axis_size

__all__ = [
    "ep_bootstrap",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
]


# ── Bootstrap ────────────────────────────────────────────────────────────────


def ep_bootstrap(
    world_size,
    rank,
    ep_size,
    num_experts,
    max_tokens_per_rank,
    max_recv_tokens_per_rank,
    hidden_dim,
):
    """Initialize the EP communicator.

    Generates a ncclUniqueId on rank 0, broadcasts to all ranks via
    jax.experimental.multihost_utils.process_allgather, then initializes NCCL EP.
    Call once per process at startup before any ep_prepare/dispatch/combine.

    Limitation: TE-EP currently supports ONE EP group per process. The
    underlying C++ `EPBackend` is a Meyers singleton and the JAX-side
    `EpConfig` cache (set via `set_ep_config`) is process-global.
    Multi-EP-group workflows in a single process are NOT supported today.

    Args:
        world_size:                Total number of ranks (= jax.process_count()).
        rank:                      This process's rank (= jax.process_index()).
        ep_size:                   Number of ranks in the EP group.
        num_experts:               Total experts across all ranks.
        max_tokens_per_rank:       Static upper bound on tokens this rank SENDS per dispatch.
        max_recv_tokens_per_rank:  Static upper bound on tokens this rank RECEIVES per
                                   dispatch (= recv_capacity_max). MUST be > 0 — NCCL EP
                                   has no default and errors out on 0. Size for worst-case
                                   top_k fan-out (e.g. ep_size * max_tokens_per_rank * top_k).
        hidden_dim:                Token hidden dimension.

    Per-handle EM zone alignment is a per-prepare knob — see
    ``ep_prepare(..., dispatch_output_per_expert_alignment=...)`` and
    ``ep_dispatch(..., dispatch_output_per_expert_alignment=...)``.
    """
    UID_SIZE = 128
    if rank == 0:
        try:
            from nccl import get_unique_id

            uid_bytes = bytes(get_unique_id())[:UID_SIZE]
        except ImportError:
            libnccl = ctypes.CDLL("libnccl.so.2", use_errno=True)
            uid_arr = (ctypes.c_uint8 * UID_SIZE)()
            ret = libnccl.ncclGetUniqueId(ctypes.cast(uid_arr, ctypes.c_void_p))
            assert ret == 0, f"ncclGetUniqueId failed with code {ret}"
            uid_bytes = bytes(uid_arr)
    else:
        uid_bytes = bytes(UID_SIZE)

    uid_arr = jnp.frombuffer(uid_bytes, dtype=jnp.uint8)
    uid_arr = jmu.process_allgather(uid_arr)[0]
    uid_bytes = bytes(uid_arr.tolist())

    ep_resource = global_mesh_resource().ep_resource
    if ep_resource is None:
        raise ValueError(
            "ep_bootstrap requires MeshResource.ep_resource to be set; enter a"
            " global_shard_guard(MeshResource(..., ep_resource=<axis name>)) before bootstrap."
        )
    mesh_ep_size = get_mesh_axis_size(ep_resource)
    if mesh_ep_size != ep_size:
        raise ValueError(
            f"ep_bootstrap: EpConfig.ep_size ({ep_size}) does not match mesh axis"
            f" '{ep_resource}' size ({mesh_ep_size})."
        )

    transformer_engine_jax.initialize_ep_communicator(
        uid_bytes,
        world_size,
        rank,
        ep_size,
        num_experts,
        max_tokens_per_rank,
        max_recv_tokens_per_rank,
        hidden_dim,
    )

    assert (
        num_experts % ep_size == 0
    ), f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
    tex.ep.set_ep_config(
        tex.ep.EpConfig(
            world_size=world_size,
            rank=rank,
            ep_size=ep_size,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            max_tokens_per_rank=max_tokens_per_rank,
            max_recv_tokens_per_rank=max_recv_tokens_per_rank,
            hidden_dim=hidden_dim,
        )
    )


# ── ep_prepare (low-level, no custom_vjp) ────────────────────────────────────
# Kept for callers that need a handle without dispatching (e.g. combine-only
# tests). For the normal MoE flow, use ep_dispatch — which folds prepare in.


def ep_prepare(topk_idx, dispatch_output_per_expert_alignment=0):
    """Routing preparation: AllGather routing map, compute per-expert token counts.

    Args:
        topk_idx: [..., top_k] int32 or int64 sparse routing indices (int32 is
            upcast to int64 on-stream by the FFI). The leading dims
            (e.g. (T,) or (B, S)) are flattened by the FFI; top_k is the last
            dim and must match what the dispatch will send.
        dispatch_output_per_expert_alignment: per-handle EM zone alignment in
            tokens (pow2; 0/1 = no padding). Threaded through to NCCL EP's
            handle config — each EM zone is padded to this many tokens with
            zero-filled trailing slots. Static across a prepare → dispatch →
            combine cycle (the same handle_mem carries it).

    Returns:
        token_counts: [num_local_experts] int32 — tokens per local expert.
        handle_mem:   [N] uint8 — routing state for dispatch/combine/bwd.
    """
    return tex.ep_prepare(
        topk_idx,
        dispatch_output_per_expert_alignment=int(dispatch_output_per_expert_alignment),
    )


# ── ep_dispatch (custom_vjp; folds prepare in; returns 4 outputs) ────────────
# NCCL EP plans to fuse metadata preprocessing into dispatch — this API shape
# anticipates that so the public surface won't change when the fusion lands.


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def ep_dispatch(
    topk_idx, tokens, topk_weights, recv_capacity, dispatch_output_per_expert_alignment=0
):
    """Prepare routing and dispatch tokens + weights to expert ranks.

    Args:
        topk_idx:      [..., top_k] int32 or int64 sparse routing indices (int32
                       upcast on-stream by the FFI). Leading dims
                       (e.g. (T,) or (B, S)) are flattened by the FFI.
        tokens:        [..., H] token activations. Same leading dims as topk_idx.
        topk_weights:  [..., top_k] float32 routing weights (sent alongside tokens).
        recv_capacity: STATIC int — number of recv slots = recv_tokens.shape[0].
                       Set to ceil(T_flat * overalloc_factor) for a balanced buffer.
        dispatch_output_per_expert_alignment: STATIC int. Per-handle EM zone
                       alignment in tokens (pow2; 0/1 = no padding). Forwarded
                       to the inner ``ep_prepare`` so the produced handle uses
                       this alignment for both fwd and bwd.

    Returns:
        Tuple of (recv_tokens [recv_capacity, H] (always 2D),
                  recv_topk_weights [recv_capacity] float32 (1 weight per slot),
                  handle_mem [N] uint8,
                  token_counts [num_local_experts] int32).

    Backward limitation (router gradient):
        `grad_topk_weights` is exact for routers where `topk_weights[t, k]` is
        uniform across `k` (e.g. `1 / top_k`). For non-uniform routers (the
        per-`k` cotangents differ at the same source token) the returned
        `grad_topk_weights[t, k]` is the per-token AVERAGE of the per-`k`
        gradients — direction is approximate, magnitude is biased. The router
        will train but updates will be noisier than analytical. See
        `_dispatch_bwd` for the underlying constraint (NCCL EP combine sums
        top_k slot contributions per source token, so per-`k` identity is
        lost in the bwd direction).
    """
    return _dispatch_fwd(
        topk_idx, tokens, topk_weights, recv_capacity, dispatch_output_per_expert_alignment
    )[0]


def _dispatch_fwd(
    topk_idx, tokens, topk_weights, recv_capacity, dispatch_output_per_expert_alignment
):
    top_k = int(topk_weights.shape[-1])
    hidden_dim = int(tokens.shape[-1])
    token_counts, handle_mem = tex.ep_prepare(
        topk_idx,
        dispatch_output_per_expert_alignment=dispatch_output_per_expert_alignment,
    )
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        handle_mem, topk_idx, tokens, topk_weights, recv_capacity, top_k
    )
    # Save the full leading-dim shape (e.g. (T,) or (B, S)) so the bwd can
    # restore N-D output without forcing a JAX-side reshape that would break
    # sharding patterns and trigger an unwanted AllGather.
    out_leading = tuple(tokens.shape[:-1])
    primal = (recv_tokens, recv_topk_weights, handle_mem, token_counts)
    return primal, (handle_mem, out_leading, top_k, hidden_dim)


def _dispatch_bwd(recv_capacity, dispatch_output_per_expert_alignment, res, g_outputs):
    # alignment is baked into handle_mem at fwd-prepare time; recv_capacity is
    # implicit in g_outputs[0].shape (= ep_size * recv_capacity_per_rank).
    del recv_capacity, dispatch_output_per_expert_alignment
    handle_mem, out_leading, top_k, hidden_dim = res
    g_recv_tokens = g_outputs[0]  # 3D [ep_size, recv_pr, H]
    g_recv_topk_weights = g_outputs[1]  # 2D [ep_size, recv_pr] f32

    grad_tokens = tex.ep_dispatch_bwd(handle_mem, g_recv_tokens, out_leading)

    # NCCL EP combine asserts the buffer's row width matches the group's
    # configured hidden_dim. Broadcast the f32 weight cotangent to a 3D
    # bf16 [ep_size, recv_pr, hidden_dim] buffer (every column = the scalar),
    # combine to [T_flat, hidden_dim], take col 0. Combine sums top_k slot
    # contributions per token, so the result is top_k * grad_recv_w[slot_for_t,k].
    # Broadcast back across top_k as grad / top_k — exact magnitude for uniform
    # routers, approximate (per-token average) for non-uniform.
    g_w_padded = jnp.broadcast_to(
        g_recv_topk_weights.astype(jnp.bfloat16)[..., None],
        g_recv_topk_weights.shape + (hidden_dim,),
    )
    # Combine the weights cotangent as a flat 1-D leading-dim tensor; reshape after.
    T_flat = 1
    for d in out_leading:
        T_flat *= int(d)
    grad_w_padded = tex.ep_dispatch_bwd(handle_mem, g_w_padded, (T_flat,))
    grad_w_per_tok = grad_w_padded[:, 0].astype(jnp.float32) / float(top_k)  # [T_flat]
    # Reshape to leading dims + add top_k axis broadcast.
    grad_topk_weights = jnp.broadcast_to(
        grad_w_per_tok.reshape(out_leading + (1,)), out_leading + (top_k,)
    ).astype(jnp.float32)

    return (None, grad_tokens, grad_topk_weights)


ep_dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)


# ── ep_combine (custom_vjp; masked hadamard) ─────────────────────────────────
#
# Caller applies per-slot weighting in JAX (hadamard product) before sending
# expert outputs back via NCCL EP combine.
#
# The hadamard is masked by `token_counts` from ep_prepare so we don't
# multiply garbage in the overallocated tail. recv_capacity is implicit
# (= expert_out.shape[0]).


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def ep_combine(handle_mem, token_counts, expert_out, recv_topk_weights, num_local_tokens):
    """Gather weighted expert outputs back to home ranks.

    Args:
        handle_mem:         [ep_size, N] uint8 from ep_prepare (sharded on ep_resource).
        token_counts:       [ep_size, num_local_experts] int32 from ep_prepare. Used to
                            mask the in-JAX hadamard so the overallocated tail of
                            expert_out is not multiplied by garbage weights.
        expert_out:         [ep_size, recv_capacity_per_rank, H] post-FFN activations.
                            ALWAYS 3D (sharded on ep_resource).
        recv_topk_weights:  [ep_size, recv_capacity_per_rank] float32 routing weights.
        num_local_tokens:   STATIC int OR tuple. int → 2D output [T, H].
                            tuple (B, S, ...) → N-D output [..., H] with the
                            same leading shape (preserves sharding pattern).

    Returns:
        result: [..., H] combined output (shape determined by num_local_tokens).
    """
    return _combine_fwd(handle_mem, token_counts, expert_out, recv_topk_weights, num_local_tokens)[
        0
    ]


def _make_valid_mask(token_counts, recv_capacity_per_rank, dtype):
    # token_counts: [ep_size, num_local_experts] int32. NCCL EP packs each local
    # expert into a fixed `recv_capacity_per_rank / NLE` stride within each
    # source-rank slice; valid slots are `[0, token_counts[ep, e])` per zone.
    # Returns mask shaped [ep_size, recv_capacity_per_rank, 1] for hadamard
    # against expert_out / grad_weighted.
    ep_size, nle = token_counts.shape
    slots_per_e = recv_capacity_per_rank // nle
    idx = jnp.arange(recv_capacity_per_rank, dtype=jnp.int32)
    e_idx = idx // slots_per_e  # [recv_capacity_per_rank]
    slot_in_e = idx % slots_per_e  # [recv_capacity_per_rank]
    counts = token_counts.astype(jnp.int32)  # [ep_size, nle]
    # Gather per-zone count for each slot: counts[:, e_idx] → [ep_size, recv_pr]
    per_slot_count = counts[:, e_idx]
    mask = (slot_in_e[None, :] < per_slot_count).astype(dtype)
    return mask[..., None]


def _combine_fwd(handle_mem, token_counts, expert_out, recv_topk_weights, num_local_tokens):
    ep_size, recv_capacity_per_rank, _ = expert_out.shape
    recv_capacity = ep_size * recv_capacity_per_rank
    w = recv_topk_weights.astype(expert_out.dtype)[..., None]  # [ep_size, recv_pr, 1]
    mask = _make_valid_mask(token_counts, recv_capacity_per_rank, expert_out.dtype)
    weighted = expert_out * w * mask
    result = tex.ep_combine_fwd(handle_mem, weighted, num_local_tokens)
    return result, (handle_mem, recv_topk_weights, expert_out, token_counts, recv_capacity)


def _combine_bwd(_, res, g_result):
    handle_mem, recv_topk_weights, expert_out, token_counts, recv_capacity = res
    # g_result may be N-D [..., H]; combine_bwd returns 3D [ep_size, recv_pr, H].
    grad_weighted = tex.ep_combine_bwd(handle_mem, g_result, recv_capacity)
    recv_capacity_per_rank = grad_weighted.shape[1]
    w = recv_topk_weights.astype(grad_weighted.dtype)[..., None]
    mask = _make_valid_mask(token_counts, recv_capacity_per_rank, grad_weighted.dtype)
    # Chain rule for `weighted = expert_out * w * mask`:
    #   grad_expert_out = grad_weighted * w * mask
    #   grad_w (per-slot) = sum_H(grad_weighted * expert_out * mask)
    grad_expert_out = grad_weighted * w * mask
    grad_recv_topk_weights = (
        (
            grad_weighted.astype(jnp.float32)
            * expert_out.astype(jnp.float32)
            * mask.astype(jnp.float32)
        )
        .sum(axis=-1)
        .astype(recv_topk_weights.dtype)
    )  # [recv_capacity]
    # handle_mem and token_counts are non-diff plumbing; return None so JAX
    # skips emitting zero cotangents for them.
    return (None, None, grad_expert_out, grad_recv_topk_weights)


ep_combine.defvjp(_combine_fwd, _combine_bwd)
