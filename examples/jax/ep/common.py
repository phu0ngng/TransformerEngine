# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared helpers for the EP pipeline example.

Mirrors the structure of examples/jax/collective_gemm/common.py: argparse,
JAX distributed init, and shared numeric tolerances. Plus EP-specific
helpers: deterministic routing replay and an allgather-based single-rank
reference for the dispatch -> grouped_gemm -> combine pipeline.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.ep import ep_bootstrap
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# Persistent global mesh guard so eager + jit'd EP calls share the same axis.
_ep_guard = None  # holds the context manager once entered


def dtype_tols(dtype, rtol=None, atol=None):
    """Expected numerical tolerance for a data type."""
    if rtol is not None and atol is not None:
        return {"rtol": rtol, "atol": atol}
    if dtype in [jnp.float32, "float32"]:
        return {"rtol": 1e-5, "atol": 1e-8}
    if dtype in [jnp.float16, "float16"]:
        return {"rtol": 1e-3, "atol": 1e-6}
    if dtype in [jnp.bfloat16, "bfloat16"]:
        return {"rtol": 5e-2, "atol": 5e-2}
    return {"rtol": 1e-5, "atol": 1e-8}


def assert_allclose(actual, desired, rtol=None, atol=None, dtype=None, **kwargs):
    if dtype is None:
        dtype = "float32" if isinstance(actual, float) else actual.dtype
    tols = {} if (rtol is not None and atol is not None) else dtype_tols(dtype)
    if rtol is not None:
        tols["rtol"] = rtol
    if atol is not None:
        tols["atol"] = atol
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)
    np.testing.assert_allclose(actual, desired, **tols, **kwargs)


_distributed_initialized = False


def _initialize_distributed(args):
    """Initialize JAX distributed and EP communicator."""
    global _distributed_initialized
    if _distributed_initialized:
        return

    if args.coordinator_address is None or args.num_processes is None or args.process_id is None:
        raise ValueError(
            "All distributed initialization arguments are required: "
            "--coordinator-address, --num-processes, --process-id"
        )

    if args.local_device_ids is None:
        device_ids = str(args.process_id)
    else:
        device_ids = args.local_device_ids

    print(
        f"[ep] Initializing JAX distributed coord={args.coordinator_address} "
        f"num_processes={args.num_processes} process_id={args.process_id}"
    )
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=device_ids,
    )
    assert (
        jax.local_device_count() == 1
    ), f"EP example requires 1 GPU per process, found {jax.local_device_count()}"

    # Factorize num_processes into (dp_size, ep_size). Default to 2×(N/2) when
    # num_processes is even and >= 4 (matches the SPRINT7 2×2 reference); else
    # fall back to 1×N. Caller may override via --dp-size.
    if args.dp_size is None:
        if args.num_processes >= 4 and args.num_processes % 2 == 0:
            dp_size = 2
        else:
            dp_size = 1
    else:
        dp_size = args.dp_size
    assert (
        args.num_processes % dp_size == 0
    ), f"num_processes ({args.num_processes}) must be divisible by dp_size ({dp_size})"
    ep_size = args.num_processes // dp_size
    args.dp_size = dp_size
    args.ep_size = ep_size
    if args.num_experts is None:
        args.num_experts = ep_size  # one expert per ep-rank by default
    assert (
        args.num_experts % ep_size == 0
    ), f"num_experts ({args.num_experts}) must be divisible by ep_size ({ep_size})"
    # Worst-case recv: every source rank sends its full quota times top_k fan-out.
    recv_capacity = ep_size * args.num_tokens * args.top_k
    args.recv_capacity = recv_capacity

    # SPRINT7: ep_bootstrap requires MeshResource.ep_resource and a Mesh whose
    # ep axis has size==ep_size. Build a dp_size×ep_size Mesh and enter
    # persistent global_shard_guard + Mesh contexts so subsequent
    # ep_dispatch/ep_combine calls inside jit pick up the axes.
    import numpy as _np
    from jax.sharding import Mesh as _Mesh

    global _ep_guard, _ep_mesh_cm
    devs = _np.asarray(jax.devices()).reshape(dp_size, ep_size)
    args.mesh = _Mesh(devs, ("dp", "ep"))
    _ep_mesh_cm = args.mesh
    _ep_mesh_cm.__enter__()
    _ep_guard = global_shard_guard(MeshResource(dp_resource="dp", ep_resource="ep"))
    _ep_guard.__enter__()

    ep_bootstrap(
        world_size=args.num_processes,
        rank=args.process_id,
        ep_size=ep_size,
        num_experts=args.num_experts,
        max_tokens_per_rank=args.num_tokens,
        max_recv_tokens_per_rank=recv_capacity,
        hidden_dim=args.hidden_dim,
    )
    _distributed_initialized = True


def make_routing(rank, num_tokens, top_k, num_experts, num_local_experts):
    """Deterministic routing matching tests/jax/test_multi_process_ep.py.

    topk_idx[t, k] = (rank * NLE + t * K + k) % E
    """
    topk_idx = np.empty((num_tokens, top_k), dtype=np.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            topk_idx[t, k] = (rank * num_local_experts + t * top_k + k) % num_experts
    return topk_idx


def reference_moe(all_tokens, all_topk_idx, all_topk_weights, kernels, bias=None):
    """Single-rank reference for the full MoE step (gather all ranks' tokens).

    Args:
        all_tokens:        [W, T, H] all ranks' input tokens stacked.
        all_topk_idx:      [W, T, K] global routing indices.
        all_topk_weights:  [W, T, K] routing weights.
        kernels:           [E, H, H_out] per-expert weights (same on every rank).
        bias:              [E, H_out] per-expert bias or None.

    Returns:
        ref_out: [W, T, H_out] expected combined output per rank.
    """
    W, T, K = all_topk_idx.shape
    H_out = kernels.shape[-1]
    out = np.zeros((W, T, H_out), dtype=np.float32)
    for w in range(W):
        for t in range(T):
            tok = all_tokens[w, t].astype(np.float32)
            for k in range(K):
                e = int(all_topk_idx[w, t, k])
                wt = float(all_topk_weights[w, t, k])
                y = tok @ kernels[e].astype(np.float32)
                if bias is not None:
                    y = y + bias[e].astype(np.float32)
                out[w, t] += wt * y
    return out


def ep_parser(description="EP MoE pipeline example"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--coordinator-address", type=str, default=None)
    p.add_argument("--num-processes", type=int, default=None)
    p.add_argument("--process-id", type=int, default=None)
    p.add_argument("--local-device-ids", type=str, default=None)
    p.add_argument("--num-tokens", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--hidden-out", type=int, default=32)
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Defaults to ep_size * num_local_experts (resolved at init).",
    )
    p.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help=(
            "DP axis size; ep_size = num_processes // dp_size. Default picks a 2×N/2"
            " factorization when num_processes is even and >= 4, else 1×N."
        ),
    )
    p.add_argument("--enable-result-check", action="store_true", default=True)
    return p
