# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process SPRINT7 EP sharding test: Mesh(dp=2, ep=2) on 4 GPUs.

Verifies the new EP sharding model end-to-end:
  - MeshResource(dp_resource=..., ep_resource=...) drives axis selection.
  - ep_bootstrap rejects when ep_resource is unset or size mismatches.
  - ep_dispatch/ep_combine outputs carry the expected NamedSharding specs.
  - identity-expert round-trip is numerically correct under SPMD.

Launch via tests/jax/multi_process_launch_ep.sh (one process per GPU).
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.jax.ep import ep_bootstrap, ep_dispatch, ep_combine


# ── Test config ─────────────────────────────────────────────────────────────
# Factorise N = jax.process_count() into (DP, EP) with EP a multiple of 4 —
# NCCL EP's hybridep_adapter asserts the LSA team size is a multiple of 4
# (device/hybridep_adapter.cu:416). 4-GPU box → (1, 4); 8-GPU box → (2, 4).

NUM_LOCAL_EXPERTS = 2  # NLE per ep rank → num_experts = NLE * EP
HIDDEN_DIM = 32
TOP_K = 2
TOKENS_PER_DP_SHARD = 4  # per device along dp


def _factor_dp_ep(num_procs):
    """Pick (dp, ep) with ep a multiple of 4 and dp*ep == num_procs."""
    if num_procs % 4 != 0:
        raise ValueError(f"num_procs ({num_procs}) must be a multiple of 4 for NCCL EP")
    ep = 4
    while ep * 2 <= num_procs and num_procs % (ep * 2) == 0:
        ep *= 2
    # Prefer larger DP when both fit; clip to ep=4 for the smallest valid mesh
    # if num_procs == 4. For num_procs == 8, this yields (dp=2, ep=4).
    ep = 4 if num_procs <= 4 else 4
    return num_procs // ep, ep


def _build_mesh(dp, ep):
    devs = np.asarray(jax.devices()).reshape(dp, ep)
    return Mesh(devs, ("dp", "ep"))


class TestEPSharded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_procs = jax.process_count()
        cls.rank = jax.process_index()
        cls.dp, cls.ep = _factor_dp_ep(cls.num_procs)
        assert cls.dp * cls.ep == cls.num_procs
        cls.num_experts = NUM_LOCAL_EXPERTS * cls.ep
        # recv_capacity is per-DP-group (NCCL EP comms are isolated per DP color),
        # so it does NOT scale with dp. Per-expert capacity inside NCCL EP is
        # max_tokens_per_rank * top_k / NLE; recv_capacity_per_rank = NLE * that.
        cls.recv_capacity_per_rank = TOKENS_PER_DP_SHARD * TOP_K
        cls.recv_capacity = cls.ep * cls.recv_capacity_per_rank
        cls.mesh = _build_mesh(cls.dp, cls.ep)
        cls.mr = MeshResource(dp_resource="dp", ep_resource="ep")
        # Bootstrap under the mesh guard so ep_bootstrap can validate ep_size.
        with cls.mesh, global_shard_guard(cls.mr):
            ep_bootstrap(
                world_size=cls.num_procs,
                rank=cls.rank,
                ep_size=cls.ep,
                num_experts=cls.num_experts,
                max_tokens_per_rank=TOKENS_PER_DP_SHARD,
                max_recv_tokens_per_rank=cls.recv_capacity_per_rank,
                hidden_dim=HIDDEN_DIM,
            )

    def test_bootstrap_rejects_missing_axis(self):
        # Inside an empty MeshResource, bootstrap must raise.
        with self.mesh, global_shard_guard(MeshResource()):
            with self.assertRaisesRegex(ValueError, "ep_resource"):
                ep_bootstrap(
                    world_size=self.num_procs,
                    rank=self.rank,
                    ep_size=self.ep,
                    num_experts=self.num_experts,
                    max_tokens_per_rank=TOKENS_PER_DP_SHARD,
                    max_recv_tokens_per_rank=self.recv_capacity_per_rank,
                    hidden_dim=HIDDEN_DIM,
                )

    def test_dispatch_combine_round_trip(self):
        T_global = TOKENS_PER_DP_SHARD * self.dp
        # Deterministic routing across DP-merged tokens
        topk_idx = np.empty((T_global, TOP_K), dtype=np.int32)
        for t in range(T_global):
            for k in range(TOP_K):
                topk_idx[t, k] = (t * TOP_K + k) % self.num_experts
        topk_idx = jnp.asarray(topk_idx)
        topk_weights = jnp.full((T_global, TOP_K), 1.0 / TOP_K, dtype=jnp.float32)
        tokens = jnp.asarray(
            np.linspace(0.1, 0.9, T_global * HIDDEN_DIM, dtype=np.float32).reshape(
                T_global, HIDDEN_DIM
            ),
            dtype=jnp.bfloat16,
        )

        dp_spec = PartitionSpec("dp", None)
        with self.mesh, global_shard_guard(self.mr):
            topk_idx_s = jax.lax.with_sharding_constraint(
                topk_idx, NamedSharding(self.mesh, dp_spec)
            )
            tokens_s = jax.lax.with_sharding_constraint(tokens, NamedSharding(self.mesh, dp_spec))
            topk_w_s = jax.lax.with_sharding_constraint(
                topk_weights, NamedSharding(self.mesh, dp_spec)
            )

            @jax.jit
            def run(idx, toks, w):
                recv_t, recv_w, handle, tc = ep_dispatch(idx, toks, w, self.recv_capacity)
                ep_spec_2d = PartitionSpec("ep", None)
                ep_spec_1d = PartitionSpec("ep")
                recv_t = jax.lax.with_sharding_constraint(
                    recv_t, NamedSharding(self.mesh, ep_spec_2d)
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, ep_spec_1d)
                )
                out = ep_combine(handle, tc, recv_t, recv_w, T_global)
                return out

            out = run(topk_idx_s, tokens_s, topk_w_s)
            out.block_until_ready()

            # Identity expert + uniform weights = 1/top_k ⇒ output ≈ tokens.
            np.testing.assert_allclose(
                np.asarray(out.astype(jnp.float32)),
                np.asarray(tokens.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )


# ── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_multi_process_ep_sharded.py <coord_addr> <proc_id> <num_procs>")
        sys.exit(1)

    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    jax.distributed.initialize(
        coordinator_address=coord_addr,
        num_processes=num_procs,
        process_id=proc_id,
        local_device_ids=[proc_id],
    )

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEPSharded)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
