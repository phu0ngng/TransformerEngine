# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""End-to-end MoE pipeline example: dispatch -> per-expert linear -> combine.

Mirrors the role of examples/jax/collective_gemm/test_dense_grad.py for EP.
Validates the public ep_dispatch / ep_combine API by comparing per-rank
output (and grad) to a single-rank reference computed by allgathering
tokens / weights and replaying the routing locally.

Run via run_test_ep.sh (one process per GPU, JAX coordinator on localhost).
"""

import sys
import unittest

import jax
# int64 is required: NCCL EP reads topk_idx as kInt64 through the FFI.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.multihost_utils as jmu
import numpy as np

from common import (
    _initialize_distributed,
    assert_allclose,
    ep_parser,
    make_routing,
    reference_moe,
)
from transformer_engine.jax.ep import ep_dispatch, ep_combine


def _batched_expert_linear(recv_tokens, kernels, num_local_experts):
    """Per-expert linear via a single JAX batched GEMM.

    HT EXPERT_MAJOR lays recv_tokens out as `num_local_experts` contiguous
    equally-sized groups (padded). Under uniform routing the slots per group
    are identical, so we can reshape and use a single batched einsum.

    recv_tokens: [recv_capacity, H]            with recv_capacity % E_local == 0
    kernels:     [E_local, H, H_out]
    Returns:     [recv_capacity, H_out]
    """
    recv_capacity, H = recv_tokens.shape
    H_out = kernels.shape[-1]
    slots_per_expert = recv_capacity // num_local_experts
    grouped = recv_tokens.reshape(num_local_experts, slots_per_expert, H)
    # ehi, eho -> esh
    out = jnp.einsum("eth,eho->eto", grouped, kernels.astype(grouped.dtype))
    return out.reshape(recv_capacity, H_out)


class TestEPPipeline(unittest.TestCase):
    """dispatch -> per-expert linear -> combine, with single-rank reference."""

    _initialized = False

    def setUp(self):
        if TestEPPipeline._initialized:
            return
        cls = TestEPPipeline
        cls.args = ep_parser().parse_args([])
        cls.args.coordinator_address = self.coordinator_address
        cls.args.num_processes = self.num_processes
        cls.args.process_id = self.process_id
        cls.args.local_device_ids = self.local_device_ids
        cls.args.num_tokens = 8
        cls.args.top_k = 2
        cls.args.hidden_dim = 32
        cls.args.hidden_out = 32
        if cls.args.num_experts is None:
            cls.args.num_experts = cls.args.num_processes
        _initialize_distributed(cls.args)

        cls.num_local_experts = cls.args.num_experts // cls.args.num_processes
        cls.recv_capacity = cls.args.recv_capacity
        cls.rank = cls.args.process_id
        cls._initialized = True

    def _make_inputs(self):
        rng = np.random.default_rng(seed=42)
        T, H, K = self.args.num_tokens, self.args.hidden_dim, self.args.top_k
        # Per-rank inputs (different seeds via offset by rank).
        rng_rank = np.random.default_rng(seed=42 + self.rank)
        tokens_np = rng_rank.standard_normal((T, H), dtype=np.float32) * 0.5
        tokens = jnp.asarray(tokens_np, dtype=jnp.bfloat16)

        topk_idx_np = make_routing(
            self.rank, T, K, self.args.num_experts, self.num_local_experts
        )
        topk_idx = jnp.asarray(topk_idx_np)

        # Equal weights: simplest case where reference math is straightforward.
        w_np = np.full((T, K), 1.0 / K, dtype=np.float32)
        topk_weights = jnp.asarray(w_np)

        # Per-expert kernels — must be IDENTICAL on every rank for the reference
        # to be correct. Seed off rng (no rank in seed) so each rank produces
        # the same draw.
        E = self.args.num_experts
        H_out = self.args.hidden_out
        kernels_np = rng.standard_normal((E, H, H_out), dtype=np.float32) * (1.0 / np.sqrt(H))
        return tokens_np, tokens, topk_idx_np, topk_idx, w_np, topk_weights, kernels_np

    def _moe_step(self, tokens, topk_idx, topk_weights, kernels):
        """Forward MoE: dispatch -> per-expert linear -> combine."""
        T = tokens.shape[0]
        recv_tokens, recv_topk_weights, handle_mem, token_counts = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        # Local experts only — slice the global kernel along the expert axis.
        E_local = self.num_local_experts
        local_kernels = jax.lax.dynamic_slice(
            kernels, (self.rank * E_local, 0, 0),
            (E_local, kernels.shape[1], kernels.shape[2])
        )
        expert_out = _batched_expert_linear(recv_tokens, local_kernels, E_local)
        result = ep_combine(
            handle_mem, token_counts, expert_out, recv_topk_weights, T
        )
        return result

    # ── Test 1: forward numerics ──────────────────────────────────────────────

    def test_moe_fwd(self):
        tokens_np, tokens, topk_idx_np, topk_idx, w_np, topk_weights, kernels_np = self._make_inputs()
        kernels = jnp.asarray(kernels_np, dtype=jnp.bfloat16)

        out = self._moe_step(tokens, topk_idx, topk_weights, kernels)
        out.block_until_ready()

        # AllGather inputs across processes for the reference.
        gathered_tokens = np.asarray(
            jmu.process_allgather(tokens.astype(jnp.float32))
        )  # [W, T, H]
        gathered_idx = np.asarray(jmu.process_allgather(topk_idx))  # [W, T, K]
        gathered_w = np.asarray(jmu.process_allgather(topk_weights))  # [W, T, K]

        ref_out_all = reference_moe(
            gathered_tokens, gathered_idx, gathered_w, kernels_np, bias=None
        )
        ref_out_self = ref_out_all[self.rank]

        if self.args.enable_result_check:
            assert_allclose(
                np.asarray(out.astype(jnp.float32)), ref_out_self,
                dtype=jnp.bfloat16,
                err_msg=f"rank {self.rank}: MoE forward mismatch",
            )

    # ── Test 2: forward + backward through tokens ─────────────────────────────

    def test_moe_fwd_bwd(self):
        tokens_np, tokens, topk_idx_np, topk_idx, w_np, topk_weights, kernels_np = self._make_inputs()
        kernels = jnp.asarray(kernels_np, dtype=jnp.bfloat16)

        def loss_fn(toks):
            out = self._moe_step(toks, topk_idx, topk_weights, kernels)
            return 0.5 * (out.astype(jnp.float32) ** 2).sum()

        loss, grad_tokens = jax.value_and_grad(loss_fn)(tokens)
        grad_tokens.block_until_ready()

        # Reference: dL/dtokens via finite-diff on the analytical reference.
        # ref_out[t] = sum_k w_k * tokens[t] @ K_e_k  ⇒  dout/dt = sum_k w_k * K_e_k^T
        # dL = out * dout, so grad_tokens[t] = out[t] @ (sum_k w_k * K_e_k)^T
        gathered_tokens = np.asarray(jmu.process_allgather(tokens.astype(jnp.float32)))
        gathered_idx = np.asarray(jmu.process_allgather(topk_idx))
        gathered_w = np.asarray(jmu.process_allgather(topk_weights))
        ref_out_all = reference_moe(
            gathered_tokens, gathered_idx, gathered_w, kernels_np, bias=None
        )
        T, H = tokens.shape
        K = self.args.top_k
        ref_grad = np.zeros((T, H), dtype=np.float32)
        for t in range(T):
            mixed_kernel = np.zeros_like(kernels_np[0])  # [H, H_out]
            for k in range(K):
                e = int(topk_idx_np[t, k])
                mixed_kernel = mixed_kernel + float(w_np[t, k]) * kernels_np[e]
            ref_grad[t] = ref_out_all[self.rank, t] @ mixed_kernel.T

        if self.args.enable_result_check:
            assert_allclose(
                np.asarray(grad_tokens.astype(jnp.float32)), ref_grad,
                dtype=jnp.bfloat16,
                err_msg=f"rank {self.rank}: grad_tokens mismatch",
            )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_ep_pipeline.py <coord_addr> <proc_id> <num_procs>")
        sys.exit(1)
    coord, pid, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    # Hand-roll command-line distributed init for the standalone path.
    args = ep_parser().parse_args([])
    args.coordinator_address = coord
    args.process_id = pid
    args.num_processes = nproc
    args.local_device_ids = str(pid)
    args.num_experts = nproc
    _initialize_distributed(args)

    # Pump it into the unittest class via setUpClass dependencies.
    TestEPPipeline.coordinator_address = coord
    TestEPPipeline.num_processes = nproc
    TestEPPipeline.process_id = pid
    TestEPPipeline.local_device_ids = str(pid)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEPPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
