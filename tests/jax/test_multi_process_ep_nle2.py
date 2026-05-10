# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process EP regression tests with num_local_experts > 1.

Mirror of test_multi_process_ep.py but with num_experts = num_procs * NLE
where NLE = 2. The companion file at NLE = 1 exercises the simpler
single-expert-per-rank path; this file is the regression test for the
multi-local-expert recv layout (where token_counts is a vector and the
combine mask must cover every filled slot across all local experts).

Bootstrapped in its own process because EPBackend is a Meyers singleton
(one EP group per process — see ep_bootstrap docstring).
"""

import sys
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.ep import ep_bootstrap, ep_prepare, ep_dispatch, ep_combine
import transformer_engine.jax.cpp_extensions as tex


T_STRIDE = 16  # encoding stride; rank*T_STRIDE + t + 1 ≤ 256 (bf16-exact)


def _encode(rank, t, h):
    del h
    return rank * T_STRIDE + t + 1


def _decode_first_elem(x):
    v = int(round(x))
    if v == 0:
        return None
    return (v - 1) // T_STRIDE, (v - 1) % T_STRIDE


class TestEPNLE2(unittest.TestCase):
    """EP regression tests with num_local_experts == 2."""

    num_tokens: int = 8
    top_k: int = 2
    hidden_dim: int = 32
    num_local_experts: int = 2

    @classmethod
    def setUpClass(cls):
        cls.num_procs = jax.process_count()
        cls.rank = jax.process_index()
        cls.ep_size = cls.num_procs
        cls.num_experts = cls.num_procs * cls.num_local_experts
        cls.max_tokens_per_rank = cls.num_tokens
        # Worst case: every src token's full top_k fan-out lands on this rank.
        cls.recv_capacity = cls.num_procs * cls.num_tokens * cls.top_k

        ep_bootstrap(
            world_size=cls.num_procs,
            rank=cls.rank,
            ep_size=cls.ep_size,
            num_experts=cls.num_experts,
            max_tokens_per_rank=cls.max_tokens_per_rank,
            max_recv_tokens_per_rank=cls.recv_capacity,
            hidden_dim=cls.hidden_dim,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _routing(self, src_rank):
        """Same formula as cpp test_ep_pipeline.cu generate_topk_idx."""
        T, K, E, NLE = self.num_tokens, self.top_k, self.num_experts, self.num_local_experts
        topk_idx = np.empty((T, K), dtype=np.int64)
        for t in range(T):
            for k in range(K):
                topk_idx[t, k] = (src_rank * NLE + t * K + k) % E
        return topk_idx

    def _make_topk_weights(self):
        return jnp.full((self.num_tokens, self.top_k), 1.0 / self.top_k, dtype=jnp.float32)

    def _make_tokens(self):
        T, H = self.num_tokens, self.hidden_dim
        np_tok = np.empty((T, H), dtype=np.float32)
        for t in range(T):
            for h in range(H):
                np_tok[t, h] = _encode(self.rank, t, h)
        return jnp.asarray(np_tok, dtype=jnp.bfloat16)

    def _expected_token_counts(self):
        """Per-local-expert unpadded counts at this rank (length NLE)."""
        base = self.rank * self.num_local_experts
        K, E = self.top_k, self.num_experts
        cnt = np.zeros(self.num_local_experts, dtype=np.int32)
        for src in range(self.num_procs):
            idx = self._routing(src)
            for t in range(self.num_tokens):
                for k in range(K):
                    e = int(idx[t, k])
                    if base <= e < base + self.num_local_experts:
                        cnt[e - base] += 1
        return cnt

    # ── Test 1: token_counts shape & values per local expert ─────────────────

    def test_prepare_token_counts_per_expert(self):
        topk_idx = jnp.asarray(self._routing(self.rank))
        token_counts, handle_mem = ep_prepare(topk_idx)
        token_counts.block_until_ready()

        self.assertEqual(token_counts.shape, (self.num_local_experts,))
        self.assertEqual(token_counts.dtype, jnp.int32)

        unpadded = self._expected_token_counts()
        got = np.asarray(token_counts)
        # Counts may be padded by NCCL EP, but each must be ≥ unpadded and the
        # sum must fit recv_capacity.
        for e in range(self.num_local_experts):
            self.assertGreaterEqual(int(got[e]), int(unpadded[e]),
                f"rank {self.rank} expert {e}: padded {got[e]} < unpadded {unpadded[e]}")
        self.assertLessEqual(int(got.sum()), self.recv_capacity)

    # ── Test 2: dispatch round-trip with identity expert ─────────────────────

    def test_combine_round_trip(self):
        """Identity-expert round-trip: result == top_k * tokens.

        Uses the low-level tex.ep_combine_fwd (unweighted sum across slots) —
        same shape as test_multi_process_ep.py::test_combine_round_trip but
        with NLE=2 so the recv buffer carries multiple expert chunks.
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        recv_tokens, _, handle_mem, _ = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        result = tex.ep_combine_fwd(handle_mem, recv_tokens, self.num_tokens)
        result.block_until_ready()

        expected = np.asarray(tokens.astype(jnp.float32)) * float(self.top_k)
        np.testing.assert_allclose(
            np.asarray(result.astype(jnp.float32)), expected,
            atol=2e-2, rtol=2e-2,
            err_msg=f"rank {self.rank}: NLE=2 identity round-trip mismatch",
        )

    # ── Test 3: public ep_combine with mask covering NLE>1 layout ────────────

    def test_public_combine_masked_round_trip(self):
        """Exercises the JAX-side hadamard mask under NLE>1.

        Public ep_combine multiplies expert_out by recv_topk_weights * mask
        before scattering. With identity expert (expert_out = recv_tokens),
        uniform weights = 1/top_k, and mask correctly zeroing the trailing
        padded tail, the combined output equals tokens (top_k slots × 1/top_k
        weight × top_k summation = 1).
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        recv_t, recv_w, hm, tc = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        out = ep_combine(hm, tc, recv_t, recv_w, self.num_tokens)
        out.block_until_ready()

        expected = np.asarray(tokens.astype(jnp.float32))
        np.testing.assert_allclose(
            np.asarray(out.astype(jnp.float32)), expected,
            atol=5e-2, rtol=5e-2,
            err_msg=(
                f"rank {self.rank}: NLE=2 public combine mismatch. If this "
                f"fails with garbage values in the output, the mask in "
                f"_make_valid_mask is failing to zero a padded slot."
            ),
        )

    # ── Test 4: router-grad regression at NLE>1 ──────────────────────────────

    def test_router_grad_through_combine(self):
        """Same as the NLE=1 router-grad test, exercising the bwd path under
        multi-local-expert recv layout. With identity expert and uniform
        weights = 1/top_k, scaled_weights[t, k] = g[t] / top_k:
            out[t] = g[t] * tokens[t]
            ⇒ grad_g[t] = out[t] · tokens[t] = g[t] * sum_h(tokens[t,h]^2)
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        base_weights = self._make_topk_weights()
        tokens = self._make_tokens()
        g = jnp.asarray(np.linspace(0.7, 1.3, self.num_tokens, dtype=np.float32))

        def loss_fn(g_):
            scaled = base_weights * g_[:, None]
            recv_t, recv_w, hm, tc = ep_dispatch(
                topk_idx, tokens, scaled, self.recv_capacity
            )
            out = ep_combine(hm, tc, recv_t, recv_w, self.num_tokens)
            return 0.5 * (out.astype(jnp.float32) ** 2).sum()

        _, grad_g = jax.value_and_grad(loss_fn)(g)
        grad_g.block_until_ready()

        toks_f32 = np.asarray(tokens.astype(jnp.float32))
        sq_per_token = (toks_f32 * toks_f32).sum(axis=-1)
        expected = np.asarray(g) * sq_per_token

        np.testing.assert_allclose(
            np.asarray(grad_g.astype(jnp.float32)), expected,
            atol=5e-2, rtol=5e-2,
            err_msg=f"rank {self.rank}: NLE=2 router grad mismatch",
        )

    # ── Test 5: full forward + backward through public API at NLE>1 ──────────

    def test_full_fwd_bwd(self):
        """Public dispatch → identity → combine → 0.5*||out||^2 backward.

        With identity expert and uniform weights, out[t] = tokens[t], so
        grad_tokens[t] = tokens[t] (chain rule). Verifies the bwd primitives
        produce non-garbage gradients under NLE>1.
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        def loss_fn(toks):
            recv_t, recv_w, hm, tc = ep_dispatch(
                topk_idx, toks, topk_weights, self.recv_capacity
            )
            out = ep_combine(hm, tc, recv_t, recv_w, self.num_tokens)
            return 0.5 * (out.astype(jnp.float32) ** 2).sum()

        _, grad_tokens = jax.value_and_grad(loss_fn)(tokens)
        grad_tokens.block_until_ready()

        expected = np.asarray(tokens.astype(jnp.float32))
        np.testing.assert_allclose(
            np.asarray(grad_tokens.astype(jnp.float32)), expected,
            atol=5e-2, rtol=5e-2,
            err_msg=f"rank {self.rank}: NLE=2 fwd+bwd grad mismatch",
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_multi_process_ep_nle2.py <coord_addr> <proc_id> <num_procs>")
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
    suite = loader.loadTestsFromTestCase(TestEPNLE2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
