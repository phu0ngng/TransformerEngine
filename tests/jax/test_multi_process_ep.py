# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process EP tests with deterministic numeric assertions.

Launched via tests/jax/multi_process_launch_ep.sh (one process per GPU,
JAX coordinator on 127.0.0.1:12345).

Routing recipe is fully deterministic so every non-padded recv slot is
addressable as (src_rank, src_t, k) and every output value is computable
locally on each rank — no process_allgather needed for the reference.

  topk_idx[t, k]     = (rank + k) % num_experts   # round-robin top-k experts
  topk_weights[t, k] = 1.0 / top_k

  tokens[t, h]       = encode(rank, t, h) = rank * RANK_STRIDE + t * T_STRIDE + h
"""

import sys
import unittest

import jax
# Enable int64 so topk_idx survives as int64 through JAX (NCCL EP reads kInt64).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.ep import ep_bootstrap, ep_prepare, ep_dispatch, ep_combine
import transformer_engine.jax.cpp_extensions as tex


# ── Encoding (bf16-exact: every value ≤ 256, 1-indexed so 0 = padding) ──────

T_STRIDE = 16  # > num_tokens; rank*T_STRIDE + t + 1 ≤ num_procs*T_STRIDE ≤ 256


def _encode(rank, t, h):
    del h  # all hidden dims share the same value; +1 reserves 0 for padding
    return rank * T_STRIDE + t + 1


def _decode_first_elem(x):
    """Recover (rank, t) from x = rank*T_STRIDE + t + 1; returns None if x==0."""
    v = int(round(x))
    if v == 0:
        return None
    return (v - 1) // T_STRIDE, (v - 1) % T_STRIDE


# ── Test class ────────────────────────────────────────────────────────────────


class TestEP(unittest.TestCase):
    """EP: prepare, dispatch (recv_tokens + recv_topk_weights), combine, bwd."""

    num_tokens: int = 8
    top_k: int = 2
    hidden_dim: int = 32

    @classmethod
    def setUpClass(cls):
        cls.num_procs = jax.process_count()
        cls.rank = jax.process_index()
        cls.ep_size = cls.num_procs
        cls.num_experts = cls.num_procs
        cls.num_local_experts = cls.num_experts // cls.ep_size  # = 1
        cls.max_tokens_per_rank = cls.num_tokens
        # HT FLAT recv layout: each rank may receive up to ep_size * max_tokens_per_rank
        # slots (worst case all sources route everything to this rank).
        cls.recv_capacity = cls.num_procs * cls.num_tokens

        ep_bootstrap(
            world_size=cls.num_procs,
            rank=cls.rank,
            ep_size=cls.ep_size,
            num_experts=cls.num_experts,
            max_tokens_per_rank=cls.max_tokens_per_rank,
            max_recv_tokens_per_rank=cls.recv_capacity,  # HT requires > 0
            hidden_dim=cls.hidden_dim,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _routing(self, src_rank):
        """Globally-known routing — matches cpp test_ep_pipeline.cu formula."""
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
        """Replay routing: how many tokens this rank receives per local expert."""
        my_expert = self.rank  # 1 expert per rank → expert_id == rank
        K, E, NLE = self.top_k, self.num_experts, self.num_local_experts
        cnt = 0
        for src in range(self.num_procs):
            for t in range(self.num_tokens):
                for k in range(K):
                    if (src * NLE + t * K + k) % E == my_expert:
                        cnt += 1
        return np.array([cnt], dtype=np.int32)

    def _expected_recv_pairs(self):
        """Multiset of (src_rank, src_t) that should appear in recv_tokens."""
        my_expert = self.rank
        K, E, NLE = self.top_k, self.num_experts, self.num_local_experts
        pairs = []
        for src in range(self.num_procs):
            for t in range(self.num_tokens):
                for k in range(K):
                    if (src * NLE + t * K + k) % E == my_expert:
                        pairs.append((src, t))
        return pairs  # length == _expected_token_counts()[0]

    # ── Test 1: ep_prepare token_counts numeric ───────────────────────────────

    def test_prepare_token_counts(self):
        topk_idx = jnp.asarray(self._routing(self.rank))
        token_counts, handle_mem = ep_prepare(topk_idx)
        token_counts.block_until_ready()
        # HT EXPERT_MAJOR: token_counts are PADDED (sum equals dispatch output slot count).
        # Verify sane bounds: ≥ unpadded routing total, ≤ recv_capacity.
        unpadded = int(self._expected_token_counts()[0])
        got = int(np.asarray(token_counts)[0])
        self.assertGreaterEqual(got, unpadded,
                                f"rank {self.rank}: padded count {got} < unpadded {unpadded}")
        self.assertLessEqual(got, self.recv_capacity,
                             f"rank {self.rank}: padded count {got} > recv_capacity {self.recv_capacity}")
        self.assertGreater(int(handle_mem.shape[0]), 0)
        self.assertEqual(handle_mem.dtype, jnp.uint8)

    # ── Test 2: ep_dispatch recv_tokens numeric ───────────────────────────────

    def test_dispatch_recv_tokens(self):
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        recv_tokens, _, _, _ = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        recv_tokens.block_until_ready()

        # HT EXPERT_MAJOR pads slots; encoding is 1-indexed so 0 = padding/empty.
        # Scan all rows, drop padding via decode==None, compare multiset to routing replay.
        recv_np = np.asarray(recv_tokens.astype(jnp.float32))
        decoded = []
        for i in range(self.recv_capacity):
            d = _decode_first_elem(recv_np[i, 0])
            if d is not None:
                decoded.append(d)
        decoded.sort()
        expected = sorted(self._expected_recv_pairs())
        self.assertEqual(decoded, expected,
                         f"rank {self.rank}: decoded recv slots != expected routing")
        self.assertEqual(recv_tokens.shape, (self.recv_capacity, self.hidden_dim))

    # ── Test 3: ep_dispatch recv_topk_weights numeric ─────────────────────────

    def test_dispatch_recv_topk_weights(self):
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        recv_tokens, recv_topk_weights, _, _ = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        recv_topk_weights.block_until_ready()

        # Find filled rows via the 1-indexed token encoding (col 0 != 0 = real token).
        recv_t = np.asarray(recv_tokens.astype(jnp.float32))
        rw = np.asarray(recv_topk_weights)
        expected_w = 1.0 / self.top_k
        n_expected = len(self._expected_recv_pairs())
        n_match = 0
        for i in range(self.recv_capacity):
            if int(round(recv_t[i, 0])) == 0:
                continue  # padding slot
            if np.isclose(rw[i], expected_w, atol=1e-5):
                n_match += 1
        self.assertEqual(n_match, n_expected,
                         f"rank {self.rank}: filled rows with weight {expected_w} "
                         f"= {n_match}, expected {n_expected}")

    # ── Test 4: combine round-trip with identity expert ───────────────────────

    def test_combine_round_trip(self):
        """Use the low-level tex.ep_combine_fwd to mirror the cpp combine test:
        unweighted sum across top_k slots ⇒ result == top_k * tokens.
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        recv_tokens, _, handle_mem, _ = ep_dispatch(
            topk_idx, tokens, topk_weights, self.recv_capacity
        )
        # Bypass JAX hadamard — cpp combine is unweighted sum.
        result = tex.ep_combine_fwd(handle_mem, recv_tokens, self.num_tokens)
        result.block_until_ready()

        expected = np.asarray(tokens.astype(jnp.float32)) * float(self.top_k)
        np.testing.assert_allclose(
            np.asarray(result.astype(jnp.float32)), expected,
            atol=2e-2, rtol=2e-2,
            err_msg=f"rank {self.rank}: identity round-trip mismatch",
        )

    # ── Test 5: full forward + backward ────────────────────────────────────────

    def test_full_fwd_bwd(self):
        """Unweighted-combine round-trip: res = top_k * tokens.
        Loss = 0.5*res^2 ⇒ grad_res = res = top_k * tokens.
        cpp combine_bwd dispatches grad_res unweighted → grad_expert[slot] = grad_res[t_orig].
        cpp dispatch_bwd combines grad_expert unweighted → grad_tokens[t] = top_k * grad_res[t]
                                                          = top_k^2 * tokens.
        """
        from functools import partial as _partial

        @_partial(jax.custom_vjp, nondiff_argnums=(2,))
        def _combine_raw(hm, expert_out, num_local_tokens):
            return tex.ep_combine_fwd(hm, expert_out, num_local_tokens)

        def _combine_raw_fwd(hm, expert_out, num_local_tokens):
            return tex.ep_combine_fwd(hm, expert_out, num_local_tokens), \
                   (hm, expert_out.shape[0])

        def _combine_raw_bwd(num_local_tokens, res, g):
            del num_local_tokens
            hm, recv_capacity = res
            g_eo = tex.ep_combine_bwd(hm, g, recv_capacity)
            return (jnp.zeros_like(hm), g_eo)

        _combine_raw.defvjp(_combine_raw_fwd, _combine_raw_bwd)

        topk_idx = jnp.asarray(self._routing(self.rank))
        topk_weights = self._make_topk_weights()
        tokens = self._make_tokens()

        def loss_fn(toks):
            recv_t, _, hm, _ = ep_dispatch(
                topk_idx, toks, topk_weights, self.recv_capacity
            )
            res = _combine_raw(hm, recv_t, self.num_tokens)
            return 0.5 * (res.astype(jnp.float32) ** 2).sum()

        _, grad_tokens = jax.value_and_grad(loss_fn)(tokens)
        grad_tokens.block_until_ready()

        scale = float(self.top_k) ** 2
        expected = np.asarray(tokens.astype(jnp.float32)) * scale
        np.testing.assert_allclose(
            np.asarray(grad_tokens.astype(jnp.float32)), expected,
            atol=5e-2, rtol=5e-2,
            err_msg=f"rank {self.rank}: full fwd+bwd grad mismatch",
        )


    # ── Test 6: router gradient via public ep_combine ─────────────────────────

    def test_router_grad_through_combine(self):
        """Router-grad regression: gradient flows through `topk_weights`.

        Wraps the public ep_dispatch -> identity expert -> ep_combine pipeline
        with a learnable per-source-token scalar `g` applied to topk_weights:
            scaled_weights[t, k] = g[t] * topk_weights[t, k]
        Loss = 0.5 * combined_out**2.
        With identity expert and uniform topk_weights = 1/top_k:
            combined_out[t] = g[t] * tokens[t]              (sum over k cancels)
            ⇒ grad_g[t] = combined_out[t] · tokens[t]
                        = g[t] * sum_h(tokens[t,h]^2)
        Asserts the analytic grad_g against jax.value_and_grad — non-zero values
        prove the topk_weights cotangent is propagated end-to-end.
        """
        topk_idx = jnp.asarray(self._routing(self.rank))
        base_weights = self._make_topk_weights()
        tokens = self._make_tokens()
        # Learnable per-token gate; use distinct values so we'd notice if grads
        # collapse to a constant.
        g = jnp.asarray(
            np.linspace(0.7, 1.3, self.num_tokens, dtype=np.float32)
        )

        def loss_fn(g_):
            scaled = base_weights * g_[:, None]
            recv_t, recv_w, hm, tc = ep_dispatch(
                topk_idx, tokens, scaled, self.recv_capacity
            )
            # Identity expert: feed recv_tokens straight back into combine.
            out = ep_combine(hm, tc, recv_t, recv_w, self.num_tokens)
            return 0.5 * (out.astype(jnp.float32) ** 2).sum()

        _, grad_g = jax.value_and_grad(loss_fn)(g)
        grad_g.block_until_ready()

        # Analytic reference: out[t] = g[t] * tokens[t] (after sum over top_k of
        # the unweighted combine, with each per-slot weight = g[t] * 1/top_k and
        # top_k slots per source token, giving net factor g[t]).
        toks_f32 = np.asarray(tokens.astype(jnp.float32))
        sq_per_token = (toks_f32 * toks_f32).sum(axis=-1)  # [T]
        expected = np.asarray(g) * sq_per_token

        np.testing.assert_allclose(
            np.asarray(grad_g.astype(jnp.float32)), expected,
            atol=5e-2, rtol=5e-2,
            err_msg=(
                f"rank {self.rank}: router grad mismatch. If this fails with "
                f"all-zero grad_g, the topk_weights cotangent is being dropped "
                f"in _dispatch_bwd or _combine_bwd."
            ),
        )


# ── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_multi_process_ep.py <coord_addr> <proc_id> <num_procs>")
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
    suite = loader.loadTestsFromTestCase(TestEP)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
