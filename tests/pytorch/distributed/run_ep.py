# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process PyTorch EP tests, launched via torchrun (one process per GPU)."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.distributed as dist

from transformer_engine.pytorch.ep import (
    EpHandle,
    EpBuffer,
    ep_bootstrap,
    ep_finalize,
    ep_prepare,
    ep_dispatch,
    ep_combine,
    symm_mem_alloc,
    _ep_combine_raw,
    _ep_dispatch_raw,
)

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex


NUM_LOCAL_EXPERTS = 2
HIDDEN_DIM = 32
TOP_K = 2
TOKENS_PER_RANK = 4


def _device_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _build_ep_group():
    """EP group spanning all ranks of the default PG."""
    world_pg = dist.distributed_c10d._get_default_group()
    ranks = list(range(world_pg.size()))
    return dist.new_group(ranks=ranks, backend="nccl")


def _make_identity_inputs(rank, ep_size, nonuniform=False, device="cuda"):
    """Per-rank identity routing + uniform weights so combine ~= tokens."""
    T = TOKENS_PER_RANK
    E = ep_size * NUM_LOCAL_EXPERTS
    topk_idx = np.empty((T, TOP_K), dtype=np.int64)
    if nonuniform:
        assert TOP_K == 2
        for t in range(T):
            topk_idx[t, 0] = 0
            topk_idx[t, 1] = 1 + (t % (E - 1))
    else:
        base = rank * T
        for t in range(T):
            for k in range(TOP_K):
                topk_idx[t, k] = ((base + t) * TOP_K + k) % E
    tokens_np = np.linspace(
        0.1 + rank * 0.01, 0.9 + rank * 0.01, T * HIDDEN_DIM, dtype=np.float32
    ).reshape(T, HIDDEN_DIM)
    topk_weights = np.full((T, TOP_K), 1.0 / TOP_K, dtype=np.float32)
    return (
        torch.from_numpy(topk_idx).to(device),
        torch.from_numpy(tokens_np).to(device=device, dtype=torch.bfloat16),
        torch.from_numpy(topk_weights).to(device),
    )


class _Cfg:
    rank: int
    world_size: int
    ep_size: int
    num_experts: int
    recv_capacity_per_rank: int
    device: torch.device


def _make_cfg() -> _Cfg:
    cfg = _Cfg()
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    cfg.ep_size = cfg.world_size
    cfg.num_experts = NUM_LOCAL_EXPERTS * cfg.ep_size
    T = TOKENS_PER_RANK
    active = min(cfg.num_experts, T * cfg.ep_size * TOP_K)
    overconc = cfg.num_experts // active
    cfg.recv_capacity_per_rank = NUM_LOCAL_EXPERTS * max(T * cfg.ep_size * TOP_K, 16) * overconc * 2
    cfg.device = torch.device("cuda", torch.cuda.current_device())
    return cfg


# -- Test class ---------------------------------------------------------------


class TestEP(unittest.TestCase):
    cfg: _Cfg
    ep_group: dist.ProcessGroup

    @classmethod
    def setUpClass(cls):
        if _device_sm() < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{_device_sm()})")
        cls.cfg = _make_cfg()
        cls.ep_group = _build_ep_group()
        ep_bootstrap(
            cls.ep_group,
            num_experts=cls.cfg.num_experts,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=cls.cfg.recv_capacity_per_rank,
            hidden_dim=HIDDEN_DIM,
            allow_handle_mem_reloc=False,
        )

    def _make_handle(self, alignment=0, top_k=TOP_K):
        return EpHandle(
            top_k=top_k,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=self.cfg.recv_capacity_per_rank,
            hidden_dim=HIDDEN_DIM,
            num_local_experts=NUM_LOCAL_EXPERTS,
            alignment=alignment,
        )

    def _make_buffers(self, dtype=torch.bfloat16):
        """Allocate raw recv buffers + token_counts for the primitive (non-autograd) tests."""
        rc = self.cfg.recv_capacity_per_rank
        return (
            torch.empty(rc, HIDDEN_DIM, dtype=dtype, device=self.cfg.device),
            torch.empty(rc, dtype=torch.float32, device=self.cfg.device),
            torch.empty(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=self.cfg.device),
        )

    def _make_ep_buffer(self, handle):
        """Default auto-alloc EpBuffer; symm-mem/HBM follows bootstrap zero_copy."""
        return EpBuffer(handle)

    @staticmethod
    def _weighted(recv_tokens, recv_w):
        """fp32 per-slot weighting + cast back, matching the combine forward path."""
        mask = (recv_w != 0).to(torch.float32).unsqueeze(-1)
        return (recv_tokens.float() * recv_w.unsqueeze(-1).float() * mask).to(recv_tokens.dtype)

    # -- prepare ----------------------------------------------------------

    def test_primitive_prepare(self):
        handle = self._make_handle()
        topk_idx, _toks, _w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        token_counts = ep_prepare(handle, topk_idx)
        torch.cuda.synchronize()
        self.assertEqual(token_counts.shape, (NUM_LOCAL_EXPERTS,))
        # Global recv total == global send * TOP_K.
        local = int(token_counts.sum().item())
        total = torch.tensor([local], dtype=torch.int64, device=self.cfg.device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.ep_group)
        self.assertEqual(int(total.item()), self.cfg.world_size * TOKENS_PER_RANK * TOP_K)

    # -- identity round-trip via primitives -------------------------------

    def _run_identity_round_trip(self, nonuniform):
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(
            self.cfg.rank, self.cfg.ep_size, nonuniform=nonuniform
        )
        recv_tokens, recv_w, _ = self._make_buffers()
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        result = torch.empty_like(tokens)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_primitive_dispatch_combine_identity_uniform(self):
        self._run_identity_round_trip(nonuniform=False)

    def test_primitive_dispatch_combine_identity_nonuniform(self):
        self._run_identity_round_trip(nonuniform=True)

    def test_3d_input_round_trip(self):
        """3D (B, S, H) inputs round-trip identically to 2D - leading dims are flattened to T."""
        handle = self._make_handle()
        topk_idx, tokens_2d, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        B, S = 2, TOKENS_PER_RANK // 2
        assert B * S == TOKENS_PER_RANK
        tokens_3d = tokens_2d.view(B, S, HIDDEN_DIM)
        topk_idx_3d = topk_idx.view(B, S, TOP_K)
        w_3d = w.view(B, S, TOP_K)
        recv_tokens, recv_w, _ = self._make_buffers()
        ep_prepare(handle, topk_idx_3d)
        _ep_dispatch_raw(handle, topk_idx_3d, tokens_3d, w_3d, recv_tokens, recv_w)
        result = torch.empty_like(tokens_3d)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        self.assertEqual(result.shape, (B, S, HIDDEN_DIM))
        torch.testing.assert_close(result.float(), tokens_3d.float(), atol=5e-2, rtol=5e-2)

    # -- autograd ---------------------------------------------------------

    def test_dispatch_fwd_bwd(self):
        """0.5*||recv_tokens||^2 => grad_tokens ~= TOP_K * tokens."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        recv_t, _recv_w, _tc = ep_dispatch(handle, buffer, tokens_p, topk_idx, w)
        loss = 0.5 * (recv_t.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            tokens_p.grad.float(), tokens.float() * float(TOP_K), atol=5e-2, rtol=5e-2
        )

    def test_combine_fwd_bwd(self):
        """const eo=c, uniform w => max|grad_eo| ~= c / TOP_K."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        _recv_t, recv_w_out, _tc = ep_dispatch(handle, buffer, tokens, topk_idx, w)
        eo_const = 0.5
        eo = torch.full(
            (self.cfg.recv_capacity_per_rank, HIDDEN_DIM),
            eo_const,
            dtype=torch.bfloat16,
            device=self.cfg.device,
            requires_grad=True,
        )
        out = ep_combine(handle, buffer, self._weighted(eo, recv_w_out))
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        arr = eo.grad.float().cpu().numpy()
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertGreater(arr.max(), 0.0)
        np.testing.assert_allclose(arr.max(), eo_const / float(TOP_K), atol=5e-2, rtol=5e-2)

    # -- coverage: top_k=1 + alignment ------------------------------------

    def test_dispatch_combine_top_k_1_all_to_expert_0(self):
        handle = self._make_handle(top_k=1)
        T = TOKENS_PER_RANK
        topk_idx = torch.zeros(T, 1, dtype=torch.int64, device=self.cfg.device)
        w = torch.ones(T, 1, dtype=torch.float32, device=self.cfg.device)
        tokens = torch.from_numpy(
            np.linspace(0.1, 0.9, T * HIDDEN_DIM, dtype=np.float32).reshape(T, HIDDEN_DIM)
        ).to(device=self.cfg.device, dtype=torch.bfloat16)
        recv_tokens, recv_w, _ = self._make_buffers()
        token_counts = ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        result = torch.empty_like(tokens)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        # Rank 0 owns expert 0 and receives world*T tokens; other ranks see 0.
        tc = token_counts.cpu().numpy()
        if self.cfg.rank == 0:
            self.assertEqual(int(tc[0]), self.cfg.world_size * T)
        else:
            self.assertEqual(int(tc[0]), 0)
        if NUM_LOCAL_EXPERTS > 1:
            self.assertEqual(int(tc[1:].sum()), 0)

    def test_dispatch_combine_alignment(self):
        alignment = 8
        handle = self._make_handle(alignment=alignment)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        result = torch.empty_like(tokens)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # -- Integration: CUDA graph, autocast, torch.compile -----------------

    def _moe_step(self, handle, buffer, topk_idx, tokens, w):
        recv_t, recv_w_out, _tc = ep_dispatch(handle, buffer, tokens, topk_idx, w)
        return ep_combine(handle, buffer, self._weighted(recv_t, recv_w_out))

    def test_cuda_graph_capture(self):
        """Capture dispatch+combine via the raw ops; replay must be bit-stable."""
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        result = torch.empty_like(tokens)

        def step():
            ep_prepare(handle, topk_idx)
            _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
            _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)

        for _ in range(3):
            step()
        torch.cuda.synchronize()

        # Routing is fixed per layer, so prepare runs once before capture and only
        # dispatch+combine go into the graph.
        ep_prepare(handle, topk_idx)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
                _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        ref = result.clone()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), ref.float(), atol=0, rtol=0)

    def test_cuda_graph_autograd_dispatch_fwd(self):
        """Capture ep_dispatch (autograd fwd) including ep_prepare-inside-forward.

        Bisects the ep_bench --cuda-graph hang: does capturing AllGather inside
        the autograd forward break?
        """
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)

        # Warm up the eager path + NCCL streams before capture.
        for _ in range(3):
            ep_dispatch(handle, buffer, tokens_p, topk_idx, w)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                ep_dispatch(handle, buffer, tokens_p, topk_idx, w)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

    def test_cuda_graph_make_graphed_dispatch_fwd_bwd(self):
        """make_graphed_callables over a module wrapping ep_dispatch (fwd+bwd).

        Bisects the ep_bench --cuda-graph hang: does make_graphed_callables
        on the autograd path break?
        """
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)

        class _DispatchMod(torch.nn.Module):
            def forward(self, x):
                return ep_dispatch(handle, buffer, x, topk_idx, w)[0]

        mod = _DispatchMod().cuda()
        graphed = torch.cuda.make_graphed_callables(mod, (tokens_p,))
        for _ in range(3):
            tokens_p.grad = None
            r = graphed(tokens_p)
            (0.5 * (r * r).sum(dtype=torch.float32)).backward()
        torch.cuda.synchronize()

    def test_cuda_graph_make_graphed_dispatch_and_combine(self):
        """make_graphed_callables on the tuple (dispatch_mod, combine_mod).

        Bisects the ep_bench --cuda-graph hang: does capturing both modules
        together (which is what the bench does) break?
        """
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        rc = self.cfg.recv_capacity_per_rank
        recv_w_persistent = torch.empty(rc, dtype=torch.float32, device=self.cfg.device)
        eo_p = torch.empty(rc, HIDDEN_DIM, dtype=torch.bfloat16, device=self.cfg.device)
        eo_p = eo_p.requires_grad_(True)

        class _DispatchMod(torch.nn.Module):
            def forward(self, x):
                return ep_dispatch(handle, buffer, x, topk_idx, w)[0]

        class _CombineMod(torch.nn.Module):
            def forward(self, eo):
                return ep_combine(handle, buffer, TestEP._weighted(eo, recv_w_persistent))

        disp_mod = _DispatchMod().cuda()
        comb_mod = _CombineMod().cuda()
        g_disp, g_comb = torch.cuda.make_graphed_callables(
            (disp_mod, comb_mod), ((tokens_p,), (eo_p,))
        )
        for _ in range(3):
            tokens_p.grad = None
            r = g_disp(tokens_p)
            (0.5 * (r * r).sum(dtype=torch.float32)).backward()
            eo_p.grad = None
            out = g_comb(eo_p)
            (0.5 * (out * out).sum(dtype=torch.float32)).backward()
        torch.cuda.synchronize()

    def test_cuda_graph_bench_capture_sequence(self):
        """Reproduce the exact ep_bench --cuda-graph capture sequence.

        Bench flow: make_graphed_callables((disp, comb), ...) -> then direct
        torch.cuda.graph captures of dispatch_raw, ep_dispatch_fwd,
        combine_raw, ep_combine_fwd on a side stream, all on the same handle.
        Hypothesis: running multiple graph captures back-to-back without a
        barrier between them deadlocks NCCL.
        """
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        rc = self.cfg.recv_capacity_per_rank
        recv_tokens = torch.empty(rc, HIDDEN_DIM, dtype=torch.bfloat16, device=self.cfg.device)
        recv_w = torch.empty(rc, dtype=torch.float32, device=self.cfg.device)

        # Prime: prepare once + one raw dispatch so recv_tokens has valid contents.
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        torch.cuda.synchronize()
        expert_out = recv_tokens.clone()

        tokens_p = tokens.detach().clone().requires_grad_(True)
        eo_p = recv_tokens.detach().clone().requires_grad_(True)

        class _DispatchMod(torch.nn.Module):
            def forward(self, x):
                return ep_dispatch(handle, buffer, x, topk_idx, w)[0]

        class _CombineMod(torch.nn.Module):
            def forward(self, eo):
                return ep_combine(handle, buffer, TestEP._weighted(eo, recv_w))

        disp_mod = _DispatchMod().cuda()
        comb_mod = _CombineMod().cuda()
        g_disp, g_comb = torch.cuda.make_graphed_callables(
            (disp_mod, comb_mod), ((tokens_p,), (eo_p,))
        )

        # Direct capture of raw + fwd-only stages on a side stream (bench pattern).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        captured = {}
        with torch.cuda.stream(side):
            for name, fn in [
                (
                    "dispatch_raw",
                    lambda: _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w),
                ),
                (
                    "ep_dispatch_fwd",
                    lambda: ep_dispatch(handle, buffer, tokens.detach(), topk_idx, w),
                ),
                (
                    "combine_raw",
                    lambda: _ep_combine_raw(handle, expert_out, torch.empty_like(tokens)),
                ),
                ("ep_combine_fwd", lambda: ep_combine(handle, buffer, self._weighted(recv_tokens, recv_w))),
            ]:
                fn()  # prime allocator
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                captured[name] = g
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Replay every captured graph + the graphed callables a few times.
        for _ in range(3):
            for g in captured.values():
                g.replay()
            tokens_p.grad = None
            r = g_disp(tokens_p)
            (0.5 * (r * r).sum(dtype=torch.float32)).backward()
            eo_p.grad = None
            out = g_comb(eo_p)
            (0.5 * (out * out).sum(dtype=torch.float32)).backward()
        torch.cuda.synchronize()

    def test_autocast_bf16(self):
        """EP under autocast must preserve dtype and identity round-trip."""
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        result = torch.empty_like(tokens)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ep_prepare(handle, topk_idx)
            _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
            _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        self.assertEqual(recv_tokens.dtype, torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_torch_compile_fullgraph(self):
        """Raw EP pipeline under torch.compile(fullgraph=True) must not graph-break."""
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, token_counts = self._make_buffers()
        result = torch.empty_like(tokens)
        alignment = handle.alignment
        handle_id = handle.handle_id
        handle_mem = handle.handle_mem

        def step(handle_mem, topk_idx, tokens, w, recv_tokens, recv_w, token_counts, result):
            torch.ops.transformer_engine_ep.prepare(
                handle_mem, handle_id, topk_idx, token_counts, alignment
            )
            torch.ops.transformer_engine_ep.dispatch(
                handle_mem, handle_id, topk_idx, tokens, w, recv_tokens, recv_w
            )
            mask = (recv_w != 0).to(torch.float32).unsqueeze(-1)
            weighted = (recv_tokens.float() * recv_w.unsqueeze(-1).float() * mask).to(
                recv_tokens.dtype
            )
            torch.ops.transformer_engine_ep.combine(handle_mem, handle_id, weighted, result)
            return result

        ref = torch.empty_like(tokens)
        step(handle_mem, topk_idx, tokens, w, recv_tokens, recv_w, token_counts, ref)
        torch.cuda.synchronize()
        ref_clone = ref.clone()

        recv_tokens.zero_()
        recv_w.zero_()
        token_counts.zero_()
        result.zero_()
        compiled = torch.compile(step, fullgraph=True, dynamic=False)
        out = compiled(handle_mem, topk_idx, tokens, w, recv_tokens, recv_w, token_counts, result)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref_clone.float(), atol=5e-2, rtol=5e-2)

    # -- Zero-copy via NCCL symmetric memory ------------------------------

    def _try_symm_alloc(self, shape, dtype):
        """Allocate a symm-mem tensor or skip if the backend is unavailable."""
        try:
            return symm_mem_alloc(shape, dtype, self.ep_group, device=self.cfg.device)
        except Exception as e:
            self.skipTest(f"NCCL symmetric memory unavailable: {e}")

    def test_zero_copy_dispatch_combine_identity(self):
        """Symm-mem payload buffers must match the HBM path bit-for-bit."""
        handle = self._make_handle()
        topk_idx, tokens_hbm, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        rc = self.cfg.recv_capacity_per_rank

        tokens_sm = self._try_symm_alloc((TOKENS_PER_RANK, HIDDEN_DIM), torch.bfloat16)
        recv_tokens_sm = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        expert_out_sm = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        # Guard against the test silently degrading to the HBM path.
        from torch.distributed._symmetric_memory import is_symm_mem_tensor

        self.assertTrue(is_symm_mem_tensor(tokens_sm), "tokens_sm not symm-mem backed")
        self.assertTrue(is_symm_mem_tensor(recv_tokens_sm), "recv_tokens_sm not symm-mem backed")
        self.assertTrue(is_symm_mem_tensor(expert_out_sm), "expert_out_sm not symm-mem backed")
        tokens_sm.copy_(tokens_hbm)
        recv_w = torch.empty(rc, dtype=torch.float32, device=self.cfg.device)

        # Symm-mem path.
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens_sm, w, recv_tokens_sm, recv_w)
        expert_out_sm.copy_(self._weighted(recv_tokens_sm, recv_w))
        result_sm = torch.empty_like(tokens_hbm)
        _ep_combine_raw(handle, expert_out_sm, result_sm)
        torch.cuda.synchronize()

        # HBM reference.
        handle_ref = self._make_handle()
        recv_tokens_hbm, recv_w_hbm, _ = self._make_buffers()
        result_hbm = torch.empty_like(tokens_hbm)
        ep_prepare(handle_ref, topk_idx)
        _ep_dispatch_raw(handle_ref, topk_idx, tokens_hbm, w, recv_tokens_hbm, recv_w_hbm)
        _ep_combine_raw(handle_ref, self._weighted(recv_tokens_hbm, recv_w_hbm), result_hbm)
        torch.cuda.synchronize()

        torch.testing.assert_close(result_sm.float(), tokens_hbm.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(result_sm, result_hbm, atol=0, rtol=0)

    def test_zero_copy_cuda_graph_capture(self):
        """Capture dispatch+combine over symm-mem payload buffers; replay must be bit-stable."""
        handle = self._make_handle()
        topk_idx, tokens_hbm, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        rc = self.cfg.recv_capacity_per_rank
        tokens = self._try_symm_alloc((TOKENS_PER_RANK, HIDDEN_DIM), torch.bfloat16)
        recv_tokens = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        expert_out = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        tokens.copy_(tokens_hbm)
        recv_w = torch.empty(rc, dtype=torch.float32, device=self.cfg.device)
        result = torch.empty_like(tokens_hbm)

        def step():
            ep_prepare(handle, topk_idx)
            _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
            expert_out.copy_(self._weighted(recv_tokens, recv_w))
            _ep_combine_raw(handle, expert_out, result)

        for _ in range(3):
            step()
        torch.cuda.synchronize()

        ep_prepare(handle, topk_idx)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
                expert_out.copy_(self._weighted(recv_tokens, recv_w))
                _ep_combine_raw(handle, expert_out, result)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        ref = result.clone()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), ref.float(), atol=0, rtol=0)

    def test_zero_copy_autograd_combine(self):
        """EpBuffer auto-allocs recv_tokens / recv_tokens_grad as symm-mem under zero_copy."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        from torch.distributed._symmetric_memory import is_symm_mem_tensor

        self.assertTrue(is_symm_mem_tensor(buffer.recv_tokens))
        self.assertTrue(is_symm_mem_tensor(buffer.recv_tokens_grad))
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_zero_copy_falls_back_when_not_registered(self):
        """Plain torch.empty tensors take the staged-copy fallback correctly."""
        try:
            from torch.distributed._symmetric_memory import is_symm_mem_tensor
        except ImportError:
            is_symm_mem_tensor = None

        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        if is_symm_mem_tensor is not None:
            self.assertFalse(is_symm_mem_tensor(tokens))
            self.assertFalse(is_symm_mem_tensor(recv_tokens))
        result = torch.empty_like(tokens)
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_gradient_checkpointing(self):
        from torch.utils.checkpoint import checkpoint

        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)

        def step(t):
            return self._moe_step(handle, buffer, topk_idx, t, w)

        out = checkpoint(step, tokens_p, use_reentrant=False)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_autocast_bf16_autograd(self):
        """Autocast must not change result/grad dtype through the autograd path."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(buffer.recv_tokens.dtype, torch.bfloat16)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        self.assertEqual(tokens_p.grad.dtype, torch.bfloat16)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # -- Snapshot / scope / multi-iter ------------------------------------

    def test_topk_int32_raises_clear_error(self):
        """int32 topk_idx must error with a message pointing to .long()."""
        handle = self._make_handle()
        topk_idx_int32 = torch.zeros(
            TOKENS_PER_RANK, TOP_K, dtype=torch.int32, device=self.cfg.device
        )
        with self.assertRaises(RuntimeError) as cm:
            ep_prepare(handle, topk_idx_int32)
        msg = str(cm.exception)
        self.assertIn("topk_idx", msg)
        self.assertIn(".long()", msg)

    def test_dispatch_fwd_bwd_multiple_iterations(self):
        """5 fwd+bwd iters on the same EpHandle + EpBuffer must be bit-stable."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        def one_step():
            tokens_p = tokens.detach().clone().requires_grad_(True)
            out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
            loss = 0.5 * (out.float() ** 2).sum()
            loss.backward()
            return out.detach().clone(), tokens_p.grad.detach().clone()

        out_ref, grad_ref = one_step()
        torch.cuda.synchronize()
        for _ in range(4):
            out_i, grad_i = one_step()
            torch.cuda.synchronize()
            torch.testing.assert_close(out_i, out_ref, atol=0, rtol=0)
            torch.testing.assert_close(grad_i, grad_ref, atol=0, rtol=0)

    def test_compile_fullgraph_with_new_api(self):
        """torch.compile(fullgraph=True) on public ep_dispatch+ep_combine; forward only."""
        import torch._dynamo

        torch._dynamo.reset()
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        def step(tokens, topk_idx, w):
            recv_t, recv_w_out, _ = ep_dispatch(handle, buffer, tokens, topk_idx, w)
            return ep_combine(handle, buffer, self._weighted(recv_t, recv_w_out))

        with torch.no_grad():
            ref = step(tokens, topk_idx, w).detach().clone()
        torch.cuda.synchronize()

        compiled = torch.compile(step, fullgraph=True, dynamic=False)
        with torch.no_grad():
            out = compiled(tokens, topk_idx, w)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)

    def test_pp_1f1b_two_handles(self):
        """PP-1F1B interleave (F0 F1 B0 F2 B1 B2) over 3 per-microbatch handles
        + buffers; each bwd must hit grad ~= TOP_K * tokens for its own scale."""
        T, H = TOKENS_PER_RANK, HIDDEN_DIM
        idx, _toks, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        # Distinct token magnitudes per microbatch so grad reuse would be visible.
        scales = (0.13, 0.41, 0.77)
        handles, buffers, tokens, tokens_p = [], [], [], []
        for s in scales:
            h = self._make_handle()
            handles.append(h)
            buffers.append(self._make_ep_buffer(h))
            t = torch.full(
                (T, H), s + self.cfg.rank * 0.01, dtype=torch.bfloat16, device=self.cfg.device
            )
            tokens.append(t)
            tokens_p.append(t.detach().clone().requires_grad_(True))

        recv = [None, None, None]

        def fwd(k):
            recv[k], _, _ = ep_dispatch(handles[k], buffers[k], tokens_p[k], idx, w)

        def bwd(k):
            (0.5 * (recv[k].float() ** 2).sum()).backward()
            recv[k] = None  # release so the next iteration's buffer can be reused safely

        # 1F1B schedule: F0 F1 B0 F2 B1 B2.
        fwd(0)
        fwd(1)
        bwd(0)
        fwd(2)
        bwd(1)
        bwd(2)
        torch.cuda.synchronize()
        for k in range(3):
            torch.testing.assert_close(
                tokens_p[k].grad.float(),
                tokens[k].float() * float(TOP_K),
                atol=5e-2,
                rtol=5e-2,
                msg=f"microbatch {k} gradient mismatch - handle isolation broken?",
            )

    def test_record_stream(self):
        """EpBuffer.record_stream(s) records on all owned tensors."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        s = torch.cuda.Stream()
        buffer.record_stream(s)
        with torch.cuda.stream(s):
            buffer.recv_tokens.add_(0)
            buffer.recv_tokens_grad.add_(0)
        torch.cuda.synchronize()

    def test_external_recv_tokens_round_trip(self):
        """Caller-allocated recv_tokens / recv_tokens_grad round-trip identically."""
        handle = self._make_handle()
        rc = self.cfg.recv_capacity_per_rank
        recv_tokens = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        recv_tokens_grad = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        buffer = EpBuffer(
            handle, recv_tokens=recv_tokens, recv_tokens_grad=recv_tokens_grad
        )
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    def test_external_recv_tokens_validation(self):
        """Caller-provided tensors must match handle shape/dtype/device — else ValueError."""
        handle = self._make_handle()
        rc = self.cfg.recv_capacity_per_rank
        good_shape = (rc, HIDDEN_DIM)
        bad_shape = torch.empty(
            (rc + 1, HIDDEN_DIM), dtype=torch.bfloat16, device=self.cfg.device
        )
        with self.assertRaisesRegex(ValueError, "recv_tokens shape"):
            EpBuffer(handle, recv_tokens=bad_shape)
        bad_dtype = torch.empty(good_shape, dtype=torch.float32, device=self.cfg.device)
        with self.assertRaisesRegex(ValueError, "recv_tokens dtype"):
            EpBuffer(handle, recv_tokens=bad_dtype)
        bad_device = torch.empty(good_shape, dtype=torch.bfloat16, device="cpu")
        with self.assertRaisesRegex(ValueError, "recv_tokens device"):
            EpBuffer(handle, recv_tokens=bad_device)
        # Same validation applies to recv_tokens_grad.
        with self.assertRaisesRegex(ValueError, "recv_tokens_grad shape"):
            EpBuffer(handle, recv_tokens_grad=bad_shape)

    def test_aliased_recv_tokens_grad_round_trip(self):
        """Aliasing recv_tokens and recv_tokens_grad to the same tensor round-trips
        (fwd-output and bwd-output lifecycles don't overlap)."""
        handle = self._make_handle()
        rc = self.cfg.recv_capacity_per_rank
        shared = self._try_symm_alloc((rc, HIDDEN_DIM), torch.bfloat16)
        buffer = EpBuffer(handle, recv_tokens=shared, recv_tokens_grad=shared)
        self.assertIs(buffer.recv_tokens, buffer.recv_tokens_grad)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)


# -- Entry point --------------------------------------------------------------


def _init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Pin symm-mem backend before any default-backend allocation can latch it.
    try:
        from torch.distributed import _symmetric_memory as _symm_mem

        _symm_mem.set_backend("NCCL")
    except (ImportError, RuntimeError):
        pass


if __name__ == "__main__":
    _init_distributed()
    loader = unittest.TestLoader()
    # Optional single-test filter for bisection: NVTE_EP_TEST_FILTER=test_name.
    name_filter = os.environ.get("NVTE_EP_TEST_FILTER")
    if name_filter:
        loader.testMethodPrefix = name_filter
    suite = loader.loadTestsFromTestCase(TestEP)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    dist.barrier()
    # Release NCCL EP's borrowed comm before torch destroys it.
    ep_finalize()
    dist.destroy_process_group()
    sys.exit(0 if result.wasSuccessful() else 1)
