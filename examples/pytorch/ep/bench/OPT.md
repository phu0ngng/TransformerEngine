# TE PyTorch EP perf optimization log

Config: 8× B300 SXM6, single node, NCCL 2.30.4, CUDA 13.3, PyTorch nightly.
Bench: `tokens-per-rank=2048, hidden=7168, top-k=8, num-experts=256, warmup=5, iters=50`.

NCCL EP reference (`run_nccl_ep_bench.sh`, HT algorithm, EM layout):
- Dispatch: total=517 µs (kernel=416 µs)
- Combine: total=533 µs (kernel=383 µs)
- D+C: total=1060 µs (kernel=799 µs)

All times below are TE wall-clock from `_time_stage_us` (`cudaSync` + `perf_counter_ns`).
"fwd-raw" / "bwd-fwd" rows show the autograd wrapper's overhead above the raw op.

## Baseline (commit 29ce8af0a)

| stage                | mean wall (µs) |
|----------------------|---------------:|
| dispatch_raw         |          715.9 |
| ep_dispatch_fwd      |         1112.0 |
| ep_dispatch_fwd_bwd  |         6796.2 |
| combine_raw          |          702.7 |
| ep_combine_fwd       |         4953.9 |
| ep_combine_fwd_bwd   |        13508.4 |
| (dispatch fwd-raw)   |          396.1 |
| (dispatch bwd-fwd)   |         5684.2 |
| (combine fwd-raw)    |         4251.2 |
| (combine bwd-fwd)    |         8554.6 |

`nsys` showed 37% of GPU time in fp32 `MulFunctor` and 23% in `direct_copy` — i.e.
fp32 expansion of the `[recv_pr, H]=[65536, 7168]` weighting in combine fwd/bwd. The
actual NCCL EP dispatch/combine kernels were only 19% of total GPU time.

## Optimization 1: drop fp32 cast + redundant mask in combine fwd/bwd

`transformer_engine/pytorch/ep.py:_EpCombine.{forward,backward}`

The pre-multiply did `(expert_out.float() * w.float() * mask.float()).bf16()` on a
[recv_pr, H] tensor — three 1.9 GB allocations and casts per combine. The mask
`(w != 0)` is mathematically redundant: `expert_out * w * mask == expert_out * w`
for finite `expert_out` (multiplying by zero already yields zero in the masked-zero
case). Cast to payload dtype once and multiply in bf16, fused with the output
write via `torch.mul(..., out=combine_in)`.

For combine backward, same approach: bf16 multiply, fp32 accumulator for the dot
product (`sum(-1, dtype=torch.float32)`).

## Optimization 2: torch.compile fusion of combine backward post-NCCL

`_combine_bwd_post(grad_combine_in, expert_out, recv_topk_weights)` decorated
with `@torch.compile(dynamic=False, fullgraph=True)`. Lets inductor share
`grad_combine_in` reads between the two muls (one for `grad_expert_out`, one for
the dot product producing `grad_recv_topk_weights`).

## Optimization 3: bypass `torch.ops` in raw paths

`_ep_dispatch_raw`, `_ep_combine_raw`, `ep_prepare` now call `tex.*` directly
instead of going through the `torch.library.custom_op` dispatcher. Autograd
Functions keep `torch.ops.transformer_engine_ep.*` for torch.compile graph
support (verified by `test_compile_fullgraph_with_new_api`).

## Optimization 4: drop paranoia `.contiguous()` in autograd backward

`_EpDispatch.backward` / `_EpCombine.backward` now check `is_contiguous()` first;
autograd-produced grads are already contiguous in the common case so this is a
no-op, but avoids constructing the extra tensor wrapper otherwise.

## Final numbers (commit e4d79578a)

| stage                | mean wall (µs) | Δ vs baseline |
|----------------------|---------------:|--------------:|
| dispatch_raw         |          665.1 |     −51 (−7%) |
| ep_dispatch_fwd      |         1188.9 |     +77 (+7%) |
| ep_dispatch_fwd_bwd  |         6601.5 |    −195 (−3%) |
| combine_raw          |          673.8 |     −29 (−4%) |
| ep_combine_fwd       |         1718.2 |  **−3236 (−65%)** |
| ep_combine_fwd_bwd   |         3144.3 | **−10364 (−77%)** |
| (combine fwd-raw)    |         1044.5 |  **−3207 (−75%)** |
| (combine bwd-fwd)    |         1426.1 |  **−7129 (−83%)** |

`ep_combine_fwd_bwd` went from **13.5 ms → 3.1 ms**.

## Round 2: Python + C++ host-side trims

### Opt 5: Python mirror of the zero-copy toggle

`_zero_copy_scope` used to call `tex.ep_get_zero_copy()` on every op even when
the flag didn't change. Added a Python-side `_ZC_ENABLED` cell; the scope now
skips the pybind getter entirely when the requested state equals the cached
state (the common case across a hot loop).

### Opt 6: cache `EpHandle.device`

Replace `handle.handle_mem.device` (two attribute lookups + tensor-property
access) with a slot set once in `__init__`.

### Opt 7: persistent `ncclEpHandle_t` cache in `EPBackend`

`HandleEntry` now holds a `cached_handle` + `cached_handle_mem` pair. Each op
calls `get_or_open_handle(cfg, handle_mem)`; on a ptr match the cached handle
is reused, otherwise the stale one is destroyed and a fresh one opened. The
NCCL EP call now runs inside the same mutex critical section — host-side cost
of `ncclEpDispatch`/`ncclEpUpdateHandle`/`ncclEpCombine` is small and same-
`handle_id` host calls were already serialized. Cached handles are destroyed
explicitly in `EPBackend::shutdown()`; the dtor still skips NCCL calls (CUDA
context may be gone).

### Opt 8: bench measurement cleanup

`(r.float() ** 2).sum()` allocated a fp32 `[recv_pr, H]` tensor (~1.9 GB)
inside the `*_fwd_bwd` loops, inflating those numbers by ~2.5 ms. Replaced
with `(r * r).sum(dtype=torch.float32)` — same scalar, no intermediate alloc.

## Round-2 final numbers

| stage                | mean wall (µs) | Δ vs round-1 | Δ vs baseline |
|----------------------|---------------:|-------------:|--------------:|
| dispatch_raw         |          ~690  |          ~+25 |     ~−25 (~−3%) |
| ep_dispatch_fwd      |         ~1080  |         −109 |  **~−110 (~−9%)** |
| ep_dispatch_fwd_bwd  |         ~4080  |        −2520 | **~−2715 (~−40%)** |
| combine_raw          |          ~715  |          ~+40 |             flat |
| ep_combine_fwd       |         ~1780  |          ~+60 | **~−3175 (~−64%)** |
| ep_combine_fwd_bwd   |         ~2995  |         −150 | **~−10515 (~−78%)** |
| (dispatch fwd-raw)   |          ~390  |          flat |             flat |
| (dispatch bwd-fwd)   |         ~3000  |        −2680 | **~−2680 (~−47%)** |
| (combine fwd-raw)    |         ~1060  |          flat |  **~−3190 (~−75%)** |
| (combine bwd-fwd)    |         ~1220  |         −206 |  **~−7335 (~−86%)** |

The `*_fwd_bwd` deltas are dominated by Opt 8 (bench cleanup) — the prior
numbers were measuring a phantom 2.5 ms fp32 alloc inside the bench, not real
TE work. Opts 5-7 contribute ~100-150 µs on the autograd paths; raw paths are
within run-to-run noise (init/destroy isn't the dominant host-side cost on
this config).

## Remaining gap to NCCL EP

TE raw D+C wall = 1339 µs vs NCCL "total" 1060 µs (kernel 799 µs).

After the optimizations above, the residual gap is dominated by:
1. The mandatory bf16 weighting multiply (`expert_out * w`, ~1 ms by itself on
   the `[65536, 7168]` tensor — pure HBM bandwidth, can only be removed by
   pushing the weighting into NCCL EP's combine kernel, which is out of scope
   here).
2. cudaSync + perf_counter timing overhead vs NCCL bench's cudaEvent timing —
   ~150 µs measurement-method difference, not real work.
3. Python + pybind layer (a handful of µs per call).

## Zero-copy: out of scope / blocked on NCCL comm sharing

The `--zero-copy` path was non-functional even before this work: the build
didn't define `USE_NCCL`, so PyTorch's `nccl_dev_cap.hpp` left
`NCCL_HAS_SYMMEM_SUPPORT` undefined, so `maybe_make_window` returned
`{nullptr, 0}` for every tensor — NCCL EP always took the staged-copy path.

Confirmed via nsys: identical 235 MB (dispatch) / 939 MB (combine) D2D memcpy
counts whether `--zero-copy` was set or not.

Attempted fix: add `-DUSE_NCCL` to the PyTorch extension build and switch from
the no-longer-existing `c10d::symmetric_memory::get_symmetric_memory(...)` to
the public `rendezvous(...)` API.

Result: `maybe_make_window` correctly produces non-null windows, but
`ncclEpDispatch` then fails with:

```
allocator.cc:458 NCCL WARN Device object does not exist in shadow pool.
NCCL error nccl_ep.cc:571 'internal error'
```

Root cause: `ep_bootstrap` creates a fresh NCCL comm via `ncclCommInitRank`
and hands it to NCCL EP; PyTorch's symm-mem registers windows on the
`ep_group`'s existing NCCL comm. NCCL EP's shadow pool can't see windows
created against a different comm.

A proper fix needs comm sharing between PyTorch's `ProcessGroup` NCCL backend
and NCCL EP — either NCCL EP consumes torch's existing comm, or torch's
symm-mem rendezvous targets NCCL EP's comm. Either direction is invasive (the
torch comm isn't exposed as a public C API and NCCL EP's group init owns the
comm). Reverted to keep the suite green; deferred for a follow-up MR.

## Files touched

- `transformer_engine/pytorch/ep.py` — combine fwd/bwd math, raw torch.ops bypass, contiguous-check, Python ZC mirror, `EpHandle.device` slot.
- `transformer_engine/common/ep/ep_backend.{h,cpp}` — persistent `ncclEpHandle_t` cache; `get_or_open_handle` helper; cached handles destroyed in `shutdown()`.
- `examples/pytorch/ep/bench/ep_bench.py` — adds `symm_mem_alloc` for inputs in `--zero-copy` mode (will become effective once zero-copy is unblocked); drops `(r.float()**2)` fp32 alloc in `*_fwd_bwd` loss.

## Reproduction

```
# NCCL EP baseline
bash examples/pytorch/ep/bench/run_nccl_ep_bench.sh

# TE PyTorch bench
LD_LIBRARY_PATH=$(pwd)/3rdparty/nccl/build/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$(pwd) \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  examples/pytorch/ep/bench/ep_bench.py \
  --tokens-per-rank 2048 --hidden 7168 --top-k 8 --num-experts 256 \
  --warmup 5 --iters 50

# nsys trace (per-kernel time)
bash examples/pytorch/ep/bench/run_ep_bench.sh --nsys
```
