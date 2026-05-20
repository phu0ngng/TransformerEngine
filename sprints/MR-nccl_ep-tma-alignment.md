# nccl_ep: round S2G prefetch copy size up to 16B for TMA alignment

## Summary

HT dispatch deadlocks when `num_of_tokens_per_rank` is odd on a single-node
EP group with 4 ranks. Root cause is a `cp.async.bulk` (TMA) in the S2G
warp group prefetching `sparse_to_dense_map` rows from global → shared:
the copy size is `current_chunk_size * s2d_inner_dim * sizeof(int32_t)`,
which is only a multiple of the required 16-byte TMA quantum when the
per-rank token count is even (for `s2d_inner_dim == 2`, top-k=2 EM).
On an odd count the TMA never advances the mbarrier's tx-count, S2G spins
on `mbarrier_try_wait_parity`, the dispatch grid barrier never sees the
block arrive, and all ranks deadlock waiting on
`intra_node_write_completion_flags`.

The fix is a one-liner per prefetch site: round the copy size up to the
next multiple of 16. The source `sparse_to_dense_map` is over-allocated
(sized for `max_num_of_tokens_per_rank`, aligned to 256 B) and the SMEM
destination stage is padded to 128 B, so the padding bytes are within
bounds for both sides. The per-token loop downstream only iterates up to
`current_chunk_size`, so the extra bytes are never read.

## Implementation

`contrib/nccl_ep/device/hybrid_ep.cuh`, two sites in
`S2G_warp_group_device_function`:

```cpp
// Initial-chunk prefetch (~line 1404).
uint32_t copy_bytes = (uint32_t)(current_chunk_size * s2d_inner_dim * sizeof(int32_t));
copy_bytes = (copy_bytes + 15u) & ~15u;
```

```cpp
// Next-chunk prefetch (~line 1481).
uint32_t copy_bytes = (uint32_t)(next_chunk_size * s2d_inner_dim * sizeof(int32_t));
copy_bytes = (copy_bytes + 15u) & ~15u;
```

No other changes; behavior for even counts is identical (already 16-aligned).

## Testing

Bisect confirmed odd-vs-even per-rank token count is the trigger
(NUM_LOCAL_EXPERTS=2, HIDDEN_DIM=32, single 4-rank EP group):

| per-rank tokens | before fix | after fix |
|---|---|---|
| 1 | deadlock | pass |
| 2 | pass | pass |
| 3 | deadlock | pass |
| 4 | pass | pass |
| 5 | deadlock | pass |
| 6 | pass | pass |

- `tests/cpp_distributed/run_test_ep.sh 4` (T=64 — was already aligned): still passes (9/9).
- TE-JAX `tests/jax/multi_process_launch_ep.sh` mesh `1x4`
  (NUM_LOCAL_EXPERTS=2, HIDDEN_DIM=32, T=4, top-k=2): previously deadlocked
  at `test_combine_vjp_fwd_bwd`; now 11/11 pass.
- TE-JAX `2x2` mesh: still 11/11 (regression check).

Why C-API gtests didn't catch this: they always use `max_tokens_per_rank=64`
producing 16-aligned `copy_bytes` for `s2d_inner_dim ∈ {1,2,4}`. The hang
only surfaces at small token counts that real JAX MoE workloads do hit.
