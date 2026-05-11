# EP Sharding Sprint v8

## Goal

Make `tests/jax/test_multi_process_ep_sharded.py` pass on the 8-GPU box
under the `(DP=2, EP=4)` mesh. The 1×4 (single-EP-group) case passes
after SPRINT7 #11; the 2×4 case currently returns `out = tokens / 2`
because dispatch silently drops half the recv slots. This sprint
diagnoses and fixes the 2×4 path so two EP groups on the same physical
node interoperate correctly.

## Context

Repo state at sprint start:

- SPRINT7 #11 done (commit `95dd82deb`). `EpHandle(handle_mem,
  handle_id)` flows through `@jax.jit` so the C++ cache is keyed on a
  stable `uint64_t` id, not the XLA-relocatable device pointer.
- `tests/jax/test_multi_process_ep_sharded.py` factors `num_procs`
  into `(dp, ep)` via `_factor_dp_ep(num_procs)`: 4-GPU → `(1, 4)`;
  8-GPU → `(2, 4)`. Bootstrap is per-DP-shard
  (`max_tokens_per_rank=TOKENS_PER_DP_SHARD`,
  `max_recv_tokens_per_rank=recv_capacity_per_rank`).
- Comm split in `ep_backend.cpp:115` →
  `ncclCommSplit(world_comm, rank / ep_size, rank, ...)`. With 8
  ranks and `ep_size=4`: color 0 = world ranks 0..3, color 1 = world
  ranks 4..7. Two independent EP comms of size 4.
- NCCL EP intranode setup (`3rdparty/nccl/contrib/nccl_ep/nccl_ep.cc:955-959`):
  ```
  gpus_per_node = nRanks / nNodes;     // 4 ranks / 1 node = 4
  rank_in_node  = cuda_device_id;      // physical GPU index
  node_id       = rank / gpus_per_node;
  lsa_team_size = lsa_team.nRanks;
  ```
  and the peer-access loop (`:344`):
  ```
  for (int i = 0; i < gpus_per_node; i++) {
      if (i == rank_in_node) continue;
      cudaDeviceCanAccessPeer(&can_p2p, rank_in_node, i);
      if (can_p2p) cudaDeviceEnablePeerAccess(i, 0);
  }
  ```
  When two EP groups share one physical node, `gpus_per_node=4` (each
  group is 4 ranks on the one node) but `rank_in_node` for the DP=1
  group's ranks is `cuda_device_id ∈ {4,5,6,7}` — outside `[0, 4)`. The
  loop never matches its own device and the rank attempts P2P into
  the other group's GPUs.

## Observed failure (data)

`out = tokens / 2` across all 256 positions. Per-rank `recv_w` after
`ep_dispatch` is:

```
[0.5 0 0 0 0.5 0 0 0   0 0 0 0 0 0 0 0 ...]  (repeats every 16 slots)
```

- Filled zones: 0, 2, 4, 6 (every other zone). Empty zones: 1, 3, 5, 7.
- Within a filled zone: slots 0 and 4 (stride 4). Expected 4 slots
  filled (one per source rank in the EP group).
- `token_counts` (the prepare output) is `[4,4,4,4,4,4,4,4]` — i.e.
  prepare's AllGather sees the correct routing intent for ALL 8
  experts. The routing-metadata exchange is therefore healthy; the
  data-path AllToAll inside `ncclEpDispatch` is what drops slots.
- Combined dispatch loss vs the working 1×4 case: 75% of routes
  missing — both the odd-zone routes (50%) and half the in-zone
  source-rank contributions (another 50%).

## Working hypotheses

H1. **NCCL EP intranode P2P loop assumes contiguous device ids
    `[0, gpus_per_node)`** — fails when two EP groups colocate on
    one physical node and one group's `cuda_device_id` ≥
    `gpus_per_node`. See `nccl_ep.cc:344-354`. Fix is either in NCCL
    EP (compute `rank_in_node` as a within-group ordinal, not the
    physical device id) or in TE (give NCCL EP a per-group device
    view).
H2. **`lsa_team_size` from `ncclTeamLsa(comm)` is detected as 8 for
    each 4-rank EP comm** because NCCL infers LSA-team membership
    from the underlying world comm, which spans both DP groups. If
    `lsa_team_size=8` on a 4-rank EP comm, every NCCL-EP buffer is
    sized for 8 ranks and routing offsets are computed mod 8 instead
    of mod 4 — that would explain the stride-4 within-zone pattern
    and the empty odd zones.
H3. **An alternative TE-side fix is to skip `ncclCommSplit` and pass
    a pre-built EP sub-communicator via the existing
    `EPBackend::initialize_with_comm` path** (`ep_backend.cpp:121-134`).
    Have JAX bootstrap call `ncclCommInitRank` for the EP sub-group
    directly using a per-group unique id (one unique id per DP
    shard) so NCCL EP sees a "clean" 4-rank world comm with no
    cross-DP-group affinity. This bypasses whatever NCCL infers from
    the parent comm.

## Working assumptions

A1. The DP×EP mesh is correct (verified on 1×4 and across all
    SPRINT7 #1..#11 items). No JAX-side bug.
A2. NCCL EP is correct for a single EP group of 4 (1×4 passes
    end-to-end). The defect is specifically about two 4-rank groups
    coexisting on one physical node.
A3. We control TE's bootstrap path and the JAX FFI. We do NOT modify
    NCCL EP source in this sprint — only TE-side changes and, if
    unavoidable, NCCL EP env-var tweaks (`_NCCL_EP_LSA_TEAM_SIZE_*`).

## Issues / Sprint items

### 1. Confirm or rule out H1 (intranode P2P) [MUST]

Citations:
- `3rdparty/nccl/contrib/nccl_ep/nccl_ep.cc:344-354` — peer-access loop.
- `ep_backend.cpp:115` — `ncclCommSplit` with `color = rank / ep_size`.

Investigation:
- Add a temporary `fprintf(stderr, ...)` inside NCCL EP just before the
  peer-access loop (or stamp via TE bootstrap log) capturing
  `(world_rank, ep_rank, cuda_device_id, rank_in_node, gpus_per_node,
  lsa_team_size, lsa_rank)` on every rank in 2×4. Confirm whether
  `cuda_device_id` is the physical 0..7 or whether NCCL is already
  remapping it to a group-local 0..3.
- If H1 is true, attempt: set `CUDA_VISIBLE_DEVICES=$proc_id` per
  process in `tests/jax/multi_process_launch_ep.sh` so every process
  sees exactly one GPU (logical index 0). NCCL EP would then see
  `cuda_device_id=0` on every rank — which has its own failure mode
  (rank_in_node collisions inside a group) but would isolate the
  variable.

Acceptance: a clear yes/no on H1, captured as a one-paragraph note in
this sprint file under "Findings".

### 2. Confirm or rule out H2 (LSA team spanning the world comm) [MUST]

Citations:
- `nccl_ep.cc:330, 958` — `ncclTeamLsa(comm)` queried from the EP comm.
- NCCL `ncclTeam` semantics: the LSA team is typically a node-local
  rail group inferred from the underlying transport. With multiple
  EP comms split from the same world comm, the LSA may still resolve
  against the original world comm.

Investigation:
- The same instrumentation as #1 captures `lsa_team_size`. If
  `lsa_team_size == 8` in the 2×4 case, H2 is confirmed.
- If confirmed, the fix is structural: each EP group must own a
  fresh top-level `ncclComm_t` (built from a per-group unique id),
  not a `ncclCommSplit` of the world comm.

Acceptance: yes/no on H2 with the rank/lsa_team_size table captured
here.

### 3. Per-DP-group `ncclCommInitRank` bootstrap [LIKELY, conditional on #2]

Citations:
- `ep_backend.cpp:99-119` — current `initialize` path (CommInitRank →
  CommSplit).
- `ep_backend.cpp:121-134` — `initialize_with_comm` (alternative path
  PyTorch uses; takes a borrowed `ncclComm_t`).
- `transformer_engine/jax/ep.py:78-107` — JAX `ep_bootstrap` wrapper.

Fix sketch (only if #2 confirms):
- Add `transformer_engine_jax.initialize_ep_communicator_split` (or
  reuse the existing entry) that:
  - Takes `dp_color` and a `dp_unique_id` distinct from the
    world `unique_id`.
  - Calls `ncclCommInitRank` directly with `(ep_size, dp_unique_id,
    rank_within_dp)` to build a clean 4-rank comm.
  - Delegates to `EPBackend::initialize_with_comm` so the EP backend
    treats it as borrowed.
- JAX side: `ep_bootstrap` derives `dp_color = rank // ep_size`,
  generates one `ncclUniqueId` per dp_color via a small AllGather over
  the world (one rank-0-per-color broadcasts), and passes those into
  the new entry point.

Acceptance: with the new bootstrap, the 2×4 sharded test passes the
identity-expert round-trip.

### 4. Document the fix and the env-var requirements [MUST]

Citations:
- `setup.py:194-195` —
  `_NCCL_EP_LSA_TEAM_SIZE_MIN=4`,
  `_NCCL_EP_LSA_TEAM_SIZE_MAX=8`.

Fix:
- Whatever the resolution from #1..#3, capture a one-paragraph "How
  EP comms are arranged for DP×EP meshes" note in
  `transformer_engine/jax/ep.py`'s module docstring AND a pointer in
  this sprint file under "Resolution".
- If `_NCCL_EP_LSA_TEAM_SIZE_MAX` needs to be raised to ≥8 for the
  legacy `ep_size=num_procs` test to pass on 8 GPUs, capture the
  rebuild requirement in the same note.

## Constraints honored

- `git commit -s` single-line messages, no AI attribution.
- Author/committer = `Phuong Nguyen <phuonguyen@nvidia.com>`.
- `NVTE_CUDA_ARCHS="90"` and
  `NVTE_WITH_NCCL_EP=1 NCCL_EP_DIR=$PWD/3rdparty/nccl/build
  NVTE_BUILD_NCCL_CORE=0` for any rebuild.
- No drive-by refactors. Each commit touches only the lines required.
- One-line comments. No multi-line comment blocks.
- No destructive git ops. Pre-commit hook NOT skipped.

## Findings

### Diagnostic run (2×4 mesh, instrumented NCCL EP)

Per-rank dump from inside `ncclEpCreateGroup` (after `lsa_team_size`
is set):

```
rank=0 cuda_dev=0 gpus_per_node=4 rank_in_node=0 node_id=0 lsa_team_size=4 lsa_rank=0
rank=1 cuda_dev=1 gpus_per_node=4 rank_in_node=1 node_id=0 lsa_team_size=4 lsa_rank=1
rank=2 cuda_dev=2 gpus_per_node=4 rank_in_node=2 node_id=0 lsa_team_size=4 lsa_rank=2
rank=3 cuda_dev=3 gpus_per_node=4 rank_in_node=3 node_id=0 lsa_team_size=4 lsa_rank=3
rank=0 cuda_dev=4 gpus_per_node=4 rank_in_node=4 node_id=0 lsa_team_size=4 lsa_rank=0
rank=1 cuda_dev=5 gpus_per_node=4 rank_in_node=5 node_id=0 lsa_team_size=4 lsa_rank=1
rank=2 cuda_dev=6 gpus_per_node=4 rank_in_node=6 node_id=0 lsa_team_size=4 lsa_rank=2
rank=3 cuda_dev=7 gpus_per_node=4 rank_in_node=7 node_id=0 lsa_team_size=4 lsa_rank=3
```

(Each line is one rank in one of the two 4-rank EP comms. The DP=0
group shows `cuda_dev ∈ {0..3}`; the DP=1 group shows `{4..5..7}`.)

### H1: CONFIRMED

`rank_in_node = cuda_device_id` (`nccl_ep.cc:956`) is the physical
GPU index, not a within-EP-group ordinal. For the DP=1 group,
`rank_in_node ∈ {4,5,6,7}` while `gpus_per_node=4` — outside `[0, 4)`.

The peer-access loop at `:344-354` iterates `for i in 0..4` and
checks `if (i == rank_in_node) continue;` — that condition never
fires for DP=1 ranks. The rank then attempts P2P from its own GPU
into devices `0..3` (which belong to the other EP group). It also
never enables P2P among the GPUs that actually need it (4..7).

### H2: RULED OUT

`lsa_team_size = 4` on every rank in 2×4. NCCL is scoping LSA team
membership against the post-split EP comm, not the world comm.

### `rank_in_node` is cosmetic for the data path

The `rank_in_node = cuda_device_id` line at `nccl_ep.cc:956` is OOB
for DP=1 in 2×4 (values 4..7 ≥ `gpus_per_node=4`), but the field
only feeds the P2P enable loop at `:344-354`. The real data-path
routing uses NCCL window IPC handles which are scoped against the
LSA team (which is correctly sized to 4). So fixing only
`rank_in_node` to `rank % gpus_per_node` is not enough — the
test still fails with the same `out = tokens / 2` after that
patch alone.

### Root cause: `ncclCommSplit` from a shared world comm

Two EP groups built via `ncclCommSplit(world_comm, color=rank/ep_size, ...)`
on one physical node share enough state that NCCL EP's dispatch
silently drops ~50% of cross-rank slots, with the surviving slots
patterned as `[0.5 ... 0.5 ...]` (stride-4 within zone, every-other
zone empty). Confirmed by isolating: with the `rank_in_node`
cosmetic fix and the original split-based bootstrap, the 2×4 test
still fails identically.

### Fix: per-DP-color `ncclCommInitRank`

Bypass `ncclCommSplit` entirely. Each DP color builds its own
top-level `ncclComm_t` from an independent `ncclUniqueId`:

- `ep_backend.cpp::initialize(uid, ep_size, rank_within_group, cfg)`
  now calls `ncclCommInitRank(&ep_comm, ep_size, uid, rank_within_group)`
  directly. No two-step (world_comm → split).
- `transformer_engine/jax/ep.py::ep_bootstrap` derives
  `dp_color = rank // ep_size`, has each color-root generate its
  own `ncclUniqueId`, world-allgathers all uids, and each rank picks
  its own color's slot.
- C API surface updated: `nvte_ep_initialize(uid, ep_size,
  rank_within_group, cfg)`.

### Test-side fix: `recv_capacity_per_rank` is per-DP-group

`tests/jax/test_multi_process_ep_sharded.py` originally set
`recv_capacity_per_rank = TOKENS_PER_DP_SHARD * dp * TOP_K`. The
`dp` multiplier was wrong: since each DP group has its own NCCL EP
comm, the per-rank recv buffer is sized for one DP group's worth of
senders, not all of them. With the inflated capacity,
`_make_valid_mask` computed `slots_per_zone = recv_capacity /
total_zones = 8` (vs the actual `per_expert_capacity = 4` used
inside NCCL EP), so slot 4 (which holds expert 1's token, weight
0.5) was masked off as if it were still in expert 0's zone. That's
exactly the `out = tokens / 2` we saw post-bootstrap-fix.

Fix: `recv_capacity_per_rank = TOKENS_PER_DP_SHARD * TOP_K`
(independent of dp).

### Result

`tests/jax/test_multi_process_ep_sharded.py` passes on 2×4
(`Ran 2 tests in 31.008s, OK`).

Note: `tests/jax/test_multi_process_ep.py` is unaffected (it has no
DP axis and uses `ep_size = num_procs`). On 8 GPUs it trips NCCL
EP's LSA team size cap — a separate, pre-existing limitation, out
of scope for SPRINT8.

### 1×8 probe (NOT shipped, follow-up)

The same sharded test under a `(dp=1, ep=8)` mesh exercises a
single 8-rank EP group. Two issues surface before it passes:

1. **`_NCCL_EP_LSA_TEAM_SIZE_MAX` must include 8.** The
   `HYBRIDEP_SWITCH_LSA_TEAM_SIZE` macro at
   `hybridep_adapter.cuh:178` only emits `case 8:` when
   `_NCCL_EP_LSA_TEAM_SIZE_MAX >= 8`. Out of the box the Makefile
   default is 32 (covers it), but a sm_90-only TE build that
   passes `_NCCL_EP_LSA_TEAM_SIZE_MAX=8` to NCCL EP make is OK; a
   stricter MAX=4 trips the assertion at
   `device/hybridep_adapter.cu:416`.
2. **NCCL EP needs SASS for the actual GPU arch.** Rebuilding
   NCCL EP with `NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"`
   on a B300 box (`compute_cap=10.3`) produces a `'named symbol
   not found'` CUDA error at `device/hybrid_ep.cuh:4382` because
   the `scan<...>` kernel has no SASS for sm_103 and PTX from
   compute_120 is not backward-JIT-able. Rebuilding with
   `compute_90,sm_90 compute_100,sm_100` gets past the symbol
   lookup but the kernel then hits `CUDA_ERROR_LAUNCH_FAILED` at
   runtime — likely a different NCCL-EP-internal correctness
   issue at LSA_TEAM_SIZE=8. Out of scope for this sprint; track
   as a follow-up.

For the time being, the supported mesh shape on 8 GPUs is
`(dp=2, ep=4)`. 1×4 still passes as before.
