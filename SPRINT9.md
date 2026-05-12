# EP Sharding Sprint v9

## Goal

Make `tests/jax/test_multi_process_ep_sharded.py` pass on the 8-GPU box
under a `(dp=1, ep=8)` mesh — a single 8-rank EP group with no DP axis.
This unblocks TP-like configs where the whole node is one EP domain.
SPRINT8 closed the 2×4 case via per-DP-color `ncclCommInitRank` but
left 1×8 failing on two distinct symptoms: a build-time
`_NCCL_EP_LSA_TEAM_SIZE_MAX` cap and a runtime
`CUDA_ERROR_LAUNCH_FAILED` inside the `lsa_team_size=8` scan kernel.

## Context

Repo state at sprint start (post SPRINT8, commit `5267463cd`):

- `EPBackend::initialize(uid, ep_size, rank_within_group, cfg)` calls
  `ncclCommInitRank` directly (no `ncclCommSplit`). `(dp=2, ep=4)` and
  `(dp=1, ep=4)` both green; `2×4` runs in ~31s.
- NCCL submodule pin at `fe8b29710` includes the cosmetic
  `rank_in_node = rank % gpus_per_node` fix.
- `setup.py` line 191-201 sets
  `_NCCL_EP_LSA_TEAM_SIZE_MIN=4`, `_NCCL_EP_LSA_TEAM_SIZE_MAX=8` via
  `setdefault` when building `libnccl_ep.so`.
- `_NCCL_EP_LSA_TEAM_SIZE_*` controls
  `HYBRIDEP_SWITCH_LSA_TEAM_SIZE` template instantiation in
  `3rdparty/nccl/contrib/nccl_ep/device/hybridep_adapter.cuh:130-189`.
  Default Makefile MAX is 32; setup.py narrows to 8 to shrink build
  time.
- Hardware: B300 SXM6 AC, compute_cap `10.3` (sm_103). nvcc 13.2.78
  supports `sm_103` directly.

Observed for `(dp=1, ep=8)` end of SPRINT8:
1. With stricter `MAX=4` an assert at
   `hybridep_adapter.cu:416` fires (`lsa_team_size=8` not in
   `[MIN, MAX]`).
2. With `MAX≥8` but only `sm_90` SASS, the scan kernel reports
   `'named symbol not found'` from `hybrid_ep.cuh:4382`.
3. With `compute_90,sm_90 compute_100,sm_100` SASS the symbol is
   found but kernel launches with `CUDA_ERROR_LAUNCH_FAILED` —
   possible kernel bug at LSA_TEAM_SIZE=8, OR runtime arch mismatch
   (B300 is sm_103, lib has only sm_100 SASS, no PTX).

Pre-sprint inspection of the currently linked
`/home/scratch.phuonguyen_sw/te/3rdparty/nccl/build/lib/libnccl_ep.so`:

```
$ cuobjdump --dump-elf-symbols libnccl_ep.so | grep "scan.*Li1ELi8E" | wc -l
4   # scan<512, 16, 1, 8, false/true> present (num_lsa_teams=1, lsa_team_size=8)
$ cuobjdump libnccl_ep.so | grep "arch ="
arch = sm_90
arch = sm_100        # NO sm_103, NO PTX
```

So the lib ALREADY contains the `lsa_team_size=8` instantiation
(symptom (2) is gone), but ships only sm_90 / sm_100 SASS for a
sm_103 device. The most defensible first probe is: rebuild with
`NVTE_CUDA_ARCHS` including `103` and re-run 1×8.

## Hypotheses (ordered by prior probability)

H1. **Missing SASS for sm_103.** `libnccl_ep.so` has sm_90 + sm_100
    elf but no PTX and no sm_103 binary. sm_100 SASS is NOT
    forward-compatible to sm_103 (different major.minor binary).
    Some smaller `lsa_team_size` paths may not hit this hot path at
    all (kernel selected by team size differs in occupancy /
    register footprint), which would explain why 4 works and 8
    fails. Fix: add sm_103 to gencode via TE's setup.py
    (`NVTE_CUDA_ARCHS="90;100;103"`).

H2. **Genuine LSA_TEAM_SIZE=8 scan kernel bug.** After H1's rebuild,
    if 1×8 still LAUNCH_FAILEDs, the kernel itself is buggy at
    team_size=8. Verify by either (a) inspecting the scan launch
    config (threads/block, dynamic shared mem) vs. B300 limits, or
    (b) bisecting against the NCCL EP upstream `main` (our pin is
    `fe8b29710`).

H3. **`_NCCL_EP_LSA_TEAM_SIZE_MAX` is silently overridden by an env
    var in the build harness.** `setup.py` uses `setdefault`, so a
    pre-existing `_NCCL_EP_LSA_TEAM_SIZE_MAX=4` in the shell would
    keep `case 8:` out of the binary entirely. We already verified
    by `cuobjdump` that `scan<...,1,8,...>` symbols ARE present in
    the linked lib — H3 is unlikely but worth a one-line audit
    while we're rebuilding.

## Assumptions

A1. NCCL EP's 1×8 path is supposed to work on Hopper+ — at minimum
    upstream tested it on H100/H200. We are not on entirely new
    ground.
A2. The TE side is already correct: per-DP-color bootstrap returns a
    single ep_size=8 comm when `dp=1, ep=8`, identical to the
    PyTorch path. No further TE changes likely needed.
A3. We will NOT modify NCCL EP source in this sprint. Fixes are
    confined to TE's setup.py (gencode/flags), to env-var hygiene,
    and to any necessary diagnostic prints behind `#ifdef`s.

## Sprint items

### 1. Verify the linked NCCL EP lib has the right gencode + LSA size [MUST]

Citations:
- `3rdparty/nccl/contrib/nccl_ep/Makefile:48-51` — env knob plumbing.
- `setup.py:191-201` — `_NCCL_EP_LSA_TEAM_SIZE_*` defaults in
  setup.py's NCCL EP build path.
- `setup.py:169-177` — `NVTE_CUDA_ARCHS` → `NVCC_GENCODE` propagation.

Investigation:
- Inspect current `libnccl_ep.so` with
  `cuobjdump --dump-elf-symbols` for `scan<...,1,8,...>` symbols
  and `cuobjdump` for the embedded `arch = sm_*` fatbins.
- Confirm setup.py's `setdefault` path is what gets executed; if env
  pollution is suspected, audit `env` shell state.
- Force a clean rebuild via:
  ```
  rm -f 3rdparty/nccl/build/lib/libnccl_ep.so
  NVTE_NCCL_EP_REBUILD=1 NVTE_WITH_NCCL_EP=1 \
    NCCL_EP_DIR=$PWD/3rdparty/nccl/build NVTE_BUILD_NCCL_CORE=0 \
    NVTE_CUDA_ARCHS="90;100" \
    pip install --no-build-isolation -e .
  ```
  Then re-inspect.

Acceptance: a one-paragraph note here recording (a) which scan
template instantiations are in the lib, (b) which `sm_*` arches,
(c) which `_NCCL_EP_LSA_TEAM_SIZE_*` values the build picked up.

### 2. Add sm_103 (and/or PTX fallback) to NCCL EP gencode [MUST, conditional on #1]

Citations:
- `setup.py:169-177` — `NVTE_CUDA_ARCHS` parsing in
  `_build_nccl_libs`. Today: drops anything not pure-digit ≥ 90
  via `a.strip().rstrip("af")`.

Sketch:
- Run a rebuild with `NVTE_CUDA_ARCHS="90;100;103"` and confirm via
  `cuobjdump` that `arch = sm_103` appears in libnccl_ep.so.
- Re-run `tests/jax/test_multi_process_ep_sharded.py` with 8 procs.
  Setup.py needs to honor `103` — verify the gencode list.
- If `sm_103` is rejected by setup.py's arch filter (it strips
  `f`/`a` suffixes, so `103` should pass), no setup.py change needed.
  Else, widen the filter.
- Consider always emitting PTX (`-gencode=arch=compute_90,code=compute_90`)
  for forward-JIT fallback. Trade-off: ~30s extra build time, but
  protects against future archs.

Acceptance: 1×8 sharded test passes OR the LAUNCH_FAILED reproduces
with confirmed sm_103 SASS in the lib (which would refute H1 and
hand off to #3).

### 3. If H1 refuted, instrument the LAUNCH_FAILED [LIKELY conditional]

Citations:
- `3rdparty/nccl/contrib/nccl_ep/device/hybrid_ep.cuh:4382` — scan
  kernel launch site.
- `device/hybridep_adapter.cu:416` — the runtime team-size assert.

Sketch:
- Capture full `CUDA_LAUNCH_BLOCKING=1` stderr from the failing
  scan call. Look for grid/block dims, dynamic shmem.
- Add a one-line `fprintf(stderr,...)` ahead of the launch
  (NCCL EP-side) dumping `(grid, block, dynShmem,
  cudaDeviceGetAttribute(maxSharedPerBlock))` — purely diagnostic,
  not merged. Bisect against a smaller config if applicable.
- If shmem requested exceeds B300 limit, file an upstream NCCL EP
  bug; for the sprint, document the cap and gate the test mesh
  selection.

Acceptance: a root-cause one-paragraph note under Findings — either
a kernel correctness bug, a launch-config mismatch on B300, or a
device-property limit.

### 4. Surface arch / LSA env in the JAX bootstrap log [NICE-TO-HAVE]

Citations:
- `transformer_engine/jax/ep.py:109-117` — `initialize_ep_communicator`
  call site.

Sketch:
- Add an opt-in `os.environ.get("NVTE_EP_VERBOSE")` branch in
  `ep_bootstrap` to print the negotiated `(world_size, ep_size,
  rank_within_group, dp_color)` plus a `cuobjdump`-derived hint
  about the linked NCCL EP archs (via a Python-level dlopen +
  `cudaDeviceGetAttribute` cross-check). Defaults off.

Acceptance: with `NVTE_EP_VERBOSE=1`, a single line per rank
captures enough state to triage future bootstrap mismatches in 30s.

## Constraints honored

- `NVTE_CUDA_ARCHS="90;100"` minimum for any rebuild (drop `;103`
  only on non-B300 boxes). Single-arch is forbidden — breaks B300.
- `pip install --no-build-isolation -e .` ONLY. Never the editable
  wheel-build path.
- NCCL EP rebuild ONLY via TE's setup.py env-var path (set
  `NVTE_NCCL_EP_REBUILD=1`) — never hand-run
  `make -C 3rdparty/nccl/contrib/nccl_ep` because gencode +
  LSA_TEAM_SIZE flags must stay in lock-step with TE's link.
- `git commit -s` single-line messages, no AI attribution.
- Author / committer = `Phuong Nguyen <phuonguyen@nvidia.com>`.
- No drive-by refactors. One-line comments only.
- No destructive git ops.

## Findings

### Item 1 — pre-rebuild inventory of current libnccl_ep.so

`/home/scratch.phuonguyen_sw/te/3rdparty/nccl/build/lib/libnccl_ep.so`
at the start of SPRINT9:

```
$ cuobjdump --dump-elf-symbols libnccl_ep.so | grep "scan.*Li1ELi8E" | wc -l
4
$ cuobjdump libnccl_ep.so | grep "arch ="
arch = sm_90   (x2: dispatch + compressed)
arch = sm_100  (x2: dispatch + compressed)
$ cuobjdump --dump-ptx libnccl_ep.so | grep -c "target sm_"
0
```

- 4 `scan<512, 16, 1, 8, ...>` instantiations present →
  `_NCCL_EP_LSA_TEAM_SIZE_MAX=8` is honored, contradicting the
  build-flag suspicion in SPRINT8 #1's third symptom.
- Lib carries sm_90 and sm_100 SASS only, NO PTX. B300 is
  compute_cap 10.3 (sm_103). sm_100 SASS is not minor-binary
  compatible to sm_103 in general. This is the single most likely
  reason 1×8 trips `CUDA_ERROR_LAUNCH_FAILED` while 1×4 passes —
  4-team and 8-team scans differ in which SASS variants the
  CUDA driver picks, and B300 may JIT/fallback differently for
  the larger team.

Acceptance partially met (a, b, c documented). Next concrete probe
is to add sm_103 to gencode and re-inspect.

### Item 1/2 — sm_103 rebuild attempt via setup.py: BLOCKED

Tried (commit-only sprint plan in place):

```
rm -f 3rdparty/nccl/build/lib/libnccl_ep.so
NVTE_NCCL_EP_REBUILD=1 NVTE_WITH_NCCL_EP=1 \
  NCCL_EP_DIR=$PWD/3rdparty/nccl/build NVTE_BUILD_NCCL_CORE=0 \
  NVTE_CUDA_ARCHS="90;100;103" \
  _NCCL_EP_LSA_TEAM_SIZE_MIN=4 _NCCL_EP_LSA_TEAM_SIZE_MAX=8 \
  pip install --no-build-isolation -e .
```

Result: TE's CMake step (`build_tools/build_ext.py:97`) exited
non-zero. The NCCL EP make step inside setup.py's
`_build_nccl_libs` did NOT produce a new `libnccl_ep.so` either
(file removed by the cleanup line and never regenerated). The
log tail captured only the Python traceback; the CMake stderr
from before that point was truncated by setuptools' output
buffering — re-run with `pip install -v` or pipe to file for the
full failure to be visible.

Follow-up (carried into SPRINT9 item #2 execution next session):
- Re-run with explicit log capture: `... pip install -v
  --no-build-isolation -e . 2>&1 | tee /tmp/sprint9-build.log`
  and grep `error:` for the first CMake failure.
- Possible causes worth checking:
  - sm_103 needs `compute_103,sm_103` AND the toolchain's CCCL
    headers may require sm_100a; nvcc 13.2 lists `sm_103a` and
    `sm_103f` as separate variants.
  - TE's CMake might filter `NVTE_CUDA_ARCHS` separately from
    setup.py's NCCL EP path, and the value "103" could be
    rejected by the TE C++ build but accepted by NCCL EP's. The
    setup.py path inspected (`setup.py:169-177`) is the NCCL EP
    side only.
- After fixing the build, re-run inventory:
  `cuobjdump --dump-elf-symbols libnccl_ep.so | grep "scan.*Li1ELi8E"`
  and `cuobjdump | grep "arch ="` should both show sm_103.

State to leave to next session: the lib was deleted, build did
NOT regenerate it. Running any EP test now will fail at link
time. Either re-run the rebuild after diagnosing the CMake
failure, OR `git checkout` is not applicable (build artifact is
gitignored). The build_dir is at
`/home/scratch.phuonguyen_sw/te/build/cmake` — its log files
should contain the original CMake error.

