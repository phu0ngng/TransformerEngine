# TE × NCCL EP (JAX) — Build & Test

Short reference for building TransformerEngine with NCCL EP for JAX
and running the multi-process EP tests.

## Requirements

- Runtime device with compute capability ≥ 9.0 (Hopper or newer;
  H100, H200, B200, B300, GH200). Enforced at runtime in
  `EPBackend::validate_config` — NOT at build time. A wheel built
  for many archs loads fine on pre-Hopper, only errors when EP is
  actually invoked.
- NCCL EP submodule under `3rdparty/nccl/contrib/nccl_ep` (already
  pinned in this repo).
- NVLink mesh fabric (NVLS multicast). Single-rail PCIe boxes are
  not supported by NCCL EP.

## Build

Standard editable install with NCCL EP wired in:

```bash
cd /path/to/te
NVTE_WITH_NCCL_EP=1 \
NCCL_EP_DIR=$PWD/3rdparty/nccl/build \
NVTE_BUILD_NCCL_CORE=0 \
NVTE_CUDA_ARCHS="90;100" \
pip install --no-build-isolation -e .
```

Key env vars:

| Var                       | Default | Meaning                                              |
| ------------------------- | ------- | ---------------------------------------------------- |
| `NVTE_WITH_NCCL_EP`       | `0`     | `1` enables NCCL EP. (SPRINT9 #5 will flip to `1`.)  |
| `NCCL_EP_DIR`             | unset   | Path to NCCL build dir (`include/`, `lib/`).         |
| `NVTE_BUILD_NCCL_CORE`    | `1`     | `0` skips building NCCL core from the submodule.     |
| `NVTE_CUDA_ARCHS`         | many    | Semicolon-separated SM list. Use `"90;100"` minimum. |
| `NVTE_NCCL_EP_REBUILD`    | `0`     | `1` forces NCCL EP rebuild even if lib exists.       |
| `NVTE_BUILD_THREADS_PER_JOB` | auto | Cap parallel compile jobs (e.g. `16`).               |

Notes:
- B300 is `sm_103` — include `100` (same major family) in
  `NVTE_CUDA_ARCHS`. Single-arch builds (e.g. `"90"`) on B300 fail
  with `CUDA error 'named symbol not found'` at runtime.
- After editing NCCL EP source under
  `3rdparty/nccl/contrib/nccl_ep`, set `NVTE_NCCL_EP_REBUILD=1`
  for one rebuild — otherwise setup.py reuses the cached
  `libnccl_ep.so`.
- Do NOT hand-run `make -C 3rdparty/nccl/contrib/nccl_ep` —
  `NVCC_GENCODE` and `_NCCL_EP_LSA_TEAM_SIZE_*` must stay in
  lock-step with what TE links against. Use the setup.py path.

## Run the EP tests

The launcher spawns one process per local GPU and pins each to
exactly one device via JAX `local_device_ids`. Mesh shape is
chosen automatically by the test:

- 4 GPUs → `(dp=1, ep=4)`
- 8 GPUs → `(dp=2, ep=4)` (1×8 not supported yet — see SPRINT9)

### Sharded EP test (recommended)

```bash
cd tests/jax
SCRIPT_NAMES=test_multi_process_ep_sharded.py \
TEST_TIMEOUT_S=180 \
bash multi_process_launch_ep.sh
```

Covers `ep_bootstrap` rejection paths plus the
dispatch+combine identity round-trip under SPMD.

### Non-sharded EP test (4 GPUs only)

```bash
SCRIPT_NAMES=test_multi_process_ep.py \
TEST_TIMEOUT_S=180 \
bash multi_process_launch_ep.sh
```

Uses `ep_size = num_procs`. On 8 GPUs this trips a NCCL EP LSA
team-size limit — run only on 4-GPU boxes.

### Reading test output

The launcher writes:

- `stdout_multi_process.txt` — rank 0 (also tee'd to console).
- `stdout_rank_<i>.txt` — non-rank-0 logs, captured for crashes.

A successful run prints `Ran N tests in Xs / OK`. Non-rank-0
crashes are dumped only when rank 0 doesn't print a summary
(hang/early-crash heuristic).

## Common failure modes

| Symptom | Likely cause |
| --- | --- |
| `NCCL EP requires SM_90+ ... compute capability X.x` | Runtime device pre-Hopper; not a build issue. |
| `Unsupported LSA team size … in [_NCCL_EP_LSA_TEAM_SIZE_MIN, _MAX]` | `_NCCL_EP_LSA_TEAM_SIZE_MAX` < runtime team size; set via setup.py. |
| `CUDA error 'named symbol not found'` | Build SASS doesn't cover this GPU; add the SM to `NVTE_CUDA_ARCHS`. |
| `out = tokens / 2` on 2×4 | Pre-SPRINT8 bug — fixed; rebuild against current `main`. |
| Hang on bootstrap | Two EP groups colocated on one node; needs the SPRINT8 per-DP-color `ncclCommInitRank` path (already merged). |
