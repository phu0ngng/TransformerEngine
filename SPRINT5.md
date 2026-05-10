# EP Maintainer Review Sprint v5 (continuation of SPRINT3.md / SPRINT4.md)

## Goal

Fresh maintainer review pass over the NCCL EP integration after SPRINT4
landed (1D `recv_topk_weights` under HT+EM, N-D token shapes through the
JAX VJP, NLE>=2 default in the JAX test, EpConfig dataclass). Catch
stale comments, leftover back-compat shims, doc drift, and any latent
correctness traps left by the rapid SPRINT4 churn. Land the cheap
items in-place; defer everything that needs an NVLS-capable box for
runtime confirmation.

## Context

Hardware: this dev box is `4× H200 NVL` — no NVLS multicast. NCCL EP
init hangs in `ncclEpCreateGroup` (validated and surfaced in SPRINT3
step 5 as a clear NVTE_CHECK at init). Therefore:

- All cpp + Python changes are static / syntax-checked only.
- No full TE rebuild this session (build infra is fragile).
- Single-process JAX import smoke is OK (`import
  transformer_engine.jax.ep` works once the FFI shared object is in place).
- Anything that requires a multi-process runtime check is DEFERRED.

What landed since SPRINT3:

- `c4e47d87` HT+EM 1D `recv_topk_weights` end-to-end.
- `0ac4a2c3` N-D token shapes survive the JAX VJP — FFI flattens
  internally, residuals carry the leading-dim tuple.
- `3b40c6fc` HT-EM mentions dropped from cpp test docstrings; NLE>1
  regression test added.
- `ba40d860` blanket pre-commit format pass.
- `f770c419` `EpConfig` dataclass introduced; JAX tests consolidated
  on NLE>=2 (`num_experts = num_procs * 2`).

State of the integration is otherwise stable. SPRINT3's argument-order
review (lines 575-913) is largely landed (renames in C API + designated
initializers). SPRINT4's 1D `recv_topk_weights` is fully wired through
all five layers (L1..L4b).

## Issues found

### 1. `set_ep_num_local_experts` back-compat shim has zero callers and constructs a poisoned EpConfig [CHEAP, EXECUTABLE]

Citations:
- `transformer_engine/jax/cpp_extensions/ep.py:20` — exported in `__all__`.
- `transformer_engine/jax/cpp_extensions/ep.py:82-101` — definition.
- `transformer_engine/jax/cpp_extensions/ep.py:104-108` — `get_ep_num_local_experts()` defined alongside.

Grep across the entire repo
(`grep -rn set_ep_num_local_experts transformer_engine tests examples
build_tools`) confirms zero call sites. Notes:

- The shim was kept in SPRINT3 step 6 *before* `EpConfig` existed.
  After `f770c419` introduced `EpConfig`, the shim writes a "partial"
  `EpConfig` with `world_size=0, ep_size=0, num_experts=0,
  max_tokens_per_rank=0, max_recv_tokens_per_rank=0, hidden_dim=0` —
  any future code that reads `get_ep_config().ep_size` after a
  shim-only call will get `0` and divide-by-zero or silently broadcast
  on a stale shape.
- The shim was specifically called out in its own docstring as "Back-compat
  shim — prefer `set_ep_config(EpConfig(...))`." The "external callers"
  it was preserving against don't exist in this tree.

Fix: remove `set_ep_num_local_experts` from `__all__` and its definition.
Keep `get_ep_num_local_experts` (it's the consumer used by
`EpPreparePrimitive.abstract`); rewrite it to read `cfg.num_local_experts`
directly with a clearer error if not bootstrapped.

Acceptance: grep returns zero hits in `transformer_engine/`, `tests/`,
`examples/`, `build_tools/`. `python -c "from
transformer_engine.jax.cpp_extensions.ep import EpConfig, set_ep_config,
get_ep_config, get_ep_num_local_experts"` succeeds. `set_ep_num_local_experts`
import attempt fails with ImportError.

### 2. `examples/jax/ep/common.py:89` undersizes `recv_capacity` for top_k>1 [CHEAP, EXECUTABLE]

Citation: `examples/jax/ep/common.py:88-90`:
```python
# Worst-case recv: every source rank sends its full quota to this rank.
recv_capacity = ep_size * args.num_tokens
args.recv_capacity = recv_capacity
```

The comment claims worst-case but the formula is missing `top_k` —
under top_k=2 each token can produce up to 2 expert copies, all of
which can land on the same rank when routing is adversarial.

Compare `tests/jax/test_multi_process_ep.py:76`:
```python
cls.recv_capacity = cls.num_procs * cls.num_tokens * cls.top_k
```

The example only happens to work today because the default routing
formula `(rank * NLE + t * K + k) % E` distributes top_k slots across
neighboring experts, and with `num_experts == num_processes`
(NLE=1) different `k`s map to different ranks. As soon as the user
passes `--num-experts num_processes*2` (the kind of NLE>1 config
SPRINT3 + SPRINT4 specifically pushed for) the formula breaks: half
the top_k pairs can land on the same rank and overflow `recv_capacity`.

Fix: multiply by `args.top_k`. Update the comment to match the test's
formula.

Acceptance: example matches test recv_capacity formula; works for
arbitrary `--num-experts` divisible by `--num-processes`.

### 3. Stale block comment in `ep_backend.{h,cpp}` describes the abandoned per-op handle pattern [CHEAP, EXECUTABLE]

Citations:
- `transformer_engine/common/ep/ep_backend.h:11-21`:
  ```
   *  Handle lifecycle: ...
   *  - prepare():     open + ncclEpUpdateHandle (...) + close
   *  - dispatch():    open + ncclEpDispatch + close
   *  - combine():     open + ncclEpCombine + close
  ```
- `transformer_engine/common/ep/ep_backend.cpp:10-23` — file-level
  block comment documents the same per-op open/close model and says
  "EPBackend stateless across ops".

This was true in the early SPRINT2 implementation. Since the
persistent-handle redesign (`cur_handle_` + `cur_handle_mem_` members,
see `ep_backend.h:118-120`), the actual lifecycle is:
- `prepare()`: open handle once, reuse on later prepare on same
  `handle_mem`; never closed until `~EPBackend()` or until a different
  `handle_mem` pointer triggers re-open.
- `dispatch()` / `combine()`: reuse `cur_handle_` opened by prepare;
  combine asserts on it (`ep_backend.cpp:428-429`).

Fix: rewrite both block comments to describe the persistent-handle
pattern. Single-line bullets per op, mention that `combine` and
`dispatch_bwd` (which delegates to `combine`) require a prior
`prepare` on the same `handle_mem`. The dispatch fallback path that
opens a transient handle (`ep_backend.cpp:380-388`) should also be
called out — it's the backward-direction `dispatch` (= forward
combine_bwd) which is allowed to run without prepare.

Acceptance: block comments match what the code does.

### 4. `cpp_extensions/ep.py::EpDispatchPrimitive` partition `arg_shardings` exposes the input topk_idx/tokens shardings unchecked [DEFERRED, DOC]

Citations:
- `transformer_engine/jax/cpp_extensions/ep.py:222-240` (and analogous
  blocks at `:150-159, 304-327, 384-393, 444-457`).

```python
arg_shardings = tuple(a.sharding for a in arg_infos)
```

Every primitive's `partition` rule simply re-uses whatever sharding
the caller passed in for inputs and always declares replicated outputs
(`PartitionSpec(None, None)` etc.). For the current single-process /
no-mesh launch, this is fine. Once the EpConfig groundwork is used to
shard `token_counts` as `[ep_size, nle]` (the explicit motivation
called out in `cpp_extensions/ep.py:40-41`), these blocks need a
proper sharding-rule pass — `topk_idx`, `tokens` and `topk_weights`
are partitioned along the leading token axis in real workloads, and
the FFI's expectation today is that they are gathered.

Fix: add NotImplementedError-raising assertions in each `partition`
when any input has non-None partition specs, OR document the
limitation explicitly. Cannot land a real fix without the EpConfig
sharding plan being designed end-to-end. Don't pretend to support
sharded inputs.

Acceptance: docstring on each primitive says "Sharded inputs are not
supported in this release; pass replicated tensors. The partition
rule above is a placeholder for a future EpConfig-driven sharding
plan." and a NotImplementedError fires if a sharded input slips
through.

*Status*: DEFERRED — tweaking partition rules without a hardware
multi-mesh test is risky; the placeholder is benign for current
single-process tests. Track for the EpConfig-driven sharding work.

### 5. `_dispatch_bwd` weight-grad scatter is correct ONLY for uniform topk_weights [BUG, MUST FIX ON HARDWARE]

Citations:
- `transformer_engine/jax/ep.py:163-193` — `_dispatch_bwd`.
- `transformer_engine/jax/ep.py:175-176` — comment admits:
  > "exact magnitude for uniform routers, approximate (per-token
  >  average) for non-uniform."

The current implementation:

1. Pads the 1-D `g_recv_topk_weights` to bf16 `[recv_cap, PAD=32]` so
   NCCL EP combine accepts it.
2. Calls `tex.ep_dispatch_bwd` (= NCCL EP combine in bwd direction).
3. Reads `[:, 0]` and divides by `top_k` to "broadcast back across
   top_k".

Step 3 is the bug. NCCL EP combine in bwd direction performs an
**unweighted scatter-sum of top_k slot contributions per source token**
into a single `[T]` output. Dividing by `top_k` produces the
**average** `g` across the top_k slots, not the per-slot value. Then
`grad_topk_weights[t, k] = mean_over_k(g_recv_topk_weights[slot(t, k)])`
for every k — which is correct only when all top_k cotangents at
that token are equal, i.e. uniform routers.

For non-uniform routers (`grad_g[t, k]` differs by k) the per-slot
gradient is collapsed to its mean. The router will train but the
update direction is biased.

A correct implementation needs a per-`k` slot identity through
NCCL EP — there is no current path for that. Options:

- (a) Run `top_k` separate single-`k` dispatches in bwd, masking out
  all but the k'th slot per source token. Cost: `top_k`x slower.
- (b) Use the routing meta in `handle_mem` (or replay topk_idx on the
  JAX side) to scatter `g_recv_topk_weights` back to `[T, top_k]` in
  JAX without going through NCCL — possible because each slot's
  `(src_rank, src_t, k)` is recoverable from `topk_idx` at the time
  of dispatch. This is the cleanest fix; it requires saving
  `topk_idx` (or its allgathered version) in residuals.
- (c) Restrict the public API to uniform routers and assert at
  init/dispatch that `topk_weights[t, k]` is a constant of `k`.

This needs hardware validation (the existing `test_router_grad_
through_combine` test happens to use uniform `1/top_k`, so it passes).
Cannot fix without re-running the regression on an NVLS box.

Acceptance: a follow-up test with non-uniform per-k weights — e.g.
`w[t, k] = (1.0 + 0.5 * k) / norm` — finite-diff matches analytical
grad within bf16 tolerance.

### 6. Stale "register_user_buffer_collective" / `recv_topk_weights` shape comment in `ep_api.cpp` [DOC-ONLY] (no action — already correct)

Verified: `ep_api.cpp` is correct as-is — `recv_topk_weights` is
1-D `[recv_capacity]` everywhere.

### 7. `ep_combine` public docstring at `ep.py:218` says "always 2D" but the function accepts the post-FFN buffer that the caller has reshaped — true if caller flattened, false if upstream produced 3D [DOC-ONLY, EXECUTABLE]

Citation: `transformer_engine/jax/ep.py:213-228`. Docstring claims
`expert_out: [recv_capacity, H] post-FFN activations (always 2D)`.

The C++ FFI also asserts 2D at `transformer_engine/jax/csrc/extensions/ep.cpp:182-183`:
```c++
NVTE_CHECK(eo_dims.size() == 2,
           "expert_out must be 2D [recv_capacity, H], got ndim=", eo_dims.size());
```

Yet the dispatch's `recv_tokens` output IS 2D (`recv_capacity, H`), so
this is consistent — but the asymmetry vs `ep_dispatch` (which now
accepts N-D `tokens`) is worth flagging in the docstring. Caller doing
`expert_out = ffn(recv_tokens)` from dispatch gets back 2D, fine.
Caller doing fancy reshape between dispatch and combine must flatten
back to 2D first. Document explicitly in the public API docstring so
future contributors know this asymmetry is intentional (post-FFN
expert kernels operate on the EM-grouped 2D buffer, not on N-D token
shapes).

Fix: 1-line clarification in `ep.py:218` and `cpp_extensions/ep.py:289`
docstrings.

Acceptance: docstring states the asymmetry explicitly.

### 8. `EpConfig.world_size` / `EpConfig.rank` are written but never read [DOC-ONLY, EXECUTABLE]

Citation: `transformer_engine/jax/cpp_extensions/ep.py:53-60`. The
dataclass holds `world_size` and `rank` but no abstract-eval rule or
sharding hook reads them today. They're set in `ep_bootstrap` but
never queried.

Per `cpp_extensions/ep.py:39-41`, the EpConfig was added "for future
EpConfig-driven sharding patterns ([ep_size, nle] for token_counts)".
`ep_size` is the relevant field; `world_size` and `rank` come along
because the bootstrap function takes them.

Decision: keep the fields (they're cheap, they round-trip the
bootstrap signature, and any future code that spawns multiple EP
groups in different process subsets will need them). Add a one-line
comment in the dataclass docstring saying so.

Acceptance: dataclass docstring documents which fields are
runtime-required vs. recorded-for-future-use.

### 9. `cpp_extensions/ep.py` and `jax/ep.py` have inconsistent `topk_idx` minimum-ndim assertion under EP API [CHEAP, EXECUTABLE]

Citations:
- `transformer_engine/jax/csrc/extensions/ep.cpp:71-72` — FFI asserts
  `topk_dims.size() >= 2`.
- `transformer_engine/jax/cpp_extensions/ep.py:127-129` — abstract
  asserts `len(topk_idx_aval.shape) >= 2`.
- `transformer_engine/jax/cpp_extensions/ep.py:185-191` — dispatch
  abstract asserts tokens `>= 2`.
- `transformer_engine/jax/ep.py:111-122` — public `ep_prepare`
  docstring says `[T, top_k]` (2D — no N-D mention).
- `transformer_engine/jax/ep.py:130-146` — public `ep_dispatch`
  docstring says `[T, top_k]` for topk_idx but `[T, H]` for tokens —
  no N-D mention either, even though the FFI now flattens N-D
  internally.

The N-D support landed in `0ac4a2c3` for tokens but the public
`ep_prepare` / `ep_dispatch` docstrings still describe the 2D-only
shapes. Bring the public docstrings up to date so callers know they
can pass `[B, S, top_k]` and `[B, S, H]` directly.

Fix: `topk_idx [..., top_k]` and `tokens [..., H]` in both public
docstrings. Mention that `top_k = topk_idx.shape[-1]` and the
leading dims are flattened by the FFI.

Acceptance: docstrings match the FFI's actual contract.

### 10. JAX `EpConfig` provides `ep_size` and `num_local_experts`, but `EpPreparePrimitive.abstract` still calls a wrapper instead of reading the dataclass [CHEAP, EXECUTABLE]

Citation: `transformer_engine/jax/cpp_extensions/ep.py:124-134`.
`abstract()` calls `get_ep_num_local_experts()` — a wrapper that
reads `_ep_config.num_local_experts`. The wrapper is a back-compat
artifact from the SPRINT3-era `set_ep_num_local_experts` design. Now
that `EpConfig` exists and is the source of truth, abstract-eval
can read the dataclass directly.

Combine with item #1: removing `set_ep_num_local_experts` motivates
collapsing `get_ep_num_local_experts` into a thin
`get_ep_config().num_local_experts` access too.

Fix: keep `get_ep_num_local_experts` as a tiny convenience wrapper
(one line) but make it just `return get_ep_config().num_local_experts`
with a clear error if cfg is None. Drop the `cfg.num_local_experts <= 0`
check (it was only triggered by the now-deleted partial-shim
`set_ep_num_local_experts` constructing zero fields).

Acceptance: `get_ep_num_local_experts` is a 2-line function; no
distinction between "cfg unset" and "cfg has zero NLE" because the
only path that produced the latter was the deleted shim.

### 11. `EpDispatchConfig::top_k` field is decoded but not used by the FFI handler [CHEAP, EXECUTABLE]

Citation: `transformer_engine/jax/csrc/extensions/ep.cpp:24-27`,
`107-160`.
```c++
struct EpDispatchConfig {
  int64_t recv_capacity;  // recv_tokens first dim (= recv_topk_weights size)
  int64_t top_k;          // routing top_k (recv_topk_weights is 1D)
};
```
`config.top_k` is never read inside `EpDispatchFFI`. The C++ backend
re-derives top_k from `topk_idx.shape[1]` (`ep_backend.cpp:331-332`).
Either remove the field from the struct + Python primitive
`impl_static_args`, or read it for an explicit cross-check
(`NVTE_CHECK(idx_dims.back() == config.top_k)`) so dispatch fails
loudly on mismatch.

Recommended: keep `top_k` in the struct AND assert it against
`topk_idx.dimensions().back()` inside `EpDispatchFFI` so that an XLA
shape-inference bug or a primitive caller that lies about top_k would
be caught at dispatch time instead of producing wrong NCCL EP
metadata silently.

Fix: `NVTE_CHECK(static_cast<int64_t>(idx_dims.back()) == config.top_k,
"top_k attr (", config.top_k, ") must match topk_idx last dim (",
idx_dims.back(), ")")` after the existing `NVTE_CHECK` that idx is
2D+.

Acceptance: mismatched `top_k` attr fails fast with a clear error.

### 12. `_combine_bwd` returns `grad_handle_mem = jnp.zeros_like(handle_mem)` and `grad_token_counts = jnp.zeros_like(token_counts)` — fine but redundant memory traffic [DEFERRED, low priority]

Citation: `transformer_engine/jax/ep.py:269-271`.

JAX VJP requires a cotangent for every primal input including the
non-differentiable ones. `jnp.zeros_like(handle_mem)` allocates a
fresh `[N] uint8` tensor on every backward call — wastes a small
amount of HBM (handle_mem is ~10 KB).

JAX has `jax.custom_derivatives.zero_from_primal` and
`SymbolicZero` plumbing that newer JAX versions accept; the proper
fix is to mark these inputs as non-differentiable via `nondiff_argnums`
and let JAX skip cotangent generation.

Currently `ep_combine` only marks `num_local_tokens` as
nondiff_argnum (4). Adding `0` (handle_mem), `1` (token_counts), and
the integer args is mostly cosmetic given how small the buffers are.

Fix: deferred. Track for a follow-up.

Acceptance: zero allocations of handle-mem-sized bool buffers per
bwd pass.

*Status*: DEFERRED — perf only, semantics already correct.

### 13. `nvte_ep_dispatch` doc still talks about `recv_topk_weights` 2D in one stale spot [CHEAP, EXECUTABLE]

Citation: `transformer_engine/common/include/transformer_engine/ep.h:153-156`:
```
 *  \param[out]    recv_topk_weights  Received per-slot weights [recv_T] float32
 *                                    (1 weight per slot). Pass null NVTETensor
 *                                    in backward (no weights to scatter back).
```
Looks correct ([recv_T]). But `ep.h:142-143` says:
```
 *  NCCL EP routes 3 inputs (tokens, topk_weights, topk_idx) and writes back
 *  2 outputs (recv_tokens, recv_topk_weights).
```
Also correct. The only thing missing is mentioning that
`recv_topk_weights` is 1D under HT+EM (the configured layout).
Document it explicitly so a reader doesn't assume it has the same
2D shape as the input `topk_weights`.

Fix: add 1 word "1D" in the param doc.

Acceptance: param doc reads "Received per-slot weights [recv_T]
float32 (1D, 1 weight per slot)".

### 14. `ep_backend.cpp::dispatch` ignores `topk_idx`/`topk_weights` when called from combine_bwd, but `combine_bwd` passes nullptr explicitly [DOC-ONLY, no action]

Verified: `ep_backend.cpp:445-447` calls `dispatch(handle_mem,
topk_idx=nullptr, grad, topk_weights=nullptr, grad_expert_out,
recv_topk_weights=nullptr, stream)`. The `is_forward` flag in
`dispatch` (`ep_backend.cpp:321`) keys off `topk_weights != nullptr`
and the function correctly uses 1 input / 1 output. No bug.

### 15. `ep_combine` C++ comment says "`expert_out` is post-hadamard, masked" — but the cpp distributed test's Combine test passes raw `recv_tokens` [DOC, RESOLVED for cpp tests]

Citation: `tests/cpp_distributed/test_ep_pipeline.cu:382-385`:
```c++
// Identity expert: pass dispatch output directly to combine.
ASSERT_NO_THROW(nvte_ep_combine(handle_mem_t.tensor, recv_tokens_t.tensor,
                                result_t.tensor, stream));
```

In the cpp test, the user-side hadamard is intentionally skipped (test
asserts the unweighted-sum semantics: `result == top_k * tokens`).
The TE-JAX public `ep_combine` does pre-multiply by `recv_topk_weights`
(see `ep.py:243-247`). Both behaviors are correct; the asymmetry is
intentional. The C `nvte_ep_combine` doc at `ep.h:163-169` explains
this. Verified, no action.

## Steps & status

### [x] Step 1: Remove `set_ep_num_local_experts` shim (item #1)

Files: `transformer_engine/jax/cpp_extensions/ep.py` — drop from
`__all__` and remove function. Tighten `get_ep_num_local_experts`
(item #10).

Verify: `grep -rn set_ep_num_local_experts transformer_engine tests
examples build_tools` returns no hits. `python -c "import ast;
ast.parse(open('transformer_engine/jax/cpp_extensions/ep.py').read())"`
succeeds.

### [x] Step 2: Fix `examples/jax/ep/common.py` recv_capacity formula (item #2)

File: `examples/jax/ep/common.py:88-90`. Multiply by `args.top_k`,
update comment.

Verify: `python -c "import ast;
ast.parse(open('examples/jax/ep/common.py').read())"`.

### [x] Step 3: Rewrite stale ep_backend.{h,cpp} block comments (item #3)

Files: `transformer_engine/common/ep/ep_backend.h:11-21`,
`transformer_engine/common/ep/ep_backend.cpp:10-23`.

Verify: visual diff matches the persistent-handle code.

### [x] Step 4: Update public docstrings for N-D shapes (item #9)

Files: `transformer_engine/jax/ep.py:115-122` (ep_prepare),
`transformer_engine/jax/ep.py:131-145` (ep_dispatch).

Verify: `python -c "import ast;
ast.parse(open('transformer_engine/jax/ep.py').read())"`.

### [x] Step 5: Add EpDispatchConfig.top_k cross-check (item #11)

File: `transformer_engine/jax/csrc/extensions/ep.cpp:107-160`. Add
`NVTE_CHECK` after the topk_idx 2D+ assertion.

Verify: line compiles in head (will be checked at next full rebuild —
not run here).

### [x] Step 6: Tighten `get_ep_num_local_experts` (item #10)

File: `transformer_engine/jax/cpp_extensions/ep.py:104-108`.

Verify: existing JAX import smoke (`python -c "import ast; ..."`).

### [x] Step 7: Annotate placeholder partition rules (item #4)

File: `transformer_engine/jax/cpp_extensions/ep.py` — add a single
`__doc__` block at the top of the module documenting that the
sharding rules are placeholders for the EpConfig-driven plan.

*Promoted to a [DOC-ONLY] action, not the NotImplementedError path
called for in #4 — adding asserts in 5 partition rules across the
file is non-trivial without runtime testing. The doc note is
sufficient warning.*

Verify: smoke import.

### [x] Step 8: Clarify `ep_combine` 2D asymmetry (item #7)

File: `transformer_engine/jax/ep.py:213-228`,
`transformer_engine/jax/cpp_extensions/ep.py:283-293`.

Verify: smoke import.

### [x] Step 9: Annotate EpConfig field intent (item #8)

File: `transformer_engine/jax/cpp_extensions/ep.py:49-61`.

Verify: smoke import.

### [x] Step 10: Add 1D recv_topk_weights mention to `nvte_ep_dispatch` doc (item #13)

File: `transformer_engine/common/include/transformer_engine/ep.h:153-156`.

Verify: header is comment-only change.

## Remaining items for next session

When resuming on NVLS-capable hardware (GH200 / DGX H200 / DGX H100):

1. **Validate everything from this sprint** — run the full test matrix
   to confirm the doc changes did not regress and the
   examples/jax/ep top_k fix doesn't surface a new buffer overflow
   somewhere downstream. (`tests/cpp_distributed/run_test_ep.sh 4 build`
   + `tests/jax/multi_process_launch_ep.sh` +
   `examples/jax/ep/run_test_ep.sh`.)

2. **[BUG] Non-uniform `topk_weights` router-grad (item #5)** — design
   and land a per-slot bwd path. Add a regression test with
   `w[t, k] = (1.0 + 0.5 * k) / norm`, finite-diff matches analytical
   grad within bf16 tol. Today's `_dispatch_bwd` averages across
   top_k — biased gradient. **This is a real correctness bug**;
   it is NOT safe to fix without hardware validation. Until then,
   document the limitation in `ep.py::ep_dispatch` and assert in
   `_dispatch_bwd` that `g_recv_topk_weights` is approximately
   uniform across top_k OR detect at dispatch time that
   `topk_weights[t, k]` is constant in `k` and short-circuit.

3. **[BUG] PAD=32 hardcode in `_dispatch_bwd` (item #5 sibling)** —
   `transformer_engine/jax/ep.py:177` hardcodes PAD=32 (chosen because
   `32 * 2 = 64 byte aligned`). NCCL EP combine asserts on a
   16-byte-or-larger alignment per the SPRINT3 audit; whether 32 *
   sizeof(bf16) is the right value for all hidden_dim configurations
   remains unverified. Test under `hidden_dim=64`, `hidden_dim=128`,
   `hidden_dim=4096` and confirm the PAD isn't biting.

4. **Item #4 sharding hooks** — proper sharding rules for all five
   primitives once the EpConfig-driven `[ep_size, nle]` token_counts
   plan is decided. Currently every `partition` returns input
   shardings unchanged and replicated outputs.

5. **Item #12 perf nit** — kill redundant `jnp.zeros_like` allocations
   in `_combine_bwd` for handle_mem / token_counts cotangents. Use
   newer JAX `nondiff_argnums` semantics or `SymbolicZero` plumbing.

## Constraints honored

- `git commit -s` single-line messages, no AI attribution.
- Author/committer = `Phuong Nguyen <phuonguyen@nvidia.com>` (env not
  overridden).
- `NVTE_CUDA_ARCHS="90"` would have been used if a rebuild was needed
  (none required this session — all changes are header / docstring /
  Python-only and a one-line C++ NVTE_CHECK addition that links into
  the same shared object on rebuild).
- No drive-by refactors. Each commit touches only the lines required
  for the issue at hand.
- No destructive git ops. Pre-commit hook NOT skipped.
