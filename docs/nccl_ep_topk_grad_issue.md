# NCCL EP — Per-k Gradient Identity in Combine Backward

## Summary

TL;DR: **NCCL EP HT already supports exact per-(t,k) `grad_topk_weights`** —
the host API and kernel both ship the dense prob payload identical to
hybrid-EP's `BACKWARD_COMBINE`. Our TE-JAX backend just isn't using it.
Action: extend `EPBackend::combine` to plumb `topk_weights` in and
`combined_topk_weights` out; no NCCL changes required.

The current TE-JAX `_combine_bwd` averages per-`k` cotangents at the same
source token, biasing the magnitude of router gradients. Fix is on our
side.

## The math we need

Forward combine:
```
out[t] = Σ_k w[t,k] · y_k        # y_k is expert e_k's output for token t
```

Analytical bwd:
```
grad_y_k     = w[t,k] · dout[t]          # vector, shape [H]
grad_w[t,k]  = dout[t] · y_k             # scalar, DIFFERENT for each k
```

The per-`k` gradient for `w[t,k]` depends on the *specific* `y_k`, so two
slots `(t, k=0)` and `(t, k=1)` produce different `grad_w` values.

## What NCCL EP HT actually exposes

NCCL EP HT combine is the same kernel as hybrid-EP (forked + NCCL GIN + opts;
see `ep/docs/nccl_ep_ht_combine_warp_specialization.md` line 476: "Per-token
combine WQEs: 1 (token) + 1 (prob, BWD) — Same: 1 + 1 (BWD)"). The dense
prob payload IS in the kernel, and the host wrapper plumbs it end-to-end.

`nccl_ep.cc:2490-2639`:

```cpp
// Trigger backward mode by passing topk_weights as an input tensor.
ncclNDTensor_t topk_weights         = find_tensor_by_tag(inputs,  NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
ncclNDTensor_t combined_topk_weights = find_tensor_by_tag(outputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
bool backward_combine = (topk_weights != nullptr);

if (backward_combine) {
    // sparse [T, topk] → dense [T, E_local_per_node] using local_expert_routing_map
    nccl_ep::hybridep::sparse_to_dense_prob_combine(...);
}

// Kernel runs with backward_combine=true → BACKWARD_COMBINE template true →
// per-token WQE carries token (bf16) + prob (fp32, E_local_per_node).

if (backward_combine && combined_topk_weights != nullptr) {
    // dense [T, num_experts] → sparse [T, topk] using global_routing_map
    nccl_ep::hybridep::dense_to_sparse_prob_combine(...);
}
```

So the public API already accepts sparse `topk_weights[T, topk]` in and
returns sparse `combined_topk_weights[T, topk]` out — exactly the
per-(t,k)-preserved channel we want. Dense conversion is internal.

## What TE-JAX is missing

`EPBackend::combine` (`transformer_engine/common/ep/ep_backend.cpp:488-519`)
only passes one input tensor (TOKENS) and one output tensor (TOKENS):

```cpp
const ncclNDTensor_t inputs[]  = { nccl_expert_in  };  // TOKENS only
const ncclNDTensor_t outputs[] = { nccl_result_out };  // TOKENS only
ncclEpCombine(entry.handle, inputs, 1, outputs, 1, nullptr, 0, 0, nullptr, stream);
```

No `topk_weights` tensor is forwarded → `backward_combine = false` → the
kernel runs in plain-fwd-add mode, no prob payload.

`combine_bwd` (`ep_backend.cpp:528`) just calls `combine()` with the gradient,
so it inherits the same limitation. The dispatch_bwd path goes through
`dispatch()` and isn't relevant to `grad_topk_weights`.

## Fix plan (no NCCL changes)

1. **C++ backend** — extend `EPBackend::combine` to accept optional
   `topk_weights_in` (sparse `[T, topk]` fp32) and `topk_weights_out`
   (sparse `[T, topk]` fp32) tensors. When present, add them to the inputs/
   outputs arrays of `ncclEpCombine` with the `NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS`
   tag. `nccl_ep.cc` already detects the presence and enters BACKWARD mode.

2. **`combine_bwd` semantics** — in the TE-JAX `_combine_bwd` path, after
   running `combine_bwd(grad_out, handle)` to get `grad_recv_tokens` at the
   expert side, locally compute
   `dot_per_slot[slot] = Σ_h grad_recv_tokens[slot,h] · y_slot[slot,h]`
   where `y_slot` is the residualized fwd `recv_tokens_out`. Then feed
   `dot_per_slot` (reshaped to expert-side sparse layout) as the
   `topk_weights` input to a second `combine` call and read
   `grad_topk_weights[t, k]` from the `combined_topk_weights` output.

3. **JAX primitive plumbing** — `EpCombinePrimitive` (and its bwd) gain an
   extra in/out pair shaped `[T_local, top_k]` fp32. `_combine_fwd`
   residualizes `recv_tokens_out` for use by `_combine_bwd`.

4. **Drop** the docstring caveat in `transformer_engine/jax/ep.py` and the
   per-token-average fallback once tests confirm exact-grad behavior.

## What we tried in TE-JAX (all blocked by the missing primitive)

### Option A — compute on expert side
1. Residualize `y_slot` (= `recv_tokens_out`) from `_combine_fwd`.
2. In bwd: `combine_bwd(grad_out, handle)` → `grad_out_slot[slot, H]` on expert.
3. Local einsum: `dot_per_slot[slot] = Σ_h grad_out_slot[slot,h] · y_slot[slot,h]`.
4. **Ship `dot_per_slot` back to source as `[T, top_k]` preserving `k`.**

Step 4 has no NCCL EP primitive:
- `combine` would sum across `k` (losing identity).
- `dispatch` is the wrong direction.

### Option B — compute on source side
1. Get expert outputs at source un-reduced, shape `[T, top_k, H]`.
2. Local einsum: `grad_w[t,k] = Σ_h dout[t,h] · y[t,k,h]`.

Step 1 has no un-reduced gather. Workaround = run `combine` `top_k` times
with k-masked weights → `top_k×` bandwidth. Unacceptable.

### Option C — reuse topk_weights channel
`dispatch` already carries `topk_weights[T, top_k]` per-slot without summing
— that's the per-`k`-preserved channel we want. But it's a fwd-direction-only
payload. Reusing it would mean a second forward dispatch in the bwd step
carrying `y_k` as the "token" — same `top_k×` bandwidth as Option B.

## Background — how the kernel suites compare

NCCL EP HT, hybrid-EP, and DeepEP-v2 are three different kernel
implementations of combine. NCCL EP HT and hybrid-EP share the **same**
combine kernel structure (NCCL EP HT is forked from hybrid-EP, plus NCCL GIN
and other opts). DeepEP-v2 is a separate, newer kernel.

| Combine kernel        | Source                                  | Per-(t,k) weight channel             | BW overhead |
|-----------------------|------------------------------------------|--------------------------------------|-------------|
| **NCCL EP HT**        | `nccl/contrib/nccl_ep/device/hybrid_ep.cuh` | dense `[E_local × ranks_per_node]` fp32 / token (BACKWARD_COMBINE) | ~7% on DS v3 |
| **Hybrid-EP**         | `DeepEP/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` | identical to NCCL EP HT | ~7% |
| **DeepEP-v2**         | `DeepEP/csrc/kernels/elastic/combine.hpp` + `deep_ep/include/deep_ep/impls/combine.cuh` | sparse `[top_k]` fp32 / slot (in `TokenLayout`) | ~0.06% |

NCCL EP HT chose hybrid-EP's dense-prob design (vs DeepEP-v2's sparse-slot
design). On the wire we pay ~7% extra in BWD combine, but the host wrapper
hides this — caller passes/receives sparse `[T, top_k]` and the
`sparse_to_dense_prob_combine` / `dense_to_sparse_prob_combine` helpers in
`nccl_ep.cc` translate at the boundary.

## How prior art handles it (kernel-level detail)

The DeepEP repo (`github.com/deepseek-ai/DeepEP`) ships two **different**
combine kernel implementations. They take different approaches to the
per-(t,k) weight channel.

### DeepEP-v2 — sparse per-(t,k) `topk_weights` in TokenLayout slot

Reference: `csrc/kernels/elastic/combine.hpp` (runtime launcher) and
`deep_ep/include/deep_ep/impls/combine.cuh` (kernel body).

V2 is structured as two PDL-chained kernels: `combine_impl` (gather/reduce)
then `CombineReduceEpilogue` (FP32 final accumulate + bias/scale). Each
wire-format `TokenLayout` slot embeds the hidden payload, src metadata, AND
a small per-(t,k) `topk_weights[top_k]` fp32 sub-region. The slot is written
into the receiver's `[GIN]` window via TMA + mbarrier.

`combine_impl` (`combine.cuh:216-219`):
```cpp
// Write topk weights
if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {
    const float value = __ldg(topk_weights + (i * kNumTopk + lane_idx));
    master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;
}
```

Each src `(t, k)` lane writes its own weight into the slot's `top_k` slice;
the receiver decodes per-(t,k) on the other side. **Same kernel handles fwd
and bwd** (per the v2 spec: combine is the backward of dispatch and supports
fwd replication via the expanded-send branch when `kAllowMultipleReduction`
is false). The `topk_weights` channel preserves per-k identity natively
because it's indexed by `lane_idx` in [0, top_k).

How this is used in backward (autograd composed at Python level):

1. Save `y_slot = recv_tokens_out` from fwd combine as residual.
2. In bwd:
   - Scatter `grad_out[t]` to (t,k) slots via the receiver-side combine path
     (with `topk_weights = 1` so it's a plain copy).
   - Locally on expert: `dot_per_slot[slot] = Σ_h grad_out_slot · y_slot`.
   - Run combine again with `topk_weights = dot_per_slot` — the slot's
     `topk_weights` sub-region carries the per-(t,k) cotangent back to the
     attention rank.

Cost: `top_k × 4` bytes per token. For DS v3 (top_k=8, H=7168, bf16) ~0.06%
overhead. Essentially free.

### Hybrid-EP — dense per-expert prob channel (BWD-only)

Reference: `csrc/hybrid_ep/backend/hybrid_ep_backend.cuh`,
`csrc/hybrid_ep/config.cuh`.

Hybrid-EP combine is **BWD-only** — `BACKWARD_COMBINE = true` is hard-wired
(`config.cuh:106`) — and it transfers a dense FP32 prob payload of size
`E_local_per_rank × ranks_per_node` per token, alongside the token, reduced
through both intra-node and inter-node stages with the same warp groups.

Concretely in the intra-node reduction warp group
(`hybrid_ep_backend.cuh:2515-2627`):
- Each src slot contributes both its `token` (bf16, HIDDEN_DIM) and its
  `prob` (fp32, `E_local_node`) into separate FP32 accumulators.
- Both are reduced over src slots that land on the same dst token.
- Both are stored to the dst buffer (`lines 2608-2648`).

Python interface (`deep_ep/hybrid_ep_buffer.py:262-287`):
```python
combined_token, combined_probs = self.runtime.combine(
    hidden=hidden, probs=probs, handle=handle_impl, with_probs=probs is not None
)
```

How this is used in backward:

1. Forward gates: dense prob `[T, E_total]` is scattered into top_k slots
   via `sparse_to_dense_map` (each (t,k) slot reads its expert's slice).
2. Backward gates: dense prob-grad `[T, E_total]` is scatter-summed back
   into a full per-token dense vector — each src expert-slot's prob-grad
   contribution lands in the corresponding column of the dst token's vector.
3. Host-side recovery — no extra wire cost:
   ```python
   grad_topk_weights[t, k] = grad_input_prob[t, topk_idx[t, k]]
   ```

Cost: `E_total × 4 bytes/token` extra bandwidth in bwd combine. For DS v3
(E=256, H=7168, bf16) ~7% overhead vs the unweighted bwd.

### Comparison

| Approach        | Side-channel shape           | Bwd-only? | BW overhead (DS v3) |
|-----------------|------------------------------|-----------|---------------------|
| DeepEP-v2       | sparse `[top_k]` fp32 / slot  | No (fwd+bwd same kernel) | ~0.06% |
| Hybrid-EP       | dense `[E_local × ranks_per_node]` fp32 / token | Yes (BWD only) | ~7%    |

(They are different kernels — `DeepEP/csrc/kernels/elastic/` vs `DeepEP/csrc/hybrid_ep/` —
see `ep/docs/deepep_v2_*.md` and `ep/docs/hybrid_ep_*.md` for full spec.)

## (Nothing to ask of NCCL team — fix lives in TE)

Earlier drafts of this doc proposed asking NCCL to expose a per-(t,k)
channel. That ask is moot: NCCL EP HT already exposes it via the
`NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS` in/out tags on `ncclEpCombine`. The work
is in TE: thread the topk_weights tensors through `EPBackend::combine` and
the JAX primitive wrapper, plus compute `dot_per_slot` in `_combine_bwd`
before the second combine call.

## References

- TE-JAX MoE bwd implementation: `transformer_engine/jax/ep.py` (`_combine_bwd`)
- TE-JAX docstring noting the limitation: `transformer_engine/jax/ep.py`
  (Backward limitation: router gradient)
- DeepEP-v2 combine kernel:
  `DeepEP/csrc/kernels/elastic/combine.hpp` (launcher) and
  `DeepEP/deep_ep/include/deep_ep/impls/combine.cuh` (kernel body, see
  `lines 216-219` for the per-(t,k) topk_weights slot write)
- Hybrid-EP combine kernel:
  `DeepEP/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` (BWD-only,
  see `lines 2515-2627` for the dense prob accumulation)
- Spec docs (cross-impl comparison):
  - `EP/ep/docs/deepep_v2_combine_warp_specialization.md`
  - `EP/ep/docs/hybrid_ep_combine_warp_specialization.md`
