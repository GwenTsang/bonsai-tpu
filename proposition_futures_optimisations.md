# Future optimization proposals

Review date: 2026-11-13. Latest inspected commit: `30a3016`, including the working-tree changes visible during review. Claude Code was actively changing the repository, so this is a snapshot rather than a review of an immutable release.

This review made no implementation changes and submitted no TPU work. The earlier local CPU test run passed all 20 tests. Two engine issues were also reproduced with host-only fixtures. Performance numbers below come from saved repository results; proposed improvements have **not** been benchmarked on the TPU.

## Priorities

| Priority | Proposal | Expected benefit | Evidence |
| --- | --- | --- | --- |
| P0 | Preserve the sampled token after a speculative rejection | Correct stochastic generation | Confirmed scheduler bias in a host-only reproduction |
| P0 | Protect the selected parked conversation during eviction | Prevent conversation-switch crashes | Confirmed host-only reproduction |
| P1 | Choose prefill chunks by estimated cost | Lower latency for short prompt suffixes | Existing 64/512 timings and current scheduling rule |
| P1 | Budget transient snapshot and parking allocations | More reliable long-context serving | Allocation order and recorded memory usage |
| P1 | Make the shared-session runner reject concurrent work before mutation | Reliable experiments and logs | Static inspection of `scripts/run.sh` |
| P2 | Reduce the sequential work in chunked GDN prefill | Faster prefill | Algebraic opportunity; needs measurement |
| P2 | Specialize attention and explore quantized KV | Long-context throughput and capacity | Context-length performance degradation |
| P3 | Avoid unnecessary greedy-sampling and streaming work | Small decode/host-latency improvements | Code inspection; needs profiling |

## 1. Fix speculative sampling before extending speculation

Location: `bonsai_tpu/engine.py`, `generate_ids()`, particularly the `acc == 0` branch and subsequent `_block_fn()` call.

When the first draft token is rejected, the verifier has already sampled a valid next token in `preds[0]`. The engine restores recurrent state but discards that sample and falls through to ordinary decoding, which samples again. This favors the draft token.

For a fixed draft token with target probability `p`, the resulting probability becomes `p + (1-p)*p`, rather than `p`. A host-only fixture exercising the actual scheduling method with a uniform two-token mock model and a constant draft produced 728 draft-token outcomes in 1,000 trials. The unbiased expectation is approximately 500; the faulty algorithm predicts 750. This is a scheduler test, not a TPU model-quality measurement.

**Proposal:** retain the verifier's mismatch sample, restore/replay precisely the accepted input prefix, and continue from that sample without another random draw. Cover zero acceptance, partial acceptance, stop tokens, generation limits, and state/history consistency. Add a statistical test with an analytically known distribution.

The batched verifier and single-token decode also use different numerical paths. Preserving the sampled token fixes this scheduling bias; it does not establish bit-identical logits between those paths. The saved greedy copy examples in `results/speculation.json` do not validate stochastic distribution preservation.

## 2. Make conversation parking eviction-safe and bounded

Location: `bonsai_tpu/engine.py`, `_select_conversation()`, `_park()`, `_unpark()`.

`_select_conversation()` chooses a parked session, then parks the active session. That operation can evict the selected restore target. `_unpark()` subsequently calls `self.parked.remove(sess)` and fails. A host-only fixture reproduced `ValueError: list.remove(x): x not in list` under a small parking budget.

**Proposal:** detach or explicitly protect the selected restore target before parking the current session, with clear ownership of its buffers. Estimate the incoming allocation and evict before copying. Define behavior when one conversation exceeds the entire budget: the present `len(self.parked) > 1` condition allows that oversized session to remain.

Parking currently materializes arrays through sequential `np.asarray()` calls and allocates the incoming session before eviction. Both transient memory and transfer scheduling deserve attention. Experiment with issuing asynchronous host copies ahead of materialization and with reusable staging buffers, measuring actual overlap rather than assuming it.

In `results/parking.json`, a roughly 19K-token conversation takes approximately 0.84–1.35 seconds to park and 0.33–0.57 seconds to restore. This is a meaningful agent-loop latency target. Validate round-trip logits and checkpoint restoration as well as transfer speed.

## 3. Use cost-aware prefill chunk scheduling

Location: `bonsai_tpu/engine.py`, `feed()` and `PREFILL_BUCKETS`.

Every remainder above 64 tokens currently selects the 512-token executable. A 65-token suffix therefore performs a padded 512-token pass. Message-boundary checkpoints can create many such short segments.

`results/prefill-speed.json` records approximately 239 ms for 512 positions and 47.4 ms for 64. From those measurements, two 64-token passes would cost approximately 94.8 ms for a 65–128-token suffix, versus 239 ms with the current rule: an estimated saving of 144 ms. Four small passes are also cheaper in this simple estimate. These are estimates from existing timings, not measured scheduler improvements; attention cost varies with context length.

**Proposal:** choose a sequence of existing buckets using a small measured cost table indexed by context range. Benchmark suffix lengths 1, 32, 64, 65, 128, 192, 256, 320, and 512 at short and long contexts. Prefer this before adding executable sizes, because each cached executable has substantial host-memory cost.

Also evaluate feeding a single-token suffix through the existing decode computation instead of padded prefill. Preserve pending-logit and recurrent-state semantics, and validate numerical differences. Skipping the vocabulary projection for intermediate prefill chunks is another opportunity, but likely smaller than eliminating padded work.

## 4. Plan memory around transient allocations

Location: `bonsai_tpu/engine.py`, `snapshot()`; `bonsai_tpu/model.py`, `init_state()`.

Snapshot creation copies recurrent state before evicting an old checkpoint. With six retained checkpoints, this can temporarily require a seventh copy. Each checkpoint is approximately 150 MiB according to the accounted recurrent-state size. Decide eviction before allocation, or reuse a checkpoint buffer while preserving asynchronous execution dependencies. Check the small-budget cases, including `max_snapshots=1`.

The saved 96K-context run records a peak of 16,904,805,376 bytes against a limit of 16,909,336,064 bytes: only 4.32 MiB below the limit. **This is a historical allocator peak, not proof that the current request has only that headroom.** The persistent session can retain peak statistics across workloads, and compact scales have since saved about 0.37 GiB. Re-measure in an isolated process before setting production limits.

At the current attention dimensions, BF16 KV storage costs 65,536 bytes per token. Adding 512 slack positions and rounding to a 4,096-position attention block creates 4,096 extra rows for an aligned context size: 256 MiB rather than the nominal 32 MiB of slack. Explore a smaller final allocation/block or masked partial-block access only if it preserves safe padded writes and attention performance.

**Proposal:** derive the context and checkpoint budget from weights, KV allocation, recurrent state, executable workspace, and a measured reserve. Include host executable caches and parking allocations in the host-memory budget. Test memory during snapshot creation, speculative rollback, and conversation switching, not only steady decode.

## 5. Move more GDN work outside the sequential scan

Location: `bonsai_tpu/model.py`, `gdn_chunked()`.

The current scan computes, in matrix notation:

```text
delta = Tinv @ (beta * (V - gamma * K @ S.T))
```

Precompute across chunks:

```text
U = Tinv @ (beta * V)
W = Tinv @ ((beta * gamma) * K)
```

The state-dependent step then uses `delta = U - W @ S.T`. This moves the triangular application outside the recurrent dependency chain. Measure the extra intermediate storage and numerical effects; algebraic equivalence does not imply identical floating-point behavior.

The older full-model ablation in `results/prefill-opt.json` suggests roughly 41 ms of GDN-associated time per 512-token prefill in that revision. Treat this as historical context, not an available guaranteed saving. A fused Pallas chunk kernel retaining recurrent state in fast memory is a subsequent experiment if profiling confirms that traffic remains material. Validate long recurrent sequences and chunk boundaries against the scan reference.

## 6. Target long-context attention next

The saved 96K run reaches approximately 1,279 prefill tokens/s and 42.2 decode tokens/s near the end of its context. The newer compact-scale headline benchmark records about 77.6 decode tokens/s for its short-prompt workload and roughly 2,154 prefill tokens/s at 512 tokens. These are different configurations, but clearly motivate context-dependent profiling.

Location: `bonsai_tpu/model.py`, `attend()` and the prefill attention path.

**Proposal:** profile decode and prefill attention separately. For prefill, investigate fused query/key tiling and online softmax to reduce large score intermediates and memory traffic. Inspect generated code or a device profile first: XLA may already eliminate some apparent intermediates. Decode has a very small query dimension and may need a different kernel.

Quantized KV, with scales decoded inside attention, is a plausible route to both lower bandwidth and larger contexts. INT8 halves the payload relative to BF16, before scale overhead. It requires retrieval tests, logit comparisons, and long-generation quality checks; capacity alone is not a sufficient result. Do not promise 128K or larger serving until workspace and transient memory have been measured.

## 7. Smaller improvements and experiments to deprioritize

- **Greedy sampling:** `sample_token()` computes top-64 filtering and categorical sampling before selecting the greedy result with `jnp.where`. Try a `lax.cond` branch using direct argmax for nonpositive temperature. Measure the full decode block; any gain is likely small.
- **Incremental streaming:** `StreamDecoder` can repeatedly decode a growing newline-free prefix, while the server output splitter repeatedly scans accumulated text. Use bounded lookbehind and incremental parsing if profiling shows host cost on long code/tool output. Preserve UTF-8 and partial delimiter handling.
- **Weight kernels:** existing profiles already show weight reads dominating short-context decode. Synthetic read-only kernels are useful comparisons, but are not proof of a physical bandwidth ceiling. Inspect their generated transfers before interpreting them as such.
- **Compact scales:** already implemented in `30a3016`. The headline report accounts for 399,974,400 scale bytes versus the previous 799,948,800. This should not be treated as future work or counted again in projected gains.

## 8. Improve experiment reliability before comparing gains

`scripts/run.sh` extracts source, removes imported modules, changes environment variables, and replaces shared `_LOG`, `_SRC`, and `print` globals **before** checking whether `_THREAD` is active. A second invocation can report `BUSY` after already disturbing the running experiment. This matters particularly while multiple collaborators share the TPU session.

Move the busy check and ownership acquisition before remote mutation; use per-run namespaces and captured source/log handles. Return structured success/failure status: `RUN_DONE` is currently emitted after exceptions too. This is a proposed change only; no runner was invoked during this review.

Further validation work:

- Preserve an explicit HTTP `seed=0`: `server.py` currently replaces it with a time-derived seed through `body.get('seed') or ...`.
- Record commit, dirty source hash, model revision, JAX/libtpu versions, context, KV dtype, checkpoint policy, and speculation settings in every benchmark. Separate compilation, warmup, transfer, and steady execution; report repeated trials and end-to-end TTFT.
- Refresh old profiling scripts before reuse. Some reference removed signatures or patch functions no longer on the hot path, including `runs/profile_prefill.py` and `runs/profile_prefill2.py`.
- Extend reference evaluation beyond the saved 320 teacher-forced comparisons. The reported 319/320 top-1 agreement and mean KL near `1.3e-5` are encouraging, but do not establish identical long generation or long-context quality.
- Add focused regression tests for speculative rejection, parking eviction, snapshot budget boundaries, rewind/replay, and interleaved conversations. Replay an actual tool-use conversation and compare logits after each restore.

Recommended execution order: fix sampling and parking correctness; secure the shared runner; benchmark adaptive prefill scheduling; measure transient memory in a fresh process; then investigate GDN and attention kernels. This prioritizes concrete latency and reliability improvements while preserving the gains already achieved.
