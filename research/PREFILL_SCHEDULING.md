# Cost-aware prefill scheduling — implementation and measurements

Implemented proposition 3 on the single Colab TPU v5e-1, with BF16 KV, a 65,536-token allocation and the compact-scale weights. The implementation is in `bonsai_tpu/engine.py`; raw measurements are in [`prefill-costs.json`](../results/prefill-costs.json) and [`prefill-schedule.json`](../results/prefill-schedule.json).

## Behavior

`prefill_plan()` minimizes estimated execution time over the existing 64- and 512-token executables. It interpolates measured costs at the starting context of each chunk, using calibration points at 0, 8,192, 32,768 and 61,440 tokens. Costs outside that range are clamped. Only the last chunk is padded. A final single token uses one step of the existing decode block, retaining its logits and discarding its sampled token and private PRNG state. No additional executable shape is introduced.

Message-boundary checkpoints still split the input at the requested positions. History, valid-token counts, recurrent state and pending logits advance only by real tokens. The planner takes O(ceil(tokens / 64)) host work.

## Measured latency and speed

Paired warm runs restore the same populated prefix before each policy, synchronize all returned buffers, alternate policy order and use three trials per case. Prefix restoration and compilation are excluded; host planning and dispatch are included. The old policy is exactly the previous rule: use 512 whenever more than 64 tokens remain. All 48 combinations of four contexts and twelve suffix lengths are saved in the raw JSON.

| Cached context | New tokens | Old median | New median | Speedup | Latency saved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 47.42 ms | 23.35 ms | 2.03× | 24.06 ms |
| 0 | 65 | 238.07 ms | 66.92 ms | 3.56× | 171.15 ms |
| 0 | 128 | 238.07 ms | 92.89 ms | 2.56× | 145.19 ms |
| 0 | 192 | 238.63 ms | 139.17 ms | 1.71× | 99.46 ms |
| 0 | 256 | 238.32 ms | 184.02 ms | 1.30× | 54.29 ms |
| 0 | 320 | 238.40 ms | 229.62 ms | 1.04× | 8.78 ms |
| 0 | 512 | 238.06 ms | 237.93 ms | 1.00× | 0.14 ms |
| 0 | 577 | 473.92 ms | 303.11 ms | 1.56× | 170.81 ms |
| 8,192 | 65 | 265.90 ms | 68.60 ms | 3.88× | 197.30 ms |
| 32,768 | 128 | 350.02 ms | 104.87 ms | 3.34× | 245.14 ms |
| 61,440 | 65 | 447.50 ms | 83.71 ms | 5.35× | 363.80 ms |
| 61,440 | 128 | 447.49 ms | 115.63 ms | 3.87× | 331.86 ms |
| 61,440 | 320 | 448.01 ms | 285.50 ms | 1.57× | 162.51 ms |
| 61,440 | 512 | 447.91 ms | 447.57 ms | 1.00× | 0.34 ms |

Cases using identical chunk sequences differ by at most 0.51 ms between paired medians in this run. This is not evidence of a steady-decode improvement: the change targets prompt/suffix processing. Large prompts made entirely of full 512-token chunks retain their existing path.

## End-to-end time to first token

These warm engine measurements include prompt processing, the prompt-end recurrent-state snapshot and first-token sampling. Each prompt generates 32 tokens; three trials are recorded. They exclude HTTP transport and compilation.

| Prompt tokens | Old TTFT | New TTFT | Reduction |
| ---: | ---: | ---: | ---: |
| 1 | 48.33 ms | 22.96 ms | 52.5% |
| 65 | 238.67 ms | 67.48 ms | 71.7% |
| 128 | 238.53 ms | 93.59 ms | 60.8% |
| 320 | 238.52 ms | 230.99 ms | 3.2% |

## Compute and memory interpretation

- For a 65-token suffix, scheduled positions fall from 512 to 65: one 64-token prefill plus one decode step. This removes 447 padded positions (87.3% of the old scheduled positions).
- For 128 real tokens, two 64-token passes replace one 512-token pass, eliminating 384 padded positions (75%). For 577 tokens, scheduled positions fall from 1,024 to 577 (43.7%).
- These counts quantify logical padding avoided, not hardware FLOPs or energy. MXU tile utilization and the decode kernel differ; multiple small passes can reread weights and repeat the vocabulary projection. No proportional HBM-traffic, energy or hardware-FLOP reduction is claimed. The measured latency is the decisive result.
- Resident weights, KV capacity and compiled bucket sizes are unchanged. The optimization does not depend on another large executable cache entry.

## Correctness and limits

Local CPU tests pass (26 tests), including an independent exhaustive scheduling-cost oracle, contextual cost interpolation, padding/history bookkeeping, message-boundary snapshots and context-overflow handling.

On TPU, all 48 paired suffix cases preserve the next-token argmax. The maximum KL divergence between policies is 1.56e-05; the maximum relative logit L2 error is 0.00184. Recurrent and valid KV rows are compared as well. A real checkpoint rewind and replay produces bit-identical logits.

- CPU-oracle comparison, old: 319/320 top-1 agreement across ten prompt types; mean KL 1.31e-05.
- CPU-oracle comparison, adaptive: 319/320 top-1 agreement across ten prompt types; mean KL 1.3e-05.

Greedy continuations are identical for the tested 1-, 65- and 128-token prompts. The 320-token prompt diverges at generated token index 5. At that shared prefix, the old path has only a 0.00011-logit gap between “ northern” and “ archive”; the new path chooses the other token, with KL approximately 5.7e-6 between the distributions. Changing BF16 chunk boundaries and using the decode path for one-token tails can change numerical rounding. Bit-identical generation across scheduling policies is therefore not a contract of this feature; the reference-quality and state checks remain the correctness gates.

The calibration targets v5e-1 and the current graph. Recalibrate after material kernel changes or on different hardware. Measurements cover cached contexts through 61,440 tokens, not a new maximum-context claim. Removing intermediate vocabulary projections was not needed for these gains and is deferred.

## Reproduction

With the existing `bonsai-tpu` session attached and idle:

```sh
RUN_ENV="BONSAI_COMMIT=$(git rev-parse HEAD)" scripts/run.sh runs/prefill_costs.py
RUN_ENV="BONSAI_COMMIT=$(git rev-parse HEAD)" scripts/run.sh runs/bench_prefill_schedule.py
colab download -s bonsai-tpu /content/bonsai-results/prefill-costs.json results/prefill-costs.json
colab download -s bonsai-tpu /content/bonsai-results/prefill-schedule.json results/prefill-schedule.json
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m pytest -q
```

The benchmark uses the existing CPU-oracle `eval-ref.npz` manifest and parts on the runtime. Results record the starting commit and engine source SHA-256 to identify the uncommitted implementation being measured. These runs stop the development server through the usual `runs/load.py` preamble; restart it afterward with `scripts/run.sh runs/serve.py`.
