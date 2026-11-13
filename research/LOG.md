# Research log

## 2026-11-13 — Initial audit and first TPU experiments

Full scope is preserved in mission.txt. This is an ongoing implementation, not a completed inference engine.

- Initialized the previously empty workspace as a Git repository.
- Read CLI bundled documentation and implementation. Backend listed an existing assignment while local state was empty. Registered that assignment using the installed CLI's StateStore and list_assignments API, then selected its sole existing idle Jupyter kernel via GET /api/kernels. No runtime allocation, restart, or termination occurred. CLI alias: bonsai-tpu.
- Reference checkout: PrismML-Eng/llama.cpp, commit 6447906bf96c1a023980d3453fb59daa6af3bb23. Ignored upstream checkout; recover via git clone and checkout of this revision.
- Model revision: 839bd57023c1558945afea70a74fa83c054f35f9. Read first 16 MiB of actual PTQ1_0 GGUF with HTTP Range; metadata and all 851 tensor descriptors saved in gguf-metadata.json. Data begins at offset 11120992. Full weights have not yet been downloaded. Reader uses HF_TOKEN if present; this public metadata read did not require authentication.
- Actual GGUF declares qwen35 architecture, 64 blocks, width 5120, FFN width 17408, full attention every fourth layer (16 full, 48 recurrent). Full attention: 24 query heads, 4 KV heads, key/value width 256. GDN: 16 key heads, 48 value heads, state width 128, convolution width 4.
- Hadamard: normalized Sylvester transform, block 1024, explicit signs, inverse transformation for token embeddings. gdn_v_grouped=true. llama-graph.cpp build_hadamard_input permutes tiled value heads to grouped order BEFORE applying signs and Hadamard. Must preserve this; ordinary dequantization is insufficient.
- PTQ1_0 blocks: 24 payload bytes, 2 tail bytes, trailing FP16 scale (28 bytes total). Decode stages 16 then 8 bytes; uint8 multiplication wraps before extracting trits. PQ2_0 blocks: leading FP16 scale plus 32 payload bytes; adjacent 2-bit values map to q-1, including code 3 -> +2 in reference decoder.

### Measurements

Source: scripts/hardware_baseline.py. Raw repeated synchronized timings: results/hardware-baseline.json. Python 3.13.18, JAX 0.11.2, one TPU v5 lite, 24 CPUs. JAX reports 16,909,336,064-byte allocation limit. This is not the free model budget.

2048-cubed matmul median: BF16 0.304429 ms; INT8 0.304030 ms. First execution including compile: 0.5725 s and 0.6878 s respectively. Inputs are runtime arrays; only all-ones correctness checked. This does not establish application speed or numerical fidelity on arbitrary inputs. Approximately 56.4 TOP/s counting two operations per multiply-add; not a hardware peak measurement.

128 MiB pinned-host -> device median: 9.685 ms (~13.86 decimal GB/s). Reverse: 96.114 ms (~1.40 GB/s), with strong variability and increasing samples. Hypothesis: host allocation/lifetime or transfer synchronization contributes. Investigate reusable buffers, changing payloads, larger trials, and profiler traces before using these values for an offload decision. Current tests include allocation and dispatch overhead.

### Memory estimates (not context support claims)

Full-attention BF16 KV: 16 layers * 2 K/V * 4 heads * 256 elements * 2 bytes = 65,536 bytes/token. Thus 8K=0.5 GiB, 16K=1 GiB, 32K=2 GiB, 64K=4 GiB, 128K=8 GiB, 262144=16 GiB. INT8 halves payload before scale overhead; 4-bit quarters it. FP32 GDN matrix states alone: 48 layers * 48 heads * 128 * 128 * 4 = 144 MiB. Convolution state, temporary buffers, compiler/runtime memory and metadata are additional. Full BF16 KV at native context cannot coexist with model weights in this HBM; quantization/offload must be measured.

### Next work

1. Establish reference CPU binary from pinned fork and real-weight test vectors; inspect exact Hadamard sign mapping and embedding inverse.
2. Implement decoders and compare exhaustive encoded-byte and actual-weight vectors with C reference.
3. Benchmark JAX and Pallas packed matmul candidates on TPU with actual layer shapes, including separate decode/prefill paths.
4. Continue all mission milestones; full execution, numerical validation, optimization, context tests and packaging remain undone.

Recovery: existing session can be used with `colab exec -s bonsai-tpu -f scripts/hardware_baseline.py`; durable source and results are local. Never use `colab run` for recovery, since that provisions a runtime. Reinspect assignments and existing kernel before attaching if local session state is lost.

## 2026-11-13 — Codec validation, CPU oracle, and first fused TPU kernels

Previous goal turn classified as progress: committed architecture audit and hardware experiments. This turn revalidated the clean worktree and live IDLE TPU assignment before work.

### Reference and real weights

Built pinned Prism fork with CMake 4.4.4/Ninja, native CPU backend. Rebuild command is `bash scripts/build_reference.sh`. Local Python dependencies recorded in local-dependencies.txt. Full pinned PTQ1_0 checkpoint downloaded to weights/ using existing Hugging Face cached authentication (HF_TOKEN itself was not in the shell environment). `scripts/download_model.py` selects HF_TOKEN first, then the authenticated HF credential cache. No token values are logged.

Reference CPU command:

```
upstream/prism-llama/build/bin/llama-completion -m weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf -p 'The capital of France is' -n 24 -c 512 -b 32 -ub 32 -t 8 --temp 0 --seed 42 -no-cnv --no-display-prompt --verbose-prompt
```

Produced ` Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is`. Full logs in reference-greedy.log/txt. CPU decode ~4 tokens/s is a correctness oracle demonstration on the local host, NOT TPU inference or a comparable hardware speed baseline.

`reference/dump_logits.cpp` saves raw little-endian float32 logits and token IDs with the public llama API, no sampling penalties. Eight steps at vocab size 248320 are finite. Prompt IDs [760,6511,314,9338,369]; greedy IDs [11751,13,198,760,6511,314,9564,369]. Files: results/reference-capital.{json,f32}. Optional BONSAI_CAPTURE_DIR saves first two evaluations' layer-0/layer-3 and embedding intermediates, preserving GGML shapes and byte strides. 117 FP32 tensors captured, 19 MiB locally. Enabling capture preserved all eight logit vectors bit-for-bit. See reference-validation.json and reference-tensors/manifest.jsonl. Binary intermediate captures are local ignored artifacts, reproducible with:

```
BONSAI_CAPTURE_DIR=results/reference-tensors reference/dump_logits weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf 'The capital of France is' results/reference-capital-capture 8
```

### Exact codec and basis tests

Implemented NumPy/JAX-compatible PTQ1_0 and PQ2_0 unpacking, FP16 scales, lossless PTQ1 packing, normalized FWHT, forward signs, inverse embedding rotation, and GDN head permutation. Five local tests pass. Codec tests call actual dequantize_row_* symbols from pinned libggml-base.so; they cover each byte value at every payload position and 1024 random mixed blocks per format, with signed/zero/subnormal/extreme finite scales. Separate PTQ1 round-trip and dense Hadamard/inverse and explicit head-index tests pass.

Sampled four nonadjacent rows each from embeddings, LM head, recurrent QKV, FFN down, GDN out, and full-attention Q. NumPy decoding matches C exactly. TPU decoding and activation transforms match NumPy exactly on these vectors (results/tpu-codecs.json). This validates sampled codec/basis behavior, not complete layers, embedding inverse on TPU, all tensor exceptions, or full-model accuracy.

Unscoped pytest initially collected unrelated upstream tests and failed on their missing optional dependencies. Added project testpaths=['tests']; intended five project tests pass. Real-vector scripts currently import C initialization from the test module; refactor into a reference utility when expanding the oracle.

### Full-size matrix experiments

Actual blk.0.ffn_gate.weight, N=17408, K=5120. All timings include Hadamard activation transformation, synchronous completion and Python dispatch. Nine warm trials per candidate; compilation separate. Inputs are reproducibly seeded normal random values, not model activations. Raw machine-readable samples and compiler allocation stats in matmul-baseline.json and pallas-matmul.json.

Baseline medians (ms):

| M | XLA FP32 highest | XLA BF16 | INT8 groupwise activation | direct FP32 reduction |
|---|---:|---:|---:|---:|
| 1 | 0.582 | 0.584 | 0.568 | 0.590 |
| 32 | 1.526 | 1.037 | 0.576 | not run |
| 128 | 1.811 | 1.033 | 1.089 | not run |

XLA compiler reports zero HBM temporaries in M=1 and INT8 grouped cases, showing fusion rather than unconditional full-layer materialization. In BF16 prefill it reports ~179 MB temporaries, FP32 prefill ~357 MB. The original script docstring's assumption that all variants materialize a layer was disproved. M=1 BF16 output and generated code exactly matched FP32/direct variants; compiler precision optimization should be inspected before inferring actual intermediate rounding. INT8 grouped activation relative L2 error ~0.64%; kept experimental only.

Implemented custom Pallas tiled matmul: lossless internal 2-bit words with 16 adjacent ternary codes/uint32, physical layout [K/16,N]; separate FP32 scales [K/128,N]; local decode and BF16 reconstruction in VMEM; FP32 accumulation. Hadamard is compiled in the wrapper. Weight storage ~25 MB for this matrix vs ~19.5 MB on-disk PTQ1 including scales; exact memory trade-off pending full checkpoint conversion. Current validation only allows aligned dimensions, with bk divisible by 1024. No fallback is silently applied.

Initial bk=512 failed because scale BlockSpec has major extent 4 instead of TPU-required multiple 8. Retained failures in pallas-matmul-initial.json; fixed validation and swept legal tiles. Initial reference used default JAX matmul precision and understated numerical error; final pallas-matmul.json uses explicit HIGHEST. Do not cite initial near-zero error as FP32 fidelity.

Final best observed medians: M=1 0.392 ms (bm=128,bk=1024,bn=1024); M=32 0.422 ms (32,1024,512); M=128 0.473 ms (128,1024,1024). These are per-sweep medians, not yet repeated interleaved trials. Relative L2 errors vs FP32 highest are ~0.23%, consistent with BF16 input/weight reconstruction. All finite; no model-quality claim. Pallas memory analysis reports zero temporary HBM, not zero VMEM scratch or zero total memory. M=128 is ~2.18x faster than measured XLA BF16 at matched math precision; end-to-end improvement is untested.

Hardware/kernel references consulted:
- https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
- https://docs.jax.dev/en/latest/pallas/tpu/details.html
- https://docs.jax.dev/en/latest/pallas/tpu/hardware.html
- https://docs.cloud.google.com/tpu/docs/v5e

### Next required implementation

1. Implement a bounded-memory GGUF tensor loader, exact embedding inverse, RMSNorm and first full recurrent block; compare all available reference tensors. Inspect qwen35.cpp for normalization offsets, grouped/tiled QKV, recurrent updates and gating.
2. Implement full-attention block and RoPE against layer-3 captures, then full 64-layer TPU forward and logits validation. Use correctness-first FP32 paths where needed; no BF16 quality assumption.
3. Explore PTQ1 Pallas layout, direct GEMV, higher precision local reconstruction and tile sweeps using real activations; preserve numerical gates and compare representations.
4. Revisit existing TPU frameworks (SGLang-JAX/MaxText) before final runtime architectural choice; this remains outstanding.
5. All full-model TPU generation, decode/prefill optimization, long contexts, quality evaluation, interface and final report requirements remain active.

All launched experiments this turn completed. Session left intact. Artifacts are durable locally; sources, tests, numeric benchmark JSON and reference logits tracked in Git. Full model and regenerable binary vectors/captures are ignored to avoid placing gigabytes in Git.

## 2026-11-13 — Full-model TPU decode and prefill; int8 block-diagonal kernel (session 3, Claude)

Resumed from the Codex transcript. Earlier "K RoPE mismatch" (38% error) was a capture-name collision
(`Kcur` is emitted both before normalisation and after RoPE); Q RoPE matched at 3e-7 with the same code and
attention outputs with cache continuation matched, so RoPE was correct. The ~0.5% per-projection deviation
from the CPU oracle is the oracle's own Q8_0 activation quantisation (`vec_dot_type = Q8_0` for PTQ1_0),
i.e. the CPU reference is itself not an exact evaluation of the ternary model.

### Decode-regime kernel study (all chained in one jitted program, 8 distinct 5120x17408 matrices)

Per-matrix medians, results in gemv-variants.json, pallas-chain.json, i8-kernels.json, bd-kernel.json,
t5-kernel.json. Standalone timings in the previous entry included ~0.4 ms of dispatch and are superseded.

| Variant | M=1 ms/matrix | Note |
|---|---:|---|
| XLA fused u32 unpack, FP32 or BF16 | 0.152 | elementwise reduce path, unpack-bound |
| XLA unpack only / read only | 0.125 / 0.057 | unpack in 32-bit lanes dominates |
| XLA same, M>=2 | 0.70-1.12 | falls off the fused path, materialises weights |
| XLA int8 pre-expanded (1 byte/weight) | 0.196 | 4x bytes: slower than packed; HBM-bound |
| Pallas u32 unpack -> BF16 (previous kernel) | 0.140 | unpack-bound |
| Pallas u8 lanes, int8 MXU, 8 dots/tile | 0.064 | unpack nearly free |
| **Pallas u8 lanes, block-diagonal int8, 1 dot/tile** | **0.061** | equals pure-read floor 0.061 |
| T5 (5 trits/byte, 1.6 bit) i32 / i16-mul / i16-shift-add | 0.114 / 0.131 / 0.169 | read floor 0.057 |

Findings:
- Mosaic supports bitwise ops on i8 vectors but no i8 arithmetic, and cannot load f16. Shift+mask
  unpacking in 8-bit lanes costs ~nothing; widening to 32-bit lanes costs more than the matmul.
- Decode is HBM-bandwidth-bound: ~25 MB (2-bit weights + FP32 scales) per 0.061 ms ~= 410 GB/s, the same
  rate as a kernel that only reads. Tile size does not matter between 0.5 and 4.4 MB.
- The denser 1.6-bit packing is SLOWER on this hardware (base-3 digit extraction needs 16/32-bit
  multiplies): 0.114 ms vs 0.061 ms although it reads 20% fewer bytes. "Smallest format" loses.
- Pre-expanded int8 weights lose too (bandwidth). 2-bit with byte-lane unpack is the optimum found.

Chosen decode kernel ("B4 block-diagonal"): weights uint8 [K/4, N], bit pair j of byte row 32g+i holds
code t+1 for input 128g+32j+i, so four whole-tile shift/masks give four contiguous planes. Activation is
quantised to 16 bits (global per-vector scale), split into signed high/low bytes, and laid out
block-diagonally: row g of the operand carries only scale group g. One int8 MXU product per 1024-input
tile yields every group's exact integer partial sum; `256*hi + lo` is exact in int32; a single FP32
multiply by the group scale and a sum over 8 rows finish the tile. The t+1 offset is removed with a
40xN correction. Error vs float64 on random data: 3.2e-5 relative (16-bit activations) — more precise
than the CPU reference's 8-bit activations and than BF16.

### Full model

- On-device PTQ1_0 -> B4 conversion (exact decode validated earlier), matrices sharing an input fused
  ([qkv|gate], [ffn_gate|ffn_up], [q|k|v]). Load 12.7 s from page cache. HBM in use 7.35 GiB.
- 64-layer decode step, single XLA program, 52 s first compile.
- Reference logits ("The capital of France is", 8 greedy steps): identical argmax every step, relative
  L2 0.21-0.32%, KL(ref||ours) 5e-7..7e-5 (first-logits.json).
- **Decode 20.65 ms/token median = 48.4 tok/s** (30 steps, min 20.39 ms), logits left on device.
  7.35 GiB / 410 GB/s = 17.9 ms is the bandwidth floor for this layout, so ~2.7 ms is everything else.
- Batched prefill: B4 GEMM kernel (8-bit-lane unpack, scale folded into BF16 weights in VMEM, FP32
  accumulate; 0.60 ms per matrix at M=512 ~= 150 TFLOP/s, 0.175 ms at M=128; gemm-b4.json).
  256-token chunk: 0.205 s = 1247 tok/s with a sequential GDN scan. Batched vs token-by-token on the same
  256 tokens: logits rel L2 0.13%, KL 9e-6, GDN state 0.5%, K cache 0.44% (BF16 prefill vs 16-bit decode).
  Prefill-then-decode vs CPU reference: same argmax, KL <= 7e-5 (prefill-check.json).

## 2026-11-13 — Decode 78 tok/s, prefill 1700 tok/s, 64K context, agentic serving (session 3, continued)

### Decode profile and fixes (profile-decode.json, profile-overhead.json, kernel-floor.json, dma-streams.json)

Ablation of the first 20.65 ms step: Pallas kernels 11.0 ms, Hadamard butterflies 6-7 ms (256 rotations x
10 tiny XLA stages), scale re-read for the code-offset correction ~1.4 ms, jit dispatch of a ~1000-leaf
pytree ~1.3 ms, recurrent math 0.9 ms, attention 0.45 ms, embedding 0.12 ms.

| Change | Step time |
|---|---:|
| first full model | 20.65 ms |
| Hadamard as H32 (x) H32 MXU products at HIGHEST precision (6.14 -> 0.63 ms/token, error 9e-8) | |
| code-offset correction inside the kernel, in integers (no second pass over scales) | 13.4 ms |
| on-device while_loop over tokens (no per-token dispatch), exact two-level top-k sampler | 12.8 ms (78 tok/s) |

- hadamard-variants.json: butterfly 6.14 ms, single HIGHEST 1024x1024 product 1.30 ms, Kronecker 0.63 ms,
  default-precision product 0.59 ms but 0.17% error.
- Read-only Pallas kernels over all 257 weight matrices take 10.5 ms for 7.2 GB = ~690 GB/s whatever the
  tile shape and with 1, 2 or 4 concurrent DMA streams per step: this is the HBM read ceiling. The real
  GEMV pass is 11.5 ms. Decode therefore sits ~10% above the bandwidth floor of the 2-bit format plus
  ~2 ms of non-matmul work. Remaining ideas: 16-bit scale storage (-0.6 ms, needs an integer->float
  bitcast in Mosaic), fusing small recurrent ops.
- lax.top_k over 248K logits costs 0.56 ms; the exact two-level variant (top rows by row maximum, then
  top-k of those rows) 0.19 ms.

### Prefill (profile-prefill.json, profile-prefill-parts.json, gdn-chunked.json)

At M=512 the sequential GDN scan cost 4.1 ms/layer and HIGHEST-precision Hadamard 1.4 ms per 17408-wide
rotation; the GEMMs were only ~40% of the time.
- Chunk-parallel gated delta rule (derivation in model.gdn_chunked): all unit-triangular chunk systems are
  inverted at once, leaving T/32 sequential steps. 1.23 ms/layer, agrees with the scan to 1.4e-5.
- XLA folds an f32->bf16->f32 round trip away, so "split into bf16 pieces" must be done by masking mantissa
  bits. With exact pieces and a +-1 matrix, two single-pass BF16 MXU products give a ~16-bit-accurate
  Hadamard at 0.57 ms (was 1.41 ms).
- `jnp` fancy-index gathers for the key-head tiling were replaced by `tile`.
- Padded chunks: prefill_chunk takes n_valid; padding is a no-op for GDN (beta=0, gate=0, k=0), the conv
  state is sliced at n_valid, KV rows beyond n_valid are overwritten later. Two compiled sizes (512, 64)
  serve any prompt length.

### Context scaling (bench-context-*.json; BF16 KV, n_ctx 65536, HBM in use 11.4 GiB)

| Prompt tokens | Prefill tok/s (warm) | Decode tok/s after prompt |
|---:|---:|---:|
| 432 (FP32 KV) | 1376 | 77.5 |
| 1968 | 1842 | 76.0 |
| 8112 | 1743 | 71.1 |
| 32688 | 1314 | 59.2 |
| 65456 | 1183 (55.3 s) | 49.9 |

These used the earlier power-of-two attention buckets. Attention is now a streaming online softmax in a
while_loop over 4096-key blocks: one program for every length, temporaries bounded by the block, cost
follows the used length. Block decode with BF16 KV: 13.1 ms at ~0 context, 13.4 ms at 8K, 18.8 ms at 60K
(att-block.json; 1024/2048/8192-key blocks are within 10%).

### Host memory and compile time

- A host OOM killed the kernel during development: every cached TPU executable pins ~1.1 GiB of host RAM
  (47 GiB after ~30 programs). `jax.clear_caches()` + `malloc_trim` releases it. The engine now needs only
  three programs (decode block, prefill 512, prefill 64) and bounds its program cache.
- Persistent XLA cache works on TPU: a 36 s compile becomes a 5 s load; engine warm-up 118 s cold, ~20 s warm.
- Engine.reset reuses KV buffers (a second multi-GiB allocation OOMed HBM at 64K).

### Agentic serving

- Recurrent-state checkpoints (~155 MiB each, in HBM): at prompt end, every 4096 tokens, at the start of
  the conversation proper (after system prompt and tool schemas) and at the last message. A request resumes
  from the nearest checkpoint at or before the first changed token; the KV cache rewinds for free.
- OpenAI-compatible server (stdlib): tools -> tool_calls (model-native <tool_call><function=..> blocks),
  reasoning_content, stop strings, cached-token usage, llama.cpp-style timings, /metrics.
- Per-message tokenization cache (exact because <|im_start|> is a control token): host preparation of a
  21K-token conversation takes 3-4 ms.
- First tokens stream after 1, 2, 4, ... token blocks.
- Reachable locally through `scripts/tunnel.sh` (OpenSSH over `colab ssh --proxy-mode`).
- pi 1.5.0 (already installed on this machine; the installer was inspected, not re-run) drives the engine
  with `PI_CODING_AGENT_DIR=pi-agent`: a write-file/run/report task finished in 3.8 s wall.
- bench/agent_loop_client.py (12 turns, +1.9K tokens of tool output per turn, context to 21K): 23.5 s
  total, 108K tokens served from state vs 20.9K fed, per-turn prefill 1.15-1.40 s, decode 62-69 tok/s,
  all 11 planted values retrieved correctly (agent-loop-client.json). Turn time is dominated by prefilling
  new tool output, so prefill throughput is the main lever for this workload.

## 2026-11-13 — Speculation, parking, 98K context, int16 scales, default model, report (session 3, end)

- User direction: agentic use; pi harness runs locally, Colab TPU is inference only; default model is the
  Qwen 3.8-based `prism-ml/Ternary-Bonsai-2-27B-gguf` (this was the model all along; `qwen35` is the GGUF
  architecture tag). `bonsai_tpu/defaults.py` pins repo/file/revision and auto-downloads.
- Thinking parameter: the template's `enable_thinking` / `reasoning_effort`; server accepts `thinking`,
  `enable_thinking`, `chat_template_kwargs`, `reasoning_effort`; default follows the template (on). Verified
  through pi: off -> 3 tokens, low -> 31, high -> 67 for the same question.
- Prompt-lookup speculative decoding (speculation.json): file rewrite 70.4 -> 571 tok/s, quoting explanation
  70.4 -> 119.2, essay 70.5 -> 70.3; identical token streams in all three.
- Conversation parking in host RAM (parking.json): 19K-token session, park 0.84-1.35 s (1.83 GiB), restore
  0.33-0.57 s, same answer after restore.
- Fused rotate kernel for prefill: 512-chunk 276 -> 239 ms. Ablation before it (prefill-opt.json): GDN 41 ms,
  rotations 58 ms, attention 21 ms, rest (GEMM) 156 ms.
- int16 half-coded scales decoded with integer ops inside the kernels: bit-identical logits, -0.37 GiB HBM,
  40-matrix chain 1.857 -> 1.793 ms. Scale statistics: all positive, 2^-18.1 .. 2^-1.2, 492 f16 subnormals.
- 96K context (bench-context-bfloat16-98304.json): 97,920-token prompt in 76.6 s, 42 tok/s decode after it,
  14.0 GiB HBM in use with three checkpoints.
- Headline (final-bench.json): decode 77.6 tok/s (5 trials), TTFT 49 ms, prefill 2154/2167/2110/1996/1800
  tok/s at 512/2K/8K/16K/32K.
- Reference evaluation (eval-reference.json): 319/320 top-1, mean KL 1.3e-5 for all three numeric paths.
- pi 1.5.0 was already installed locally; the pi.dev installer was downloaded and read, not executed.
  `PI_CODING_AGENT_DIR=pi-agent` keeps the user's own pi configuration untouched.
- Correction: SGLang-JAX does contain a qwen3_5 model and GDN kernels (no ternary/Hadamard support); see the
  report, section 4.
- A mistake to avoid: `pkill -f` with a pattern that also matches the invoking shell killed the tool shell
  twice.

Recovery: `scripts/start_server.sh [n_ctx]` (attach-only; never provisions). See README "Recovery".

## 2026-11-13 — Cost-aware prefill scheduling (proposition 3)

- Calibrated the existing 64/512 prefill programs and one-step decode at 0, 8K, 32K and 60K context.
  A host dynamic program selects chunk sizes using interpolated costs; one-token tails reuse the warmed
  decode block without affecting generation's PRNG. No new model executable shape.
- Paired TPU benchmark: 48 context/suffix combinations, three trials per policy, identical restored prefixes.
  A 65-token suffix improves from ~238 to ~67 ms at short context and ~447 to ~84 ms at 60K context;
  a 128-token suffix improves from ~238 to ~94 ms. Full 512-token chunks retain their existing path.
- CPU: 26 tests pass. TPU: state/KV comparisons, exact checkpoint replay and CPU-reference evaluation
  (both policies 319/320 top-1, mean KL ~1.3e-5) pass. All 48 immediate next-token argmaxes agree.
- BF16 chunk boundaries can flip near ties: the 320-token generation example diverges at output index 5,
  where the old top-two logit gap is 0.00011 and the shared-prefix distribution KL is 5.7e-6. This is
  documented rather than asserting bit-identical free generation across policies.
- Reproduction, TTFT, padding-work interpretation and raw data: research/PREFILL_SCHEDULING.md,
  results/prefill-costs.json, results/prefill-schedule.json. Final TPU run: PREFILL_SCHEDULE_VALIDATED.

## 2026-11-13 — Portable Colab startup and pi integration

- New entrypoint: python3 -m bonsai_tpu.setup_colab --hf-token ... . Local-only preparation can install
  pi before TPU allocation; existing pi is reused. Installation uses a private npm prefix and the current
  tested pi 1.5.2. models.json is merged atomically, backed up, and left unchanged on repeated setup;
  other providers, credentials, models and custom sampling values are preserved.
- Runtime bootstrap attaches only, installs missing dependencies, stages/consumes HF credentials without
  embedding their values in executed source, reuses resident parameters, warms the engine and serves.
  Same-source/context setup rejoins a running job or reuses a healthy server. JAX upgrades explicitly
  request a kernel restart when needed. No automatic runtime allocation or replacement.
- The primary local API bridge carries HTTP/SSE through authenticated Colab CLI exec; no SSH is required.
  README now documents authentication, Node/pi prerequisites, token arguments/environment/files, setup,
  pi/API smoke tests, recovery, cleanup, and the variable free-tier limits (90 minutes is not guaranteed).
- Validation: 37 CPU tests pass, including configuration preservation/idempotence, missing/existing pi,
  token staging, dependency-restart handling, transport round trips/errors and literal prompt handling.
  Installed pi 1.5.2, ran a real pi prompt through the bridge (42), real nonstream API (21) and streamed
  French greeting. Tested server reuse and placeholder-token cleanup on the existing loaded runtime;
  a fresh authenticated 5.9 GB download was not remeasured. Raw smoke evidence: results/setup-colab.json.
- Current local pi profile points at port 8001 because the earlier development SSH tunnel occupies 8000.
  The new setup defaults to 8000 on a fresh machine. Existing unrelated defaults.py edits are not included.
