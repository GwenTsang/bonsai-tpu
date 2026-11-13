# Ternary Bonsai 2 27B on one TPU v5e — technical report

State at 2026-11-13. Every number below was measured on the Colab TPU v5e-1 runtime (JAX 0.11.2, 16.9 GB
allocator limit, 24 vCPU, 47 GiB host RAM) unless it is marked as an estimate. Raw data: `results/*.json`;
chronology and method: [LOG.md](LOG.md).

Model: `prism-ml/Ternary-Bonsai-2-27B-gguf` (ternary Qwen 3.8 27B), PTQ1_0 file, revision `839bd57`.
The GGUF's architecture tag is `qwen35`: 64 blocks, width 5120, FFN 17408; every fourth block is full attention
(24 heads, 4 KV heads, dim 256), the other 48 are Gated DeltaNet (16 key / 48 value heads, 128x128 state,
causal conv width 4). 26.87 B ternary weights with FP16 scales per 128 inputs; activations pass through a
signed, normalized 1024-block Hadamard transform before every ternary matrix.

## 1. Results

| Metric | Value |
|---|---|
| Model / weight format | Ternary Bonsai 2 27B PTQ1_0 -> resident "B4" 2-bit + 16-bit scales |
| Batch size | 1 |
| Decode, warm, short context | **77.6 tok/s** greedy (5 runs of 512 tokens, 77.52-77.64); 77.6 with sampling |
| Decode after 8K / 16K / 32K / 64K / 98K prompt | 71.8 / 68.0 / 60.9 / 49.9 / 42.2 tok/s |
| Decode on copy-heavy output (speculative lookup) | 571 tok/s (file rewrite), 119 tok/s (quoting explanation) |
| Prefill, warm | 2154 tok/s (512), 2167 (2K), 2110 (8K), 1996 (16K), 1800 (32K), 1279 (98K, 76.6 s) |
| Time to first token | 49 ms (37-token prompt, warm) |
| Context tested | 98,304 (BF16 KV, 14.0 GiB HBM); serving default 65,536 (12.0 GiB) |
| HBM | model 6.71 GiB; + KV 65.5 KB/token; + recurrent state 0.15 GiB; + checkpoints 0.15 GiB each |
| Host RAM | ~10 GiB after load (5.8 GiB of it page cache of the GGUF) + ~1.1 GiB per compiled program (4) |
| Load | 13-14 s (page cache -> HBM, conversion on the TPU) |
| Compilation | 135 s cold for the four programs; ~20 s from the persistent XLA cache |
| Correctness vs Prism llama.cpp (CPU) | top-1 319/320 teacher-forced steps, mean KL 1.3e-5, max 4.5e-4 |
| CPU reference speed (dev host, 8 threads) | 4.0 tok/s decode, 4.6 tok/s prompt (correctness oracle, not a hardware baseline) |
| Commit | see `results/final-bench.json` (`commit` field) |

Agent-loop measurements (through the OpenAI endpoint, client on another machine, SSH tunnel):

| Scenario | Result |
|---|---|
| 12-turn tool loop, +1.9K tokens of tool output per turn, context to 21K | 20.3 s total; 108K tokens served from state vs 20.9K processed; mean TTFT 1.39 s |
| Follow-up turn on a 2.9K-token conversation | 0.11 s wall |
| Edited last message (checkpoint rewind) | 0.24 s wall |
| Unrelated request interleaved with a 19K-token conversation | conversation parked in 0.8-1.3 s, restored in 0.3-0.6 s (re-prefill would be 10 s) |
| pi 1.5.0: rename + constant refactor in a 300-line file, compile check | 13.6 s wall, 5 requests, correct |
| pi 1.5.0 with reasoning: module + 8 unit tests, run, report | 29.7 s wall, 4 requests, tests pass |

## 2. What limits speed on this chip

- **Decode is HBM-read-bound.** A Pallas kernel that only reads all weight matrices takes 10.5 ms for 7.2 GB:
  ~690 GB/s, independent of tile shape (0.5-4.4 MB) and of 1, 2 or 4 concurrent DMA streams. Every token must
  read every weight, so bytes per weight set the floor. With 2-bit codes and 16-bit scales the read floor is
  ~9.9 ms (101 tok/s, estimate from the measured rate); the measured step is 12.9 ms: ~11 ms kernels,
  ~0.9 ms GDN/conv, ~0.45 ms attention at short context, ~0.5 ms rotations and operand preparation.
- **Prefill is MXU-bound.** The GEMM kernel reaches ~150 TFLOP/s at M=512 (v5e nominal BF16 peak: 197). At
  51 GFLOP per token that is 2900 tok/s for the matmuls alone (estimate); measured end to end 2150 tok/s, the
  rest being GDN (15%), Hadamard (10%), attention (8%, growing with context).
- **Many small XLA ops are expensive.** Ten butterfly stages on a 5120-vector cost 27 us; 256 rotations per
  token cost 6-7 ms until they became two 32x32 MXU products. Fancy-index gathers and per-token jit dispatch
  (~1.3 ms for a ~1000-leaf pytree) showed the same pattern.

## 3. Kernels

Resident layout "B4" for W[N, K]: `uint8 [K/4, N]`, byte row `32g+i`, bit pair `j` holds `t+1` for input
`128g + 32j + i`; scales `int16 [K/128, N]` holding the released IEEE-half bits.

**Decode GEMV (block-diagonal int8).** Four whole-tile shift/mask operations in 8-bit lanes yield the codes
(Mosaic has bitwise but no arithmetic i8 ops; widening to 32-bit lanes costs more than the matmul). The
activation is quantised to 16 bits and split into signed high/low bytes laid out block-diagonally: row g of
the int8 operand holds only scale group g. One MXU product per 1024-input tile returns all eight groups'
partial sums as exact integers; `256*hi + lo - sum(q)` removes the code offset in int32; one FP32 multiply
by the decoded scale and a row sum finish the tile. Exact ternary weights, 16-bit activations (3.2e-5
relative error on random data) — more precise than the CPU reference (8-bit activations) or BF16.

**Prefill GEMM.** Same unpack, then `(code-1) * scale` reconstructed as BF16 in VMEM and four MXU products per
tile with FP32 accumulation. A second kernel fuses sign flip, Hadamard (two exact BF16 passes on a
bf16-split activation), the column permutation the GEMM expects, and the BF16 cast.

**Measured alternatives** (per 5120x17408 matrix, chained in one program):

| Variant | ms (M=1) | Verdict |
|---|---:|---|
| XLA fused u32 unpack + reduce | 0.152 | unpack-bound; falls to 0.7-1.1 ms for M>=2 |
| Pre-expanded int8 weights | 0.196 | 4x the bytes; slower. No room for 27 GB anyway |
| Pallas u32 unpack -> BF16 | 0.140 | unpack-bound |
| **B4 block-diagonal int8** | **0.061** | at the read floor |
| 1.6-bit "T5" (5 trits/byte) | 0.114-0.169 | 20% fewer bytes but base-3 digit extraction needs 16/32-bit multiplies |
| PTQ1_0 as released (26-byte blocks + inline scale) | not run as a kernel | not tileable (blocks are not lane-aligned); its density class is T5's |
| Direct ternary add/subtract, bit planes | not competitive by construction | the MXU does the multiply for free; cost is unpack and bytes |

So the smaller format is slower here, pre-expansion is slower, and 2-bit with byte-lane unpacking is the
optimum found. PQ2_0 (2-bit on disk) would map to B4 directly; PTQ1_0 is converted on the TPU in 13 s.

## 4. Architecture decisions

- **Standalone runtime.** SGLang-JAX (checkout `b3e3e16`, Apache-2.0) does implement this architecture family
  (`models/qwen3_5.py`, a Gated DeltaNet backend with a chunked JAX path and a fused chunk-parallel Pallas
  prefill kernel for BF16 activations), but it has no ternary formats, no group scales and no Hadamard
  folding. The dominant cost here is the packed ternary matmul and the memory layout around it, which would
  replace that framework's linear layers, weight loading and state pools; a single-sequence runtime written
  around the kernels was smaller than the adaptation. Nothing was reused. Its fused GDN prefill kernel is
  a credible way to cut the 15% of prefill time spent in the XLA chunked GDN here; it was not ported or
  measured.
- **One decode program, token loop on device.** `lax.while_loop` over steps with an exact two-level top-k
  sampler and stop detection; the host is touched once per block (1, 2, 4, 8, 16 tokens).
- **Chunk-parallel Gated DeltaNet** for prefill: all unit-triangular chunk systems are inverted at once;
  T/32 sequential steps remain. Agrees with the sequential rule to 1.4e-5.
- **Streaming attention**: online softmax over 4096-key blocks in a while_loop; one program for every
  context length, bounded temporaries.
- **Padded prefill chunks**: sizes 512 and 64 serve any length; padding is a no-op for all state.
- **Four compiled programs** (decode block, prefill 512, prefill 64, verify 32).

## 5. Agent-loop design

- Recurrent state cannot be rolled back, the KV cache can. Checkpoints of the recurrent state (0.15 GiB) are
  taken at the prompt end, every 4096 tokens, after the system prompt/tool schemas and at the last message;
  a request resumes from the nearest one before the first changed token.
- Conversations that a request would displace are parked in host RAM and re-uploaded on return.
- Prompt-lookup speculation: decode is bandwidth-bound, so a 32-position batched pass costs about two
  steps. Drafts are the continuation of the latest context occurrence of the last six tokens; each position
  is still sampled from the model's distribution. On the tested tasks the token stream was identical to
  plain decoding; with no match there is no overhead.
- Per-message tokenization cache (3-4 ms host preparation for a 21K-token conversation).
- Tool calls, reasoning (`thinking` parameter, effort levels), stop strings, cached-token accounting.

## 6. Memory

HBM at n_ctx 65536 with six checkpoints (`results/memory-report.json`, before the int16-scale change):
weights 5.96, scales 0.745 (now 0.37), embedding 0.28, small tensors 0.10, KV 4.25, recurrent 0.15,
checkpoints 0.88 GiB; accounted 12.35 of 12.46 GiB in use.

KV cost is 65,536 B/token in BF16 (16 layers x 2 x 4 heads x 256 x 2 B). With 6.71 GiB of model:

| Context | BF16 KV | Fits in 15.7 GiB? |
|---|---:|---|
| 32K | 2.0 GiB | yes |
| 64K | 4.0 GiB | yes (tested) |
| 98K | 6.0 GiB | yes (tested, 14.0 GiB in use with 3 checkpoints) |
| 128K | 8.0 GiB | no (BF16); int8 KV would need 4 GiB — not implemented |
| 262K | 16 GiB | no; int8 8 GiB would not leave working memory; needs host paging — not implemented |

Host->device transfers run at ~14 GB/s, device->host at 1.4-2 GB/s (measured while parking). Host offload is
used for parked conversations only; it is not on the token path.

## 7. Numerical accuracy

- Codecs: exhaustive byte tests against ggml's C dequantizers; on-TPU decode equals NumPy on real rows.
- Per-layer: recurrent update isolated matches at 1e-6; full blocks differ from the CPU by 0.1-0.6%, which
  is the CPU backend's own Q8 activation quantisation.
- End to end (`results/eval-reference.json`): 10 prompt types (knowledge, math, code, instruction following,
  tool use, multi-turn, 722-token retrieval, thinking, French, raw completion) x 32 teacher-forced steps
  against the CPU oracle. FP32-KV decode, BF16-KV decode and BF16 prefill each give top-1 319/320 and mean
  KL 1.3e-5 (max 2.2e-4 to 4.5e-4). The single top-1 difference is the same step in all three paths.
- Long context: a 21K-token agent conversation retrieved 11 of 11 planted values; a 19K-token document
  query gave the same answer before and after park/restore.
- Not done: benchmark-suite scores (e.g. task accuracy against the reference at scale), and logit comparison
  beyond 32 generated tokens per prompt.

## 8. Approaches that failed or lost

- 1.6-bit packing, pre-expanded int8, u32-lane unpack, per-group MXU products in XLA (see section 3).
- Butterfly Hadamard; single default-precision Hadamard product (0.17% error); `f32->bf16->f32` splitting
  (XLA folds it away — mask mantissa bits instead); H8 (x) H128 factorisation (slower than two 1024 passes).
- Power-of-two attention buckets: multiplied compiled programs (1.1 GiB host RAM each) and caused a host OOM.
- Parallel DMA streams to beat the read rate: no effect.
- int8 MXU for prefill: per-group scales would need eight K=128 products and M x N x 8 vector work per tile.
- Loading FP16 scales directly: Mosaic cannot load f16; solved with integer decoding of the half bits.

## 9. Low-level (C/C++/Rust) assessment

The token path contains no host code: steps run inside one XLA program, kernels are Mosaic (the lowest level
the TPU exposes publicly). Host work per request is 3-4 ms (template, cached tokenization — the tokenizer is
already Rust) plus one device round trip per 16 tokens. A native server would save well under 1% of a turn.
The 24 vCPUs cannot contribute model compute (the CPU path is ~4 tok/s class); host RAM is used for parking.

## 10. Remaining bottlenecks and next steps

- Decode: ~3 ms above the read floor (GDN/conv op count, attention loop, kernel compute). A denser code
  with a byte-lane-cheap decoder would lower the floor itself; none was found.
- Prefill: MXU-bound GEMM; GDN chunk solve (a fused Pallas kernel like SGLang-JAX's is untested here);
  attention at long context (O(L) per token).
- Tail chunks are padded to 64 or 512 tokens (up to ~0.2 s of waste per request).
- Parking is synchronous (0.8-1.3 s for 19K tokens); it could run in the background or be avoided for small
  side requests with a second KV region.
- int8 KV for 128K+; host paging for 262K; multimodal input; batch > 1; grammar-constrained tool calls.
- Quality: larger benchmark-based comparison with the reference.
