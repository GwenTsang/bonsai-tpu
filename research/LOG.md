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
