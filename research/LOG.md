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
