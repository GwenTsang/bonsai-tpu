# Bonsai TPU — Ternary Bonsai 2 27B on one TPU v5e

A standalone JAX/Pallas inference engine for `prism-ml/Ternary-Bonsai-2-27B-gguf` (PTQ1_0 release, qwen35
hybrid Gated-DeltaNet / attention architecture, 26.9 B ternary weights) on a single Google TPU v5e-1 (16 GB HBM).

The whole model stays resident in HBM in a 2-bit layout (7.2 GiB) and is evaluated by custom Pallas kernels.
It is built for agent loops: conversation state is reused across requests, tool calls are parsed, and copied
spans are verified in batches.

Measured on Colab TPU v5e-1 (details, raw data and method: [research/REPORT.md](research/REPORT.md),
[research/LOG.md](research/LOG.md), `results/*.json`):

| | |
|---|---|
| Decode, short context | 70-78 tok/s (CPU reference on the dev host: 4 tok/s) |
| Decode at 32K / 64K context | 59 / 50 tok/s |
| Decode while copying context (file edits) | up to 571 tok/s, identical tokens |
| Prefill | 1700-1850 tok/s (1183 tok/s averaged over a 64K prompt) |
| Context tested | 65,536 tokens (BF16 KV), HBM in use 11.4-12.4 GiB |
| Load / warm start | 13 s load; programs load from the XLA cache in ~20 s (2-3 min first time) |
| Agreement with the Prism llama.cpp CPU reference | same greedy tokens; KL ~1e-5 per step |

## Layout

```
bonsai_tpu/
  gguf.py          GGUF reader (metadata, tensor directory, mmap views)
  quantization.py  exact PTQ1_0 / PQ2_0 codecs (validated against ggml's C dequantizers)
  pack.py          on-device PTQ1_0 -> B4 (2-bit) conversion
  kernels.py       Pallas kernels: int8 block-diagonal GEMV (decode), BF16 GEMM (prefill)
  hadamard.py      Prism Hadamard activation transforms (MXU formulations)
  model.py         qwen35 graph: decode step, batched prefill, chunk-parallel GDN, streaming attention
  engine.py        scheduling, sampling, checkpoints/prefix reuse, speculative lookup decoding
  server.py        OpenAI-compatible HTTP server (tools, reasoning, streaming)
  cli.py           generate / chat / serve / bench
  tokenizer.py     tokenizer + chat template built from the GGUF metadata
runs/              experiment and service scripts executed inside the TPU kernel
bench/             kernel micro-benchmarks and the client-side agent-loop benchmark
eval/              reference-comparison prompt set
reference/         CPU oracle tools built against the pinned Prism llama.cpp
tests/             CPU tests (pytest)
```

## Quick start (existing Colab TPU session)

The Colab CLI must already be attached to a TPU v5e-1 session under the alias `bonsai-tpu`.
Nothing here provisions or stops a runtime.

```sh
scripts/start_server.sh 65536     # downloads weights on the runtime if needed, loads, warms up, serves, tunnels
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

`HF_TOKEN` (or the local Hugging Face login) is forwarded to the runtime for the download; it is never logged.

Agent harness ([pi](https://pi.dev)):

```sh
PI_CODING_AGENT_DIR=$PWD/pi-agent pi --provider bonsai-tpu --model ternary-bonsai-2-27b
```

`pi-agent/models.json` declares the endpoint; copy its provider block into `~/.pi/agent/models.json` to make it
available everywhere.

### On the runtime itself

```sh
python -m bonsai_tpu.cli chat  --model /content/weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf --ctx 32768
python -m bonsai_tpu.cli serve --model ... --ctx 65536 --port 8000
python -m bonsai_tpu.cli bench --model ...
```

```python
from bonsai_tpu.engine import Engine
eng = Engine('/content/weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf', n_ctx=32768)
eng.warmup()
for piece in eng.chat([{'role': 'user', 'content': 'Hi'}], temperature=0.7, max_new_tokens=256):
    print(piece, end='')
```

Dependencies on the runtime: `jax`/`jaxlib` with TPU support (0.11.2 tested), `numpy`, `tokenizers`, `jinja2`,
`huggingface_hub` (download only). Local test dependencies: `research/local-dependencies.txt`.

## Server

`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`.

- `tools` are rendered by the model's chat template; `<tool_call>` blocks in the output become `tool_calls`.
- `enable_thinking` (or `reasoning_effort`) turns the reasoning block on; it is returned as `reasoning_content`.
- `usage.prompt_tokens_details.cached_tokens` and `timings` report how much of the prompt was served from state.
- One request at a time (single-user engine).

## How requests reuse work

- The KV cache is position-addressed; the recurrent (GDN, convolution) state is checkpointed at the prompt
  end, every 4096 tokens, after the system prompt/tool schemas and at the last message. A request resumes from
  the nearest checkpoint before the first token that differs from the previous request.
- Per-message tokenization is cached.
- When the last six generated tokens also occur earlier in the context, the continuation found there is
  verified 32 positions at a time.

## Tests

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q        # CPU: codecs, packing, kernel algebra, GDN, ...
scripts/run.sh runs/validate_all.py                                  # TPU: logits vs CPU reference
scripts/run.sh runs/eval_reference.py                                # TPU: representative prompt set
```

The CPU oracle is the pinned PrismML llama.cpp fork (`scripts/build_reference.sh`, revision in the log).

## Recovery after a lost runtime

Source, results and the research log are in this repository; nothing on the runtime is needed.

1. Attach the Colab CLI to an authorized TPU v5e-1 session and alias it `bonsai-tpu` (`colab sessions`,
   `colab status`). Do not use `colab run`; it provisions a new VM.
2. `scripts/start_server.sh` — re-downloads the GGUF onto the runtime (5.9 GB), converts it on the TPU
   (13 s), compiles the four programs (2-3 min cold) and opens the tunnel.
3. Development runs: `scripts/run.sh runs/<script>.py` executes a script in the persistent kernel (the model
   stays loaded between runs) and polls its log.

## Limitations

- Text only. One sequence at a time.
- Prompt processing uses BF16 activations and BF16-reconstructed weights; decode uses exact ternary weights
  with 16-bit activations. Both agree with the CPU reference to KL ~1e-5 per step, but they are not bitwise
  identical to each other or to the reference (which quantises activations to 8 bits itself).
- 64K context is the largest tested. 128K with BF16 KV does not fit next to the weights; see the report.
- Each compiled program pins ~1.1 GiB of host RAM; the engine uses four.
