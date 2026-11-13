# Bonsai TPU — Ternary Bonsai 2 27B on one TPU v5e

A standalone JAX/Pallas inference engine for
[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) — the ternary
quantisation of **Qwen 3.8 27B** — on a single Google TPU v5e-1 (16 GB HBM). That repository (PTQ1_0 file)
is the built-in default: with no `--model` the engine downloads it.

The GGUF labels its graph `general.architecture = qwen35`; that is the llama.cpp architecture name of this
hybrid Gated-DeltaNet / attention family (26.9 B ternary weights), not a different model.

The whole model stays resident in HBM in a 2-bit layout (7.2 GiB) and is evaluated by custom Pallas kernels.
It is built for agent loops: conversation state is reused across requests, tool calls are parsed, and copied
spans are verified in batches.

Measured on Colab TPU v5e-1 (details, raw data and method: [research/REPORT.md](research/REPORT.md),
[research/LOG.md](research/LOG.md), `results/*.json`):

| | |
|---|---|
| Decode, short context | 77.6 tok/s (CPU reference on the dev host: 4 tok/s) |
| Decode at 32K / 64K context | 59 / 50 tok/s |
| Decode while copying context (file edits) | up to 571 tok/s, identical tokens |
| Prefill | 2110-2170 tok/s up to 8K prompts, 1800 at 32K, 1280 averaged over a 98K prompt |
| Context tested | 98,304 tokens (BF16 KV, 14.0 GiB HBM); 65,536 is the serving default (12 GiB) |
| Load / warm start | 13 s load; programs load from the XLA cache in ~20 s (2-3 min first time) |
| Agreement with the Prism llama.cpp CPU reference | top-1 319/320 teacher-forced steps over 10 prompt types; mean KL 1.3e-5 |
| Interleaved conversations | 19K-token session parked to host RAM in ~1 s, restored in 0.3-0.6 s |

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
python -m bonsai_tpu.cli chat  --ctx 32768 --thinking off     # default model is downloaded if absent
python -m bonsai_tpu.cli serve --ctx 65536 --port 8000
python -m bonsai_tpu.cli bench --model /path/to/other.gguf     # any PTQ1_0 GGUF of this architecture
```

```python
from bonsai_tpu.engine import Engine
eng = Engine(n_ctx=32768)            # prism-ml/Ternary-Bonsai-2-27B-gguf by default
eng.warmup()
for piece in eng.chat([{'role': 'user', 'content': 'Hi'}], thinking=False, temperature=0.7, max_new_tokens=256):
    print(piece, end='')
```

Dependencies on the runtime: `jax`/`jaxlib` with TPU support (0.11.2 tested), `numpy`, `tokenizers`, `jinja2`,
`huggingface_hub` (download only). Local test dependencies: `research/local-dependencies.txt`.

## Server

`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`.

- `tools` are rendered by the model's chat template; `<tool_call>` blocks in the output become `tool_calls`.
- Thinking is the model's own switch and is on by default, as in its chat template. Per request:
  `"thinking": false` (also `enable_thinking`, `chat_template_kwargs.enable_thinking`,
  `{"thinking": {"type": "disabled"}}`), and `reasoning_effort` `low | medium | high` (mapped to the
  template's low / medium / xhigh). Reasoning is returned as `reasoning_content`.
  Server default: `BONSAI_THINK=0 scripts/start_server.sh` or `cli serve --thinking off`.
- `usage.prompt_tokens_details.cached_tokens` and `timings` report how much of the prompt was served from state.
- One request at a time (single-user engine).

## How requests reuse work

- The KV cache is position-addressed; the recurrent (GDN, convolution) state is checkpointed at the prompt
  end, every 4096 tokens, after the system prompt/tool schemas and at the last message. A request resumes from
  the nearest checkpoint before the first token that differs from the previous request.
- Per-message tokenization is cached.
- Prompt suffixes use a measured, context-dependent schedule of 64- and 512-token chunks; a one-token
  tail reuses the decode program. This reduces padding and time to first token for short updates; see
  [prefill scheduling measurements](research/PREFILL_SCHEDULING.md).
- When the last six generated tokens also occur earlier in the context, the continuation found there is
  verified 32 positions at a time.
- A conversation about to be displaced by a different one is parked in host RAM (KV rows, recurrent state,
  checkpoints) and uploaded again when it returns.

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
- 98K context is the largest tested. 128K with BF16 KV does not fit next to the weights; see the report.
- Each compiled program pins ~1.1 GiB of host RAM; the engine uses four.
