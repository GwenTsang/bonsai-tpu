# Bonsai TPU — ternary Qwen 3.8 27B on one TPU v5e

A standalone JAX/Pallas inference engine for the Ternary Bonsai 2 family — ternary quantisations of
**Qwen 3.8 27B** — on a single Google TPU v5e-1 (16 GB HBM).

**Default model:**
[`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF),
file `Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PTQ1_0.gguf` (6.6 GB). With no `--model` the engine
downloads it. See [Default model](#default-model) for why this file, and what has and has not been validated
with it.

The GGUF labels its graph `general.architecture = qwen35`; that is the llama.cpp architecture name of this
hybrid Gated-DeltaNet / attention family (26.9 B weights), not a different model.

The whole model stays resident in HBM (2-bit layout for ternary matrices, one byte per weight for the
re-quantised abliterated ones; 9.4 GiB for the default file) and is evaluated by custom Pallas kernels.
It is built for agent loops: conversation state is reused across requests, tool calls are parsed, and copied
spans are verified in batches.

Measured on Colab TPU v5e-1 **with the original `prism-ml/Ternary-Bonsai-2-27B-gguf` PTQ1_0 file** (6.7 GiB
resident); these numbers have not been re-measured with the Huihui default (details, raw data and method:
[research/REPORT.md](research/REPORT.md), [research/LOG.md](research/LOG.md), `results/*.json`):

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

## Default model

The Huihui repository contains about fifty GGUF files. Only its `Ternary-Bonsai` builds belong to the family
this engine implements (Prism ML's Hadamard-folded ternary weights); the others (Q2_K ... Q8_0, IQ2/IQ3,
UD, GSQ-RCO, BF16) are conventional quantisations of 8.4-55 GB that neither match the kernels nor leave room
for a context next to them in 16 GB. Of the ternary builds, `Ternary-Bonsai-PTQ1_0` is the same container as
the original Prism release, so it is the default.

It is not byte-compatible with the Prism file. Huihui abliterated layers 22-52 and stored the edited
projections in stock llama.cpp K-quants: 31 `ffn_down` matrices as Q3_K and 23 `ssm_out` matrices as Q2_K
(3.5 B of the 26.9 B weights). The loader folds their sub-block scales into integer codes and keeps them one
byte per weight ("P8"); the GEMV/GEMM kernels read that layout with the same shift/mask and int8-MXU steps as
the ternary one. Consequences:

- Resident weights are 9.4 GiB instead of 6.7 GiB, so a 65,536-token context is expected to need about
  14.7 GiB of the 15.75 GiB HBM instead of 12 GiB, and 98K no longer fits. Use `--ctx 32768` if the default
  runs out of memory.
- Decode reads 2.6 GB more per token; at the measured ~690 GB/s HBM rate that is roughly 4 ms on top of
  12.9 ms (an estimate of ~60 instead of 77 tok/s).
- **Validation status.** On CPU: the K-quant decode is bit-exact against ggml's C dequantizers; the kernels,
  run through the Pallas interpreter, match dense products; and the full graph on the real file agrees with the
  Prism llama.cpp reference on a short prompt (decode: same top token on 3/3 steps, KL 2e-5 to 5e-5;
  batched prefill: same top token, KL 2.1e-5). **On the TPU: not run yet** — no runtime was attached when the default was changed. The
  HBM and speed figures above are estimates, and the new kernel variants have not been compiled by Mosaic.
  `scripts/run.sh runs/validate_all.py` and `runs/final_bench.py` are the first things to run.
- The model is abliterated (refusal behaviour removed); see the usage warnings on its model card.

Other models: `--model /path/to.gguf`, `$BONSAI_MODEL`, or `$BONSAI_HF_REPO` / `$BONSAI_HF_FILE` for another
Hugging Face file of this family, for example the original
`BONSAI_HF_REPO=prism-ml/Ternary-Bonsai-2-27B-gguf BONSAI_HF_FILE=Ternary-Bonsai-2-27B-PTQ1_0.gguf`.

## Layout

```
bonsai_tpu/
  gguf.py          GGUF reader (metadata, tensor directory, mmap views)
  defaults.py      default Hugging Face repository / file and download
  quantization.py  exact PTQ1_0 / PQ2_0 / Q2_K / Q3_K codecs (validated against ggml's C dequantizers)
  pack.py          on-device PTQ1_0 -> B4 (2-bit) and Q2_K / Q3_K -> P8 (8-bit) conversion
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

## Quick start: local pi harness, remote Colab TPU

Run setup on your **local Linux/macOS machine**, from this repository's root. Pi's tools read and edit
files locally; the model runs in the attached Colab session. No local JAX, weights, GPU, or OpenAI account
is needed. The connection uses authenticated `colab exec`, including streaming, with no SSH/public tunnel.

### 1. Prepare locally before allocating the TPU

You need Python 3.11+, the [Colab CLI](https://github.com/googlecolab/google-colab-cli), and Node.js
22.19+ if pi is missing. Install Node from [nodejs.org](https://nodejs.org/), or use
[pi's official installer](https://github.com/earendil-works/pi/tree/main/packages/coding-agent), which
can also install Node. Install the Colab CLI with `uv` (or `pipx install google-colab-cli`):

```sh
uv tool install google-colab-cli
python3 -m bonsai_tpu.setup_colab --prepare-only
colab sessions
```

`colab sessions` authenticates your Google account. On first use, follow its browser URL and paste the
authorization code in your own terminal. Colab authentication and Hugging Face authentication are
separate. Complete these local preparations before starting the TPU clock.

`--prepare-only` installs pi **only when it cannot find an existing installation**, into
`~/.local/share/bonsai_tpu/pi`, without sudo or a global npm upgrade. It merges the `bonsai-tpu` provider
and model into `~/.pi/agent/models.json` (or `$PI_CODING_AGENT_DIR/models.json`). Existing providers,
credentials, other models and custom sampling settings are preserved. It backs up an existing file before
changing it; an unchanged configuration is not rewritten. `auth.json` and pi's default model are untouched.
To test with an isolated profile, add `--pi-agent-dir /path/to/test-profile` to both setup commands.

### 2. Attach a v5e-1 runtime, then start with one command

Reuse your existing CLI session named `bonsai-tpu`, or request one explicitly:

```sh
colab new -s bonsai-tpu --tpu v5e1
python3 -m bonsai_tpu.setup_colab --session bonsai-tpu --hf-token hf_YOUR_READ_TOKEN
```

Setup itself **never provisions another runtime**. It checks the attached v5e-1, uploads source, installs
missing runtime dependencies (including tested JAX/TPU 0.11.2), downloads the 6.6 GB GGUF directly onto
Colab, loads the model, warms the programs, starts the local OpenAI-compatible bridge, and configures pi.
The default context is 65,536 tokens. Use `--ctx 32768` for a smaller cache, or `--port 8001` if local port
8000 is occupied. Select a download revision with `BONSAI_REVISION` on the runtime if needed.

A Hugging Face **read** token is optional for this public model. Authenticated requests have
[higher rate limits](https://huggingface.co/docs/hub/rate-limits), which can avoid anonymous download
throttling; higher bandwidth is not guaranteed. Xet high-performance transfer mode is enabled. The token
is staged as a private file, consumed on the runtime, and removed. It is not embedded in executed source,
pi config or setup logs. For a token kept out of shell history and process arguments, use `HF_TOKEN` or
`--hf-token-file` instead:

```sh
read -rs -p 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
python3 -m bonsai_tpu.setup_colab --session bonsai-tpu
unset HF_TOKEN
```

Downloads, converted weights and compilation caches are reused while the runtime survives. Repeating
the same setup reuses a healthy server. If dependencies change after JAX was imported, setup asks you
to run `colab restart-kernel -s bonsai-tpu` and rerun it; downloaded files and the compilation cache survive
a kernel restart, while in-memory state is cleared.

### 3. Use pi or test the API

Use the exact `Run:` line printed by setup. It uses an existing pi executable or the full path of the
private installation, so pi does not have to be on your PATH. If it is on PATH, the equivalent is:

```sh
pi --provider bonsai-tpu --model huihui-qwen3.8-27b-abliterated
# Short smoke test with reasoning/tools disabled:
pi --provider bonsai-tpu --model huihui-qwen3.8-27b-abliterated --thinking off --no-tools --no-session -p 'What is 2+2?'
```

For an isolated profile, prefix pi with `PI_CODING_AGENT_DIR=/path/to/test-profile`. Existing interactive
pi sessions can reload the model list with `/model`.

```sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"thinking":false,"stream":true,"max_tokens":64}'
```

The Colab-exec bridge adds a request round trip (about 0.5–1 s in a warm smoke test); throughput numbers
above measure the TPU engine, not this transport. Streamed/non-streamed responses and tool calls use
the same server API. One request runs at a time.

### A short free-Colab session

Plan around your available time, including the roughly **1h30** budget reported for your account.
Colab does **not** guarantee that duration or v5e-1 availability: [limits vary with account, demand and
usage](https://research.google.com/colaboratory/faq.html#resource-limits). A fresh runtime needs the
download plus roughly 2–3 minutes of first compilation; network and dependency installation times vary.
Prepare pi/authentication locally first, start inference once, and avoid the development benchmarks
during a short session. Free-tier rules also restrict SSH remote control; the primary setup uses Colab
CLI execution. Follow Colab's current policies; this helper does not extend quotas, prevent expiry or
automatically allocate replacement sessions.

When finished, release the runtime explicitly:

```sh
colab stop -s bonsai-tpu
```

Local bridge logs/PID files are `~/.cache/bonsai_tpu/bridge-<port>.log` and `.pid`. To stop the bridge,
send SIGTERM to the saved PID after checking it still belongs to `bonsai_tpu.colab_bridge`. Releasing
the runtime and stopping the local bridge are separate actions.

### Troubleshooting setup

- **Authentication/session missing:** run `colab sessions` in your terminal, then `colab status -s bonsai-tpu`.
  Accelerator allocation depends on your account's current quota/availability.
- **An observation timed out:** the remote job may still be running. Rerun the same setup to inspect/join
  it; do not restart solely because an observation failed.
- **Runtime package restart requested:** follow the printed `colab restart-kernel` command, then rerun
  setup. Files survive a kernel restart, but not losing the runtime VM.
- **Setup failed:** retrieve its runtime log with
  `colab download -s bonsai-tpu /content/bonsai-setup.log ./bonsai-setup.log`.
- **Invalid existing pi config:** setup stops without replacing it. Repair the JSON or use an isolated
  `--pi-agent-dir`. Backups are `models.json.bonsai-backup-<timestamp>`.
- **Development runs:** `scripts/run.sh`/`runs/load.py` stop the server while experimenting. Rerun setup
  afterward. `scripts/start_server.sh`/`scripts/tunnel.sh` are the older SSH development flow and require
  your own SSH setup and an account/runtime where SSH is permitted.

### On the runtime itself

```sh
python -m bonsai_tpu.cli chat  --ctx 32768 --thinking off     # default model is downloaded if absent
python -m bonsai_tpu.cli serve --ctx 65536 --port 8000
python -m bonsai_tpu.cli bench --model /path/to/other.gguf     # any PTQ1_0 (+ Q2_K/Q3_K) GGUF of this architecture
```

```python
from bonsai_tpu.engine import Engine
eng = Engine(n_ctx=32768)            # huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF (ternary PTQ1_0 file) by default
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

The TPU checks compare with CPU-oracle logits of the model that is loaded. `eval/run_reference.sh` produces
them for the default file in `results/ref-huihui/` and prints the three upload commands in its header
(`reference-capital.f32`, `reference-capital.json`, `eval-ref.npz`). The files directly under `results/`
(`reference-capital.*`, `eval-ref*`, `eval-reference.json`) belong to the original Prism file.

The CPU oracle is the pinned PrismML llama.cpp fork (`scripts/build_reference.sh`, revision in the log).

## Recovery after a lost runtime

Source, results and the research log are in this repository. Attach a newly authorized v5e-1 CLI session
and rerun `python3 -m bonsai_tpu.setup_colab --session bonsai-tpu` with your HF token if desired. It
re-downloads and recompiles; local pi configuration is reused. `colab run` creates/releases a separate
VM, so it is not the persistent interactive startup command.

## Limitations

- Text only. One sequence at a time.
- Prompt processing uses BF16 activations and BF16-reconstructed weights; decode uses exact ternary weights
  with 16-bit activations. Both agree with the CPU reference to KL ~1e-5 per step, but they are not bitwise
  identical to each other or to the reference (which quantises activations to 8 bits itself).
- 98K context is the largest tested (Prism file). 128K with BF16 KV does not fit next to the weights; see the
  report. With the Huihui default, 65,536 is the expected ceiling and is not yet measured.
- Each compiled program pins ~1.1 GiB of host RAM; the engine uses four.
