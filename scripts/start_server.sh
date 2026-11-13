#!/bin/bash
# Bring the engine up on the existing Colab TPU session and expose it on localhost:8000.
#   scripts/start_server.sh [n_ctx]        (default 65536)
# Never provisions a runtime: it only attaches to the session alias `bonsai-tpu`.
set -e
cd "$(dirname "$0")/.."
colab status 2>&1 | grep -q 'bonsai-tpu' || { echo "no attached session named bonsai-tpu (see README: Recovery)"; exit 1; }
# forward Hugging Face credentials (HF_TOKEN or the local HF cache) without printing them
/home/gwen/.local/share/uv/tools/google-colab-cli/bin/python scripts/stage_hf_auth.py >/dev/null 2>&1 || echo "note: no HF credentials staged"
RUN_TIMEOUT=1800 RUN_TAIL=1 scripts/run.sh runs/ensure_weights.py
RUN_ENV="BONSAI_NCTX=${1:-65536} BONSAI_THINK=${BONSAI_THINK:-0}" RUN_TIMEOUT=1800 RUN_TAIL=2 scripts/run.sh runs/serve.py
if ! curl -s -m 5 http://127.0.0.1:8000/health >/dev/null; then
  nohup scripts/tunnel.sh 8000 > /tmp/bonsai-tunnel.log 2>&1 &
  for i in $(seq 1 20); do sleep 1; curl -s -m 3 http://127.0.0.1:8000/health >/dev/null && break; done
fi
curl -s -m 5 http://127.0.0.1:8000/v1/models && echo && echo "ready: http://127.0.0.1:8000/v1"
