#!/bin/bash
# Forward the runtime's OpenAI-compatible server to localhost:${1:-8000} over the Colab SSH bridge.
cd "$(dirname "$0")/.."
PORT=${1:-8000}
exec ssh -i .colab_ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -o LogLevel=ERROR \
  -o ProxyCommand="colab ssh --proxy-mode -s bonsai-tpu -i $PWD/.colab_ssh/id_ed25519.pub" \
  -N -L 127.0.0.1:$PORT:127.0.0.1:8000 root@colab
