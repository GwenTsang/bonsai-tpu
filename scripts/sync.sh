#!/bin/bash
# Upload the current package to the runtime and unpack it; kernel modules are purged by run.sh preamble.
set -e
cd "$(dirname "$0")/.."
tar -czf /tmp/bonsai-pkg.tar.gz --exclude=__pycache__ bonsai_tpu bench runs 2>/dev/null
colab upload -s bonsai-tpu /tmp/bonsai-pkg.tar.gz /content/bonsai-pkg.tar.gz >/dev/null
