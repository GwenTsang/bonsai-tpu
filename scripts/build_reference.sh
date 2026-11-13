#!/usr/bin/env bash
set -euo pipefail
.venv/bin/cmake -S upstream/prism-llama -B upstream/prism-llama/build -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$(pwd)/.venv/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
.venv/bin/cmake --build upstream/prism-llama/build --target llama-cli llama-completion -j 8
g++ -std=c++17 -O2 reference/dump_logits.cpp -I upstream/prism-llama/include \
  -I upstream/prism-llama/ggml/include -L upstream/prism-llama/build/bin \
  -Wl,-rpath,"$(pwd)/upstream/prism-llama/build/bin" -lllama -lggml -lggml-base -o reference/dump_logits
