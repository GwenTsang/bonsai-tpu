# Bonsai TPU inference research

Ongoing development for Prism ML Ternary Bonsai 2 27B on a single TPU v5e-1.
No complete inference engine or validated model throughput is available yet.

The full mission is in [research/mission.txt](research/mission.txt).
Evidence, findings, limitations and next experiments are in [research/LOG.md](research/LOG.md).
Hardware measurements are saved in [results/hardware-baseline.json](results/hardware-baseline.json).

Run hardware experiments on the existing attached session:

```sh
colab exec -s bonsai-tpu -f scripts/hardware_baseline.py
colab download -s bonsai-tpu /content/bonsai-results/hardware-baseline.json results/hardware-baseline.json
```

Inspect the pinned model's actual GGUF metadata without downloading all weights:

```sh
python3 scripts/inspect_gguf.py
```

The reader consumes HF_TOKEN from the environment when present. Never commit credentials.

Validated building blocks now include exact NumPy/JAX PTQ1_0 and PQ2_0 decoding,
Hadamard activation transforms, a CPU reference logit/tensor dumper, and an
experimental Pallas matmul with local weight reconstruction. Full-model TPU
execution is still under development. BF16/INT8 numerical changes have not
passed end-to-end quality validation.

Local correctness tests (requires the pinned reference build):

```sh
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/real_weight_vectors.py
```

Remote experiment scripts require uploading the package tarball and vectors as
described in the research log and script paths. The Colab kernel retains imported
modules; reload changed modules or use an isolated namespace, without restarting
the TPU runtime unnecessarily.
