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
