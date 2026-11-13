#!/bin/bash
# CPU oracle for the default model (pinned Prism llama.cpp): the 8-step "capital" smoke reference and greedy
# 32 steps with full logits per step for each evaluation prompt, packed into eval-ref.npz.
#   eval/run_reference.sh [model.gguf] [out-dir]      defaults: weights/<default file>, results/ref-huihui
# Upload for the TPU checks (runs/validate_all.py, runs/eval_reference.py):
#   colab upload -s bonsai-tpu <out>/reference-capital.f32  /content/reference-capital.f32
#   colab upload -s bonsai-tpu <out>/reference-capital.json /content/reference-capital.json
#   python scripts/upload_chunks.py <out>/eval-ref.npz
cd "$(dirname "$0")/.."
MODEL="$1" OUT="${2:-results/ref-huihui}" PYTHONPATH=. .venv/bin/python - <<'PY'
import json, subprocess, os
import numpy as np
from bonsai_tpu import defaults
model = os.environ['MODEL'] or 'weights/' + defaults.DEFAULT_FILE
out_dir = os.environ['OUT']
os.makedirs(out_dir + '/eval-ref', exist_ok=True)

def oracle(text, out, steps):
    if not os.path.exists(out + '.json'):
        r = subprocess.run(['reference/dump_logits', model, text, out, str(steps)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit('dump_logits failed (%d) for %s\n%s' % (r.returncode, out, r.stderr[-2000:]))
    return json.load(open(out + '.json'))

oracle('The capital of France is', out_dir + '/reference-capital', 8)
p = json.load(open('eval/prompts.json'))
arrays = {}
for name, c in p.items():
    out = out_dir + '/eval-ref/' + name
    meta = oracle(c['text'], out, 32)
    assert meta['prompt_ids'] == c['ids'], 'tokenization mismatch for ' + name
    arrays[name + '::prompt'] = np.asarray(meta['prompt_ids'], np.int32)
    arrays[name + '::greedy'] = np.asarray(meta['greedy_ids'], np.int32)
    arrays[name + '::logits'] = np.fromfile(out + '.f32', '<f4').reshape(32, -1).astype(np.float16)   # as before: halves the upload
    print(name, 'ok', len(meta['prompt_ids']), flush=True)
np.savez(out_dir + '/eval-ref.npz', **arrays)
print('REFERENCE_READY', out_dir, model, flush=True)
PY
