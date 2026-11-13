#!/bin/bash
# CPU oracle: greedy 32 steps with full logits per step for each evaluation prompt (pinned Prism llama.cpp).
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import json, subprocess, os
p = json.load(open('eval/prompts.json'))
for name, c in p.items():
    out = 'results/eval-ref/' + name
    if os.path.exists(out + '.json'):
        continue
    r = subprocess.run(['reference/dump_logits', 'weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf', c['text'], out, '32'],
                       capture_output=True, text=True)
    meta = json.load(open(out + '.json'))
    assert meta['prompt_ids'] == c['ids'], 'tokenization mismatch for ' + name
    print(name, 'ok', len(meta['prompt_ids']), flush=True)
PY
