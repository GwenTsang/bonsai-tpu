"""Download the pinned released PTQ1_0 model, resuming through HF cache."""
import json, os
from pathlib import Path
from huggingface_hub import hf_hub_download, get_token
manifest=json.loads(Path('research/hf-manifest.json').read_text())
token=os.environ.get('HF_TOKEN') or get_token()
print('HF authentication available:',bool(token),flush=True)
p=hf_hub_download('prism-ml/Ternary-Bonsai-2-27B-gguf',
    'Ternary-Bonsai-2-27B-PTQ1_0.gguf',revision=manifest['sha'],
    local_dir='weights',token=token)
print('MODEL_READY',p,flush=True)
