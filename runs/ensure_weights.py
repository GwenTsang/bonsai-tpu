# Download the default GGUF onto the runtime if it is not there (uses the staged HF token when present).
import os
from bonsai_tpu import defaults
P='/content/weights/'+defaults.HF_FILE
if os.path.exists(P) and (defaults.HF_SIZE is None or os.path.getsize(P)==defaults.HF_SIZE):
    print('weights present',flush=True)
else:
    from huggingface_hub import hf_hub_download
    tok=None
    if os.path.exists('/content/.bonsai-hf-token'):tok=open('/content/.bonsai-hf-token').read().strip()
    tok=tok or os.environ.get('HF_TOKEN')
    p=hf_hub_download(defaults.HF_REPO,defaults.HF_FILE,revision=os.environ.get('BONSAI_REVISION') or None,token=tok,local_dir='/content/weights')
    print('downloaded',p,os.path.getsize(p),flush=True)
