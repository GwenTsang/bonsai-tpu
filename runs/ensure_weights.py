# Download the pinned PTQ1_0 GGUF onto the runtime if it is not there (uses the staged HF token when present).
import os
P='/content/weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
if os.path.exists(P) and os.path.getsize(P)==5946648928:
    print('weights present',flush=True)
else:
    from huggingface_hub import hf_hub_download
    tok=None
    if os.path.exists('/content/.bonsai-hf-token'):tok=open('/content/.bonsai-hf-token').read().strip()
    tok=tok or os.environ.get('HF_TOKEN')
    p=hf_hub_download('prism-ml/Ternary-Bonsai-2-27B-gguf','Ternary-Bonsai-2-27B-PTQ1_0.gguf',revision='839bd57023c1558945afea70a74fa83c054f35f9',token=tok,local_dir='/content/weights')
    print('downloaded',p,os.path.getsize(p),flush=True)
