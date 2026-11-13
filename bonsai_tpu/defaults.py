"""Default model: Prism ML Ternary Bonsai 2 27B (ternary quantisation of Qwen/Qwen3.8-27B).

The GGUF tags its graph as `general.architecture = qwen35` (the llama.cpp architecture name shared by this
model family); the weights are the Qwen 3.8-based release below.
"""
import os

HF_REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
HF_FILE = 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
HF_REVISION = '839bd57023c1558945afea70a74fa83c054f35f9'     # validated revision; set BONSAI_REVISION=main to track
MODEL_NAME = 'ternary-bonsai-2-27b'


def resolve_model(path=None, log=None):
    """Return a local GGUF path: the given one, $BONSAI_MODEL, or the default release (downloaded with
    HF_TOKEN / the local Hugging Face login into $BONSAI_WEIGHTS_DIR, default ~/.cache/bonsai_tpu/weights)."""
    path = path or os.environ.get('BONSAI_MODEL')
    if path:
        return path
    local_dir = os.path.expanduser(os.environ.get('BONSAI_WEIGHTS_DIR', '~/.cache/bonsai_tpu/weights'))
    for cand in (os.path.join(local_dir, HF_FILE), os.path.join('/content/weights', HF_FILE)):
        if os.path.exists(cand):
            return cand
    from huggingface_hub import hf_hub_download
    if log:
        log('downloading %s/%s ...' % (HF_REPO, HF_FILE))
    return hf_hub_download(HF_REPO, HF_FILE, revision=os.environ.get('BONSAI_REVISION', HF_REVISION),
                           token=os.environ.get('HF_TOKEN') or None, local_dir=local_dir)
