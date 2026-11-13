"""Default model: huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF (abliterated Qwen/Qwen3.8-27B).

That repository holds many quantisations.  This engine runs the Ternary Bonsai family, so the default file is
its `Ternary-Bonsai-PTQ1_0` build: Prism ML's ternary Qwen 3.8 27B with layers 22-52 abliterated.  In it the
edited projections (31 ffn_down, 23 ssm_out) are stored as Q3_K / Q2_K instead of PTQ1_0; the loader keeps
those in the byte-per-weight P8 layout (kernels.py).  Other files of the repository (Q4_K, IQ3, BF16, ...)
are not ternary and would not fit a 16 GB TPU in this engine; $BONSAI_HF_FILE selects another file of the
same family (PTQ1_0 / Q2_K / Q3_K matrices), $BONSAI_HF_REPO another repository.

The GGUF tags its graph as `general.architecture = qwen35` (the llama.cpp architecture name shared by this
model family); the weights are the Qwen 3.8-based release above.
"""
import os

DEFAULT_REPO = 'huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF'
DEFAULT_FILE = 'Huihui-Qwen3.8-27B-abliterated-Ternary-Bonsai-PTQ1_0.gguf'
HF_REPO = os.environ.get('BONSAI_HF_REPO') or DEFAULT_REPO
HF_FILE = os.environ.get('BONSAI_HF_FILE') or DEFAULT_FILE
# bytes of the default file, for download completeness checks; unknown for an overridden one
HF_SIZE = 6608603616 if (HF_REPO, HF_FILE) == (DEFAULT_REPO, DEFAULT_FILE) else None
MODEL_NAME = 'huihui-qwen3.8-27b-abliterated'
MODEL_OWNER = 'huihui-ai'


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
    return hf_hub_download(HF_REPO, HF_FILE, revision=os.environ.get('BONSAI_REVISION') or None,
                           token=os.environ.get('HF_TOKEN') or None, local_dir=local_dir)
