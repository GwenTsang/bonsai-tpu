"""Initial setup on an already attached TPU. Executed by setup_colab's background job."""
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_requirements = []
try:
    _jax_version = importlib.metadata.version('jax')
except importlib.metadata.PackageNotFoundError:
    _jax_version = None
_jax_change = _jax_version != '0.11.2' or importlib.util.find_spec('libtpu') is None
if _jax_change:
    _requirements.append('jax[tpu]==0.11.2')
for _module, _package in [('tokenizers', 'tokenizers'), ('jinja2', 'jinja2'),
                          ('huggingface_hub', 'huggingface_hub>=0.32'), ('hf_xet', 'hf_xet')]:
    if importlib.util.find_spec(_module) is None:
        _requirements.append(_package)
if _requirements:
    _BONSAI_SETUP['phase'] = 'installing runtime dependencies'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *_requirements],
                   stdout=_BONSAI_LOG, stderr=_BONSAI_LOG, check=True)
    if _jax_change and 'jax' in sys.modules:
        _BONSAI_SETUP['phase'] = 'restart_required'
        raise RuntimeError('Dependencies installed. Restart the Colab kernel and rerun setup.')

_BONSAI_SETUP['phase'] = 'downloading/loading model'
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
_token_file = Path('/content/.bonsai-hf-token')
if _token_file.exists():
    _token_file.chmod(0o600)
    os.environ['HF_TOKEN'] = _token_file.read_text().strip()
    _token_file.unlink()
try:
    exec(Path('/content/runs/load.py').read_text(), globals())
finally:
    os.environ.pop('HF_TOKEN', None)

import jax
if jax.devices()[0].platform != 'tpu':
    raise RuntimeError('This session is not a TPU runtime')
from bonsai_tpu.engine import Engine
from bonsai_tpu.server import serve
import threading
import socketserver

E = Engine(n_ctx=_BONSAI_SETUP['ctx'], preloaded=(G, CFG, PARAMS))
_BONSAI_SETUP['phase'] = 'warming inference programs'
_BONSAI_SETUP['warmup_s'] = E.warmup()
socketserver.TCPServer.allow_reuse_address = True
HTTPD = serve(E, '127.0.0.1', 8000)
threading.Thread(target=HTTPD.serve_forever, daemon=True).start()
_BONSAI_SETUP['phase'] = 'ready'
print('READY: model loaded and programs warmed', flush=True)
