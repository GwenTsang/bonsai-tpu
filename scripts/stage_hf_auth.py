"""Forward existing HF credentials to the authorized runtime without printing them."""
import os,subprocess,tempfile
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
value=os.environ.get('HF_TOKEN')
if not value:
    value=subprocess.check_output(['.venv/bin/python','-c','from huggingface_hub import get_token; print(get_token() or "", end="")'],text=True)
if not value:raise SystemExit('No HF credentials available')
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'hf-token';p.write_text(value);p.chmod(0o600)
    try:ContentsClient(state.get_session('bonsai-tpu')).upload(str(p),'/content/.bonsai-hf-token')
    except Exception as e:raise SystemExit('Credential staging failed: '+type(e).__name__) from None
print('Credential staged for authenticated model download; value not logged.')
