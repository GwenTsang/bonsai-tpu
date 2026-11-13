"""Run with the installed Colab CLI Python; bounded upload, no credential logs."""
import argparse,hashlib,json,tempfile
from pathlib import Path
from colab_cli.common import state
from colab_cli.contents import ContentsClient
p=argparse.ArgumentParser();p.add_argument('file');p.add_argument('--session',default='bonsai-tpu');args=p.parse_args()
client=ContentsClient(state.get_session(args.session))
path=Path(args.file);parts=[];sha=hashlib.sha256()
with path.open('rb') as f,tempfile.TemporaryDirectory() as tmp:
    while data:=f.read(16*1024**2):
        sha.update(data);name=f'{path.name}.part{len(parts):04d}';local=Path(tmp)/name;local.write_bytes(data)
        try:client.upload(str(local),'/content/'+name)
        except Exception as e:raise SystemExit(f'Upload failed: {type(e).__name__}; no URL logged') from None
        parts.append(name);print('Uploaded',name,flush=True)
    manifest={'name':path.name,'sha256':sha.hexdigest(),'parts':parts}
    local=Path(tmp)/'manifest.json';local.write_text(json.dumps(manifest))
    try:client.upload(str(local),'/content/'+path.name+'.manifest.json')
    except Exception as e:raise SystemExit(f'Manifest upload failed: {type(e).__name__}') from None
print('CHUNKS_UPLOADED',flush=True)
