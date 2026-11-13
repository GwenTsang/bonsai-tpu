"""Inspect GGUF metadata and tensor directory with a bounded HTTP range read."""
import json, os, struct, urllib.request
from pathlib import Path
REPO='prism-ml/Ternary-Bonsai-2-27B-gguf'
revision=json.loads(Path('research/hf-manifest.json').read_text())['sha']
url=f'https://huggingface.co/{REPO}/resolve/{revision}/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
headers={'Range':'bytes=0-16777215'}
if os.environ.get('HF_TOKEN'): headers['Authorization']='Bearer '+os.environ['HF_TOKEN']
req=urllib.request.Request(url,headers=headers)
with urllib.request.urlopen(req,timeout=60) as r:
    if r.status!=206: raise RuntimeError('Server did not honor bounded range request')
    data=r.read(16777216)
pos=0
def read(fmt):
    global pos
    fmt='<'+fmt
    out=struct.unpack_from(fmt,data,pos);pos+=struct.calcsize(fmt)
    return out[0] if len(out)==1 else out
def string():
    global pos
    n=read('Q');v=data[pos:pos+n].decode();pos+=n;return v
formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def value(t):
    if t==8:return string()
    if t==9:
        subtype,n=read('IQ');return [value(subtype) for _ in range(n)]
    return read(formats[t])
assert data[:4]==b'GGUF'
pos=4;version,nt,nkv=read('IQQ')
meta={}
for _ in range(nkv):
    key=string();meta[key]=value(read('I'))
tensors=[]
for _ in range(nt):
    name=string();ndim=read('I');shape=[read('Q') for _ in range(ndim)]
    typ,offset=read('IQ');tensors.append(dict(name=name,shape=shape,type=typ,offset=offset))
alignment=meta.get('general.alignment',32)
out=dict(repository=REPO,revision=revision,version=version,metadata=meta,tensors=tensors,data_offset=(pos+alignment-1)//alignment*alignment)
Path('research/gguf-metadata.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:(f'<array length {len(v)}>' if isinstance(v,list) and len(v)>16 else v) for k,v in meta.items() if not k.startswith('tokenizer.')},indent=2))
print('Tensor count:',len(tensors),'Data offset:',out['data_offset'])
