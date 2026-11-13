import json
from pathlib import Path
import numpy as np
from bonsai_tpu.loader import GGUFWeights
import argparse
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=0);args=p.parse_args()
model=GGUFWeights('weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf','research/gguf-metadata.json')
w=model.layer(args.layer)
a={}
for name,v in w.items():
 if isinstance(v,dict):
  for sub,data in v.items():a[name+'::'+sub]=data
 else:a[name]=v
for width,s in model.signs.items():a['signs::'+str(width)]=s
manifest=[json.loads(l) for l in Path('results/reference-tensors-v2/manifest.jsonl').read_text().splitlines()]
for t in manifest:
 if t['name'].startswith('norm-'):continue
 raw=(Path('results/reference-tensors-v2')/t['file']).read_bytes()
 array=np.ndarray(t['shape'],dtype='<f4',buffer=raw,strides=t['strides']).transpose(3,2,1,0).copy()
 a[f'ref::{t["step"]}::{t["name"]}']=array
ids=json.loads(Path('results/reference-capital.json').read_text())
a['embeddings::0']=model.embeddings(ids['prompt_ids'])
a['embeddings::1']=model.embeddings(ids['greedy_ids'][:1])
output='results/layer-validation.npz' if args.layer==0 else f'results/layer-{args.layer}-validation.npz'
np.savez_compressed(output,**a)
print('saved',Path(output).stat().st_size,'bytes')
