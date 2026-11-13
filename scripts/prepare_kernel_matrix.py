import json
import numpy as np
from bonsai_tpu.quantization import split_blocks
from bonsai_tpu.hadamard import sign_table
m=json.load(open('research/gguf-metadata.json'))
t=next(t for t in m['tensors'] if t['name']=='blk.0.ffn_gate.weight')
k,n=t['shape']
with open('weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf','rb') as f:
 f.seek(m['data_offset']+t['offset']);raw=f.read(k*n//128*28)
p,s=split_blocks(raw,143)
np.savez('results/kernel-matrix.npz',payload=p.reshape(n,k//128,26),scales=s.reshape(n,k//128),signs=sign_table(m['metadata'])[k])
print('Prepared real matrix',n,k)
