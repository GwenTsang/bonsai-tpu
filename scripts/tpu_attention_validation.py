import tarfile,sys,importlib,json,time,hashlib,shutil
from pathlib import Path
with tarfile.open('/content/bonsai-kernels.tar.gz') as f:f.extractall('/content',filter='data')
sys.path.insert(0,'/content')
import numpy as np
import jax
import jax.numpy as jnp
import bonsai_tpu.layers
importlib.reload(bonsai_tpu.layers)
from bonsai_tpu.layers import full_attention_block,rope_text

def main():
    name='layer-3-validation.npz'
    manifest=json.loads(Path('/content/'+name+'.manifest.json').read_text())
    with open('/content/'+name,'wb') as dest:
        for part in manifest['parts']:
            with open('/content/'+part,'rb') as source:shutil.copyfileobj(source,dest)
    with open('/content/'+name,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256']
    a=np.load('/content/'+name);w={};signs={};records=[]
    for name in a.files:
        if name.startswith(('ref::','embeddings::')):continue
        if name.startswith('signs::'):signs[int(name.split('::')[1])]=jax.device_put(a[name]);continue
        if '::' in name:
            key,sub=name.split('::');w.setdefault(key,{})[sub]=jax.device_put(a[name])
        else:w[name]=jax.device_put(a[name])
    def ref(step,name):return a[f'ref::{step}::{name}-3']
    def compare(name,got,reference,step,stage):
        got=np.asarray(got);reference=reference.reshape(got.shape);delta=got-reference
        row=dict(name=name,step=step,stage=stage,shape=list(got.shape),max_abs=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30)))
        records.append(row);print(json.dumps(row),flush=True)
        assert np.isfinite(got).all()
    k=jnp.zeros((0,4,256),jnp.float16);v=k
    fn=jax.jit(full_attention_block)
    for step in (0,1):
        # The incoming residual is reconstructed from two independently captured nodes.
        x=(ref(step,'attn_residual')-ref(step,'attn_output')).reshape(-1,5120)
        out,k,v,trace=fn(jax.device_put(x),w,signs,k,v);out.block_until_ready()
        for name,value in trace.items():
            key=f'ref::{step}::{name}-3'
            if key in a:compare(name,value,a[key],step,'block')
        for prefix in ('Q','K'):
            y=ref(step,prefix+'cur_normed').reshape(-1,24 if prefix=='Q' else 4,256)
            positions=jnp.arange(y.shape[0])+(0 if step==0 else 5)
            target=prefix+'cur'
            if f'ref::{step}::{target}-3' not in a:target='k_cache_in'
            compare(prefix+'rope',jax.jit(rope_text)(y,positions),ref(step,target),step,'isolated')
        with open('/content/bonsai-results/attention-validation.json','w') as f:json.dump(records,f,indent=2)
    print('ATTENTION_VALIDATION_COMPLETE',flush=True)
main()
