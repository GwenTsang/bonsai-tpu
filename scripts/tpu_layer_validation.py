"""Validate first recurrent block, including prefill -> decode state continuity."""
import tarfile,sys,importlib,json,time
with tarfile.open('/content/bonsai-kernels.tar.gz') as f:f.extractall('/content',filter='data')
sys.path.insert(0,'/content')
import numpy as np
import jax
import jax.numpy as jnp
import bonsai_tpu.layers
importlib.reload(bonsai_tpu.layers)
from bonsai_tpu.layers import recurrent_block,delta_net,causal_conv,l2_norm,rms_norm

def main():
    from pathlib import Path
    import hashlib,shutil
    manifest=json.loads(Path('/content/layer-validation.npz.manifest.json').read_text())
    with open('/content/layer-validation.npz','wb') as dest:
        for part in manifest['parts']:
            with open('/content/'+part,'rb') as source:shutil.copyfileobj(source,dest)
    digest=hashlib.file_digest(open('/content/layer-validation.npz','rb'),'sha256').hexdigest()
    assert digest==manifest['sha256'], 'Upload checksum mismatch'
    a=np.load('/content/layer-validation.npz')
    w={};signs={}
    for name in a.files:
        if name.startswith(('ref::','embeddings::')):continue
        if name.startswith('signs::'):signs[int(name.split('::')[1])]=jax.device_put(a[name]);continue
        if '::' in name:
            key,sub=name.split('::');w.setdefault(key,{})[sub]=jax.device_put(a[name])
        else:w[name]=jax.device_put(a[name])
    records=[]
    def compare(name,got,ref,stage,step):
        got=np.asarray(got);ref=np.asarray(ref).reshape(got.shape)
        d=got-ref;den=float(np.linalg.norm(ref))
        row=dict(name=name,stage=stage,step=step,shape=list(got.shape),max_abs=float(np.max(np.abs(d))),
                 relative_l2=float(np.linalg.norm(d)/max(den,1e-30)),reference_rms=float(np.sqrt(np.mean(ref*ref))))
        records.append(row);print(json.dumps(row),flush=True)
        if not np.isfinite(got).all():raise ValueError('Nonfinite output')
        return row
    def ref(step,name):return a[f'ref::{step}::{name}']
    conv_state=jnp.zeros((3,10240),jnp.float32);state=jnp.zeros((48,128,128),jnp.float32)
    fn=jax.jit(recurrent_block)
    for step in (0,1):
        x=jax.device_put(a[f'embeddings::{step}'])
        compare('embedding',x,ref(step,'model_input_embed'),'end_to_end',step)
        start=time.perf_counter()
        out,conv_state,state,trace=fn(x,w,signs,conv_state,state)
        out.block_until_ready();print('BLOCK_COMPLETE',step,time.perf_counter()-start,flush=True)
        for name,value in trace.items():
            key=f'ref::{step}::{name}-0'
            if key in a:compare(name,value,a[key],'end_to_end',step)
        # Isolate recurrent math from CPU-vs-TPU projection differences.
        q=ref(step,'q_conv_predelta-0').reshape(-1,16,128)
        k=ref(step,'k_conv_predelta-0').reshape(-1,16,128)
        v=ref(step,'v_conv_predelta-0').reshape(-1,48,128)
        stkey='state_predelta-0' if f'ref::{step}::state_predelta-0' in a else 'state_cache_view-0'
        st=ref(step,stkey).reshape(48,128,128)
        alpha=ref(step,'alpha-0').reshape(-1,48);beta=ref(step,'beta-0').reshape(-1,48)
        y,new=jax.jit(delta_net)(q,k,v,alpha,beta,w['ssm_dt.bias'],w['ssm_a'],st)
        compare('attn_output',y,ref(step,'attn_output-0'),'isolated',step)
        if f'ref::{step}::new_state-0' in a:compare('new_state',new,ref(step,'new_state-0'),'isolated',step)
        with open('/content/bonsai-results/layer-validation.json','w') as f:json.dump(records,f,indent=2)
    print('LAYER_VALIDATION_COMPLETE',flush=True)
main()
