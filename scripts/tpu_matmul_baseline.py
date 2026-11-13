"""Full-size real projection: baseline unpack + MXU vs direct reduction.

XLA may fuse decoding or materialize a layer depending on workload. Inspect
compiler memory statistics. INT8 activation quantization is experimental.
"""
import sys,json,time,statistics,gc
sys.path.insert(0,'/content')
import numpy as np
import jax
import jax.numpy as jnp
from bonsai_tpu.quantization import decode_codes
from bonsai_tpu.hadamard import activation_transform

def main():
    a=np.load('/content/kernel-matrix.npz');device=jax.devices('tpu')[0]
    p=jax.device_put(a['payload']);s=jax.device_put(a['scales']);signs=jax.device_put(a['signs'])
    n,groups,_=p.shape;k=groups*128
    def kernel(x,p,s,signs,mode):
        x=activation_transform(x,signs,xp=jnp)
        codes=decode_codes(p,143,jnp)
        if mode=='int8_grouped':
            x=x.reshape(x.shape[0],groups,128)
            scale=jnp.maximum(jnp.max(jnp.abs(x),axis=-1)/127,1e-30)
            q=jnp.clip(jnp.rint(x/scale[...,None]),-127,127).astype(jnp.int8)
            acc=jnp.einsum('mgk,ngk->mgn',q,codes,preferred_element_type=jnp.int32)
            return jnp.sum(acc.astype(jnp.float32)*scale[...,None]*s.T[None,...],axis=1)
        w=(codes.astype(jnp.float32)*s[...,None]).reshape(n,k)
        if mode=='direct_f32':
            return jnp.sum(w*x[0],axis=-1)[None,:]
        if mode=='bf16':
            return jnp.matmul(x.astype(jnp.bfloat16),w.T.astype(jnp.bfloat16),preferred_element_type=jnp.float32)
        return jnp.matmul(x,w.T,precision=jax.lax.Precision.HIGHEST)
    records=[]
    rng=np.random.default_rng(47)
    for m in (1,32,128):
        x=jax.device_put(rng.normal(size=(m,k)).astype(np.float32))
        reference=None
        for mode in ('f32_highest','bf16','int8_grouped','direct_f32'):
            if mode=='direct_f32' and m!=1:continue
            fn=jax.jit(lambda x,p,s,signs:kernel(x,p,s,signs,mode))
            start=time.perf_counter();compiled=fn.lower(x,p,s,signs).compile();compile_s=time.perf_counter()-start
            start=time.perf_counter();y=compiled(x,p,s,signs);y.block_until_ready();first=time.perf_counter()-start
            samples=[]
            for _ in range(9):
                start=time.perf_counter();y=compiled(x,p,s,signs);y.block_until_ready();samples.append(time.perf_counter()-start)
            host=np.asarray(y)
            if reference is None:reference=host.copy()
            delta=host-reference
            row=dict(mode=mode,m=m,n=n,k=k,compile_seconds=compile_s,first_seconds=first,
                     seconds=samples,median_seconds=statistics.median(samples),
                     max_abs_error=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
                     memory_analysis=str(compiled.memory_analysis()),device_memory=device.memory_stats())
            records.append(row);print(json.dumps(row),flush=True)
            with open('/content/bonsai-results/matmul-baseline.json','w') as f:json.dump(records,f,indent=2)
            del fn,compiled,y;gc.collect();jax.clear_caches()
    print('MATMUL_BASELINE_COMPLETE',flush=True)
main()
