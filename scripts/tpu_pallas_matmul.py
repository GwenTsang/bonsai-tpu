import tarfile,sys,json,time,statistics,traceback
with tarfile.open('/content/bonsai-kernels.tar.gz') as f:f.extractall('/content',filter='data')
sys.path.insert(0,'/content')
import importlib
import bonsai_tpu.pallas_matmul
importlib.reload(bonsai_tpu.pallas_matmul)
import numpy as np
import jax
import jax.numpy as jnp
from bonsai_tpu.pallas_matmul import packed_matmul,pack_words
from bonsai_tpu.quantization import decode_codes
from bonsai_tpu.hadamard import activation_transform

def main():
    a=np.load('/content/kernel-matrix.npz');n,g,_=a['payload'].shape;k=g*128
    codes=decode_codes(a['payload'],143).reshape(n,k)
    words=jax.device_put(pack_words(codes));scales=jax.device_put(a['scales'].T.copy());signs=jax.device_put(a['signs'])
    dense=jax.device_put((codes.astype(np.float32)*np.repeat(a['scales'],128,axis=1)).T)
    records=[]
    rng=np.random.default_rng(47)
    for m in (1,32,128):
        x=jax.device_put(rng.normal(size=(m,k)).astype(np.float32))
        ref_fn=jax.jit(lambda x,w,s:jnp.matmul(activation_transform(x,s,xp=jnp),w,precision=jax.lax.Precision.HIGHEST))
        expected=np.asarray(ref_fn(x,dense,signs))
        for bm,bk,bn in ((32,1024,256),(32,1024,512),(128,1024,512),(128,1024,1024)):
            row=dict(m=m,k=k,n=n,bm=bm,bk=bk,bn=bn)
            try:
                fn=jax.jit(lambda x,w,s,signs:packed_matmul(activation_transform(x,signs,xp=jnp),w,s,bm=bm,bk=bk,bn=bn))
                start=time.perf_counter();compiled=fn.lower(x,words,scales,signs).compile();row['compile_seconds']=time.perf_counter()-start
                y=compiled(x,words,scales,signs);y.block_until_ready()
                times=[]
                for _ in range(9):
                    start=time.perf_counter();y=compiled(x,words,scales,signs);y.block_until_ready();times.append(time.perf_counter()-start)
                got=np.asarray(y);delta=got-expected
                row.update(seconds=times,median_seconds=statistics.median(times),max_abs_error=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected)),memory_analysis=str(compiled.memory_analysis()))
                assert np.isfinite(got).all()
                assert row['relative_l2']<0.01
            except Exception as e:
                row['error']=str(e);traceback.print_exc(limit=2)
            records.append(row);print(json.dumps(row),flush=True)
            with open('/content/bonsai-results/pallas-matmul.json','w') as f:json.dump(records,f,indent=2)
    print('PALLAS_COMPLETE',flush=True)
main()
