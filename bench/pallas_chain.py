"""Chained (dispatch-amortized) Pallas packed matmul vs XLA M=1 unpack formulations."""
import json,time
import numpy as np, jax, jax.numpy as jnp
from bonsai_tpu.pallas_matmul import packed_matmul
K,N,L=5120,17408,8
rng=np.random.default_rng(0)
ws=[jnp.asarray(rng.integers(0,2**32,size=(K//16,N),dtype=np.uint32)&np.uint32(0x55555555)) for _ in range(L)]
ss=[jnp.asarray(rng.random((K//128,N),dtype=np.float32)) for _ in range(L)]
def chain(f):
    def g(x,ws,ss):
        acc=0.
        for w,s in zip(ws,ss):
            y=f(x,w,s);acc=acc+y[:,:8];x=x+y[:,:1]*1e-9
        return acc
    return jax.jit(g)
def timeit(fn,*a,n=15):
    t=time.perf_counter();fn(*a).block_until_ready();c=time.perf_counter()-t
    ts=[]
    for _ in range(n):
        t=time.perf_counter();fn(*a).block_until_ready();ts.append(time.perf_counter()-t)
    return c,ts
res=[]
def rec(name,M,f,extra=None):
    x=jnp.asarray(rng.standard_normal((M,K),dtype=np.float32))
    try:
        c,ts=timeit(chain(f),x,ws,ss)
        r=dict(variant=name,M=M,compile_s=c,per_matrix_ms=float(np.median(ts))/L*1e3,min_ms=min(ts)/L*1e3,max_ms=max(ts)/L*1e3)
    except Exception as e:r=dict(variant=name,M=M,error=repr(e)[:400])
    if extra:r.update(extra)
    res.append(r);print(json.dumps(r),flush=True)
for M,bm in ((1,8),(1,16),(8,8),(32,32),(128,128),(512,256)):
    for bk,bn in ((1024,256),(1024,512),(1024,1024),(1024,2176),(2560,1088),(5120,544),(5120,1088)):
        rec('pallas',M,lambda x,w,s:packed_matmul(x,w,s,bm=bm,bk=bk,bn=bn),dict(bm=bm,bk=bk,bn=bn))
json.dump(res,open('/content/bonsai-results/pallas-chain.json','w'),indent=1)
