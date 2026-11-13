"""Decode-regime (M small) packed GEMV variants, chained in ONE jitted program to amortize dispatch.
Random ternary data; timings do not depend on values. Results -> /content/bonsai-results/gemv-variants.json"""
import json,time,sys
import numpy as np, jax, jax.numpy as jnp
K,N,L=5120,17408,8
HI=jax.lax.Precision.HIGHEST
rng=np.random.default_rng(0)
def mk():
    words=jnp.asarray(rng.integers(0,2**32,size=(K//16,N),dtype=np.uint32)&np.uint32(0x55555555)|(rng.integers(0,2**32,size=(K//16,N),dtype=np.uint32)&np.uint32(0xAAAAAAAA)&np.uint32(0))) # codes 0/1 -> -1/0
    return words
SH=jnp.arange(16,dtype=jnp.uint32)*2
def unpack(w,dt):
    q=((w[:,None,:]>>SH[None,:,None])&3).astype(jnp.int32)-1
    return q.reshape(K,N).astype(dt)
def v_f32(x,w,s):
    a=(unpack(w,jnp.float32).reshape(K//128,128,N)*s[:,None,:]).reshape(K,N)
    return jnp.matmul(x,a,precision=HI)
def v_bf16(x,w,s):
    a=(unpack(w,jnp.float32).reshape(K//128,128,N)*s[:,None,:]).reshape(K,N).astype(jnp.bfloat16)
    return jax.lax.dot_general(x.astype(jnp.bfloat16),a,(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
def v_group(x,w,s):
    q=unpack(w,jnp.bfloat16).reshape(K//128,128,N)
    xg=x.astype(jnp.bfloat16).reshape(-1,K//128,128).transpose(1,0,2)
    p=jax.lax.dot_general(xg,q,(((2,),(1,)),((0,),(0,))),preferred_element_type=jnp.float32)
    return jnp.sum(p*s[:,None,:],axis=0)
def v_group_i8(x,w,s):
    q=unpack(w,jnp.int8).reshape(K//128,128,N)
    xi=jnp.clip(jnp.round(x*20),-127,127).astype(jnp.int8).reshape(-1,K//128,128).transpose(1,0,2)
    p=jax.lax.dot_general(xi,q,(((2,),(1,)),((0,),(0,))),preferred_element_type=jnp.int32)
    return jnp.sum(p.astype(jnp.float32)*s[:,None,:],axis=0)*0.05
def v_i8pre(x,a,s):
    xi=jnp.clip(jnp.round(x*20),-127,127).astype(jnp.int8)
    return jax.lax.dot_general(xi,a,(((1,),(0,)),((),())),preferred_element_type=jnp.int32).astype(jnp.float32)
def v_unpack_only(x,w,s):
    return jnp.sum(unpack(w,jnp.int8).astype(jnp.int32),axis=0,keepdims=True).astype(jnp.float32)+x[:, :1]
def v_read_only(x,w,s):
    return jnp.sum(w,axis=0,keepdims=True).astype(jnp.float32)+x[:, :1]
def chain(f):
    def g(x,ws,ss):
        acc=0.
        for w,s in zip(ws,ss):
            y=f(x,w,s);acc=acc+y[:,:8];x=x+y[:,:1]*1e-9  # serial dependency
        return acc
    return jax.jit(g)
def timeit(fn,*a,n=15):
    t=time.perf_counter();fn(*a).block_until_ready();c=time.perf_counter()-t
    ts=[]
    for _ in range(n):
        t=time.perf_counter();fn(*a).block_until_ready();ts.append(time.perf_counter()-t)
    return c,ts
ws=[mk() for _ in range(L)];ss=[jnp.asarray(rng.random((K//128,N),dtype=np.float32)) for _ in range(L)]
res=[]
only=sys.argv[1:] if False else None
for M in (1,2,8):
    x=jnp.asarray(rng.standard_normal((M,K),dtype=np.float32))
    for name,f in [('f32',v_f32),('bf16',v_bf16),('group_bf16',v_group),('group_i8',v_group_i8),('unpack_only',v_unpack_only),('read_only',v_read_only)]:
        try:
            c,ts=timeit(chain(f),x,ws,ss)
            r=dict(variant=name,M=M,compile_s=c,per_matrix_ms=float(np.median(ts))/L*1e3,min_ms=min(ts)/L*1e3,max_ms=max(ts)/L*1e3)
        except Exception as e:r=dict(variant=name,M=M,error=repr(e)[:300])
        res.append(r);print(json.dumps(r),flush=True)
del ws
a8=[jnp.asarray(rng.integers(-1,2,size=(K,N),dtype=np.int8)) for _ in range(L)]
for M in (1,8):
    x=jnp.asarray(rng.standard_normal((M,K),dtype=np.float32))
    c,ts=timeit(chain(v_i8pre),x,a8,ss)
    r=dict(variant='int8_preexpanded',M=M,compile_s=c,per_matrix_ms=float(np.median(ts))/L*1e3,min_ms=min(ts)/L*1e3,max_ms=max(ts)/L*1e3)
    res.append(r);print(json.dumps(r),flush=True)
del a8
json.dump(res,open('/content/bonsai-results/gemv-variants.json','w'),indent=1)
