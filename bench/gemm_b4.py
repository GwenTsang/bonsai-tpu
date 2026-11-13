"""Prefill GEMM over B4: 8-bit-lane unpack, scale folded into BF16 weights in VMEM, one MXU op per plane/tile."""
import json,time,functools
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
K,N,L=5120,17408,8
rng=np.random.default_rng(0)
def kernel(x,w,s,*,bm,bn,bk=1024,mode='planes'):
    m,k=x.shape;n=w.shape[1];ng=bk//128;nt=k//bk
    def body(xr,wr,sr,yr,acc):
        @pl.when(pl.program_id(2)==0)
        def _():acc[...]=jnp.zeros((bm,bn),jnp.float32)
        wv=wr[...];sv=sr[...]
        srep=jnp.concatenate([jnp.broadcast_to(sv[g:g+1],(32,bn)) for g in range(ng)],axis=0)   # [bk/4,bn]
        planes=[wv&3,(wv>>2)&3,(wv>>4)&3,wv>>6]
        ws=[((p.astype(jnp.int32).astype(jnp.float32)-1.0)*srep).astype(jnp.bfloat16) for p in planes]
        xv=xr[...]
        if mode=='planes':
            tot=acc[...]
            for j in range(4):
                tot=tot+jax.lax.dot_general(xv[:,j*(bk//4):(j+1)*(bk//4)],ws[j],(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
            acc[...]=tot
        else:
            acc[...]+=jax.lax.dot_general(xv,jnp.concatenate(ws,axis=0),(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
        @pl.when(pl.program_id(2)==nt-1)
        def _():yr[...]=acc[...]
    return pl.pallas_call(body,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm,bk),lambda i,j,g:(i,g)),pl.BlockSpec((bk//4,bn),lambda i,j,g:(g,j)),pl.BlockSpec((ng,bn),lambda i,j,g:(g,j))],
            out_specs=pl.BlockSpec((bm,bn),lambda i,j,g:(i,j)),
            scratch_shapes=[pltpu.VMEM((bm,bn),jnp.float32)],grid=(m//bm,n//bn,nt)),
        out_shape=jax.ShapeDtypeStruct((m,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel','arbitrary')))(x,w,s)
def gemm(x,w,s,**kw):
    m,k=x.shape
    xp=x.reshape(m,k//1024,8,4,32).transpose(0,1,3,2,4).reshape(m,k).astype(jnp.bfloat16)   # (tile,j,g,i)
    return kernel(xp,w,s,**kw)
codes=[rng.integers(0,3,size=(K//4,4,N),dtype=np.uint8) for _ in range(L)]
ws=[jnp.asarray(c[:,0]|c[:,1]<<2|c[:,2]<<4|c[:,3]<<6) for c in codes]
ss_np=[rng.random((K//128,N),dtype=np.float32)+.5 for _ in range(L)]
ss=[jnp.asarray(s) for s in ss_np]
c0=(codes[0].reshape(K//128,32,4,N).transpose(0,2,1,3).reshape(K//128,128,N).astype(np.float32)-1)*ss_np[0][:,None,:]
x8=rng.standard_normal((8,K)).astype(np.float32)
ref=x8.astype(np.float64)@c0.reshape(K,N)[:,:512].astype(np.float64)
def chain(f):
    def g(x,ws,ss):
        acc=0.
        for w,s in zip(ws,ss):
            y=f(x,w,s);acc=acc+y[:8,:8];x=x+y[:,:K]*1e-9
        return acc
    return jax.jit(g)
def timeit(fn,*a,n=10):
    t=time.perf_counter();fn(*a).block_until_ready();c=time.perf_counter()-t
    ts=[]
    for _ in range(n):
        t=time.perf_counter();fn(*a).block_until_ready();ts.append(time.perf_counter()-t)
    return c,ts
res=[]
for M,bm,bn,mode in ((8,8,2176,'planes'),(8,8,2176,'cat'),(128,128,2176,'planes'),(128,128,2176,'cat'),(512,256,2176,'planes'),(512,512,2176,'planes'),(512,512,2176,'cat'),(512,512,1024,'planes'),(512,512,4352,'planes'),(2048,512,2176,'planes'),(2048,1024,2176,'planes'),(2048,2048,2176,'planes')):
    f=functools.partial(gemm,bm=bm,bn=bn,mode=mode);r=dict(M=M,bm=bm,bn=bn,mode=mode)
    try:
        if M==8:
            y=np.asarray(jax.jit(f)(jnp.asarray(x8),ws[0],ss[0]))[:,:512];r['rel_l2']=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
        x=jnp.asarray(rng.standard_normal((M,K),dtype=np.float32))
        c,ts=timeit(chain(f),x,ws,ss)
        r.update(compile_s=round(c,2),ms=round(float(np.median(ts))/L*1e3,4),min_ms=round(min(ts)/L*1e3,4))
        r['tflops']=round(2*M*K*N/(r['ms']*1e-3)/1e12,1)
    except Exception as e:r['error']=repr(e)[:400]
    res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/gemm-b4.json','w'),indent=1)
