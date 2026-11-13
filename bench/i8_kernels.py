"""uint8-lane unpack + int8 MXU group kernels (decode regime), chained to amortize dispatch."""
import json,time,functools
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
K,N,L=5120,17408,8
rng=np.random.default_rng(0)
def kernel(x,w,s,*,mode,bm,bk,bn,xdt):
    m,k=x.shape;n=w.shape[1];ng=bk//128
    def body(xr,wr,sr,yr,acc):
        @pl.when(pl.program_id(2)==0)
        def init():acc[...]=jnp.zeros((bm,bn),jnp.float32)
        wv=wr[...];xv=xr[...];sv=sr[...]
        if mode=='bf16':
            t=jnp.concatenate([jnp.concatenate([(wv[32*g:32*g+32]>>(2*j))&3 for j in range(4)],axis=0) for g in range(ng)],axis=0)
            t=(t.astype(jnp.int8)-1).astype(jnp.float32).reshape(ng,128,bn)*sv[:,None,:]
            acc[...]+=jax.lax.dot_general(xv,t.reshape(bk,bn).astype(jnp.bfloat16),(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
        else:
            tot=jnp.zeros((bm,bn),jnp.float32)
            for g in range(ng):
                b=wv[32*g:32*g+32]
                if mode=='nounpack':t=jnp.concatenate([b,b,b,b],axis=0).astype(jnp.int8)
                else:
                    t=jnp.concatenate([b&3,(b>>2)&3,(b>>4)&3,b>>6],axis=0).astype(jnp.int8)
                    if mode=='signed':t=t-1
                p=jax.lax.dot_general(xv[:,128*g:128*g+128],t,(((1,),(0,)),((),())),preferred_element_type=jnp.int32)
                tot=tot+p.astype(jnp.float32)*sv[g:g+1,:]
            acc[...]+=tot
        @pl.when(pl.program_id(2)==k//bk-1)
        def store():yr[...]=acc[...]
    return pl.pallas_call(body,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm,bk),lambda i,j,g:(i,g)),pl.BlockSpec((bk//4,bn),lambda i,j,g:(g,j)),pl.BlockSpec((bk//128,bn),lambda i,j,g:(g,j))],
            out_specs=pl.BlockSpec((bm,bn),lambda i,j,g:(i,j)),
            scratch_shapes=[pltpu.VMEM((bm,bn),jnp.float32)],grid=(m//bm,n//bn,k//bk)),
        out_shape=jax.ShapeDtypeStruct((m,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel','arbitrary')))(x.astype(xdt),w,s)
codes=[rng.integers(0,3,size=(K//4,4,N),dtype=np.uint8) for _ in range(L)]
ws=[jnp.asarray(c[:,0]|c[:,1]<<2|c[:,2]<<4|c[:,3]<<6) for c in codes]
ss_np=[rng.random((K//128,N),dtype=np.float32) for _ in range(L)]
ss=[jnp.asarray(s) for s in ss_np]
def chain(f):
    def g(x,ws,ss):
        acc=0.
        for w,s in zip(ws,ss):
            y=f(x,w,s);acc=acc+y[:,:8];x=x+(y[:,:1]*0).astype(x.dtype)
        return acc
    return jax.jit(g)
def timeit(fn,*a,n=15):
    t=time.perf_counter();fn(*a).block_until_ready();c=time.perf_counter()-t
    ts=[]
    for _ in range(n):
        t=time.perf_counter();fn(*a).block_until_ready();ts.append(time.perf_counter()-t)
    return c,ts
# numerics: signed kernel vs numpy on matrix 0
xi=rng.integers(-127,128,size=(8,K)).astype(np.int8)
c0=codes[0].reshape(K//128,32,4,N).transpose(0,2,1,3).reshape(K,N).astype(np.int32)-1  # k=128g+32j+i
ref=np.einsum('mgk,gkn->mn',xi.astype(np.float64).reshape(8,K//128,128)[:, :, :],(c0.reshape(K//128,128,N)*ss_np[0][:,None,:]).astype(np.float64)[:, :, :64])
res=[]
for mode,xdt in (('signed',jnp.int8),('unsigned',jnp.int8),('nounpack',jnp.int8),('bf16',jnp.bfloat16)):
    for bm in (8,32):
        for bk,bn in ((1024,1088),(1024,2176),(2560,2176),(5120,2176),(5120,4352)):
            f=functools.partial(kernel,mode=mode,bm=bm,bk=bk,bn=bn,xdt=xdt)
            r=dict(mode=mode,bm=bm,bk=bk,bn=bn)
            try:
                x=jnp.asarray(xi if bm==8 else np.tile(xi,(4,1)))
                if mode=='signed' and bk==1024 and bn==2176:
                    y=np.asarray(jax.jit(f)(x,ws[0],ss[0]))[:8,:64];r['rel_err']=float(np.abs(y-ref).max()/np.abs(ref).max())
                c,ts=timeit(chain(f),x,ws,ss)
                r.update(compile_s=round(c,2),ms=round(float(np.median(ts))/L*1e3,4),min_ms=round(min(ts)/L*1e3,4))
            except Exception as e:r['error']=repr(e)[:300]
            res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/i8-kernels.json','w'),indent=1)
