"""Block-diagonal int8 GEMV: one MXU op per tile, exact per-group integer partial sums."""
import json,time,functools
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
K,N,L=5120,17408,8
rng=np.random.default_rng(0)
def prep_x(x,bk):
    """x f32 [K] -> (Xbd int8 [(K/bk)*2ng, bk] in (j,g,i) column order, xsum f32 [G], alpha)"""
    k=x.shape[0];ng=bk//128;nt=k//bk
    alpha=32639.0/jnp.max(jnp.abs(x))
    q=jnp.round(x*alpha).astype(jnp.int32)
    hi=(q+128)>>8;lo=q-(hi<<8)
    def bd(v):
        v=v.reshape(nt,ng,4,32).transpose(0,2,1,3)            # [t,j,g,i]
        eye=jnp.eye(ng,dtype=jnp.int32)                         # row g keeps only group g
        return (eye[None,:,None,:,None]*v[:,None]).reshape(nt,ng,bk)  # [t,row g,(j,g',i)]
    X=jnp.concatenate([bd(hi),bd(lo)],axis=1).reshape(nt*2*ng,bk).astype(jnp.int8)
    return X,q.reshape(k//128,128).sum(-1).astype(jnp.float32),alpha
def kernel(X,w,s,*,bk,bn,mode='bd'):
    k4,n=w.shape;k=k4*4;ng=bk//128
    def body(xr,wr,sr,yr,acc):
        @pl.when(pl.program_id(1)==0)
        def init():acc[...]=jnp.zeros((1,bn),jnp.float32)
        wv=wr[...]
        if mode=='read':
            acc[...]+=jnp.sum(wv[:8].astype(jnp.int32),axis=0,keepdims=True).astype(jnp.float32)
        else:
            t=jnp.concatenate([wv&3,(wv>>2)&3,(wv>>4)&3,wv>>6],axis=0).astype(jnp.int8)
            p=jax.lax.dot_general(xr[...],t,(((1,),(0,)),((),())),preferred_element_type=jnp.int32)
            i=(p[:ng]<<8)+p[ng:]
            acc[...]+=jnp.sum(i.astype(jnp.float32)*sr[...],axis=0,keepdims=True)
        @pl.when(pl.program_id(1)==k//bk-1)
        def store():yr[...]=acc[...]
    return pl.pallas_call(body,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((2*ng,bk),lambda j,g:(g,0)),pl.BlockSpec((bk//4,bn),lambda j,g:(g,j)),pl.BlockSpec((ng,bn),lambda j,g:(g,j))],
            out_specs=pl.BlockSpec((1,bn),lambda j,g:(0,j)),
            scratch_shapes=[pltpu.VMEM((1,bn),jnp.float32)],grid=(n//bn,k//bk)),
        out_shape=jax.ShapeDtypeStruct((1,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')))(X,w,s)
def gemv(x,w,s,*,bk,bn,mode='bd'):
    X,xsum,alpha=prep_x(x,bk)
    y=kernel(X,w,s,bk=bk,bn=bn,mode=mode)[0]
    return (y-xsum@s)/alpha
codes=[rng.integers(0,3,size=(K//4,4,N),dtype=np.uint8) for _ in range(L)]
ws=[jnp.asarray(c[:,0]|c[:,1]<<2|c[:,2]<<4|c[:,3]<<6) for c in codes]
ss_np=[rng.random((K//128,N),dtype=np.float32)+.5 for _ in range(L)]
ss=[jnp.asarray(s) for s in ss_np]
x_np=rng.standard_normal(K).astype(np.float32)
c0=codes[0].reshape(K//128,32,4,N).transpose(0,2,1,3).reshape(K//128,128,N).astype(np.float64)-1
ref=np.einsum('gk,gkn->n',x_np.astype(np.float64).reshape(K//128,128),c0*ss_np[0][:,None,:])
def chain(f):
    def g(x,ws,ss):
        acc=0.
        for w,s in zip(ws,ss):
            y=f(x,w,s);acc=acc+y[:8];x=x+y[:K]*1e-9
        return acc
    return jax.jit(g)
def timeit(fn,*a,n=15):
    t=time.perf_counter();fn(*a).block_until_ready();c=time.perf_counter()-t
    ts=[]
    for _ in range(n):
        t=time.perf_counter();fn(*a).block_until_ready();ts.append(time.perf_counter()-t)
    return c,ts
res=[];x=jnp.asarray(x_np)
for mode in ('bd','read'):
    for bk,bn in ((1024,1088),(1024,2176),(1024,4352),(1024,8704),(1024,17408),(5120,2176),(5120,4352)):
        f=functools.partial(gemv,mode=mode,bk=bk,bn=bn);r=dict(mode=mode,bk=bk,bn=bn)
        try:
            if mode=='bd':
                y=np.asarray(jax.jit(f)(x,ws[0],ss[0]));r['rel_l2']=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
            c,ts=timeit(chain(f),x,ws,ss)
            r.update(compile_s=round(c,2),ms=round(float(np.median(ts))/L*1e3,4),min_ms=round(min(ts)/L*1e3,4))
        except Exception as e:r['error']=repr(e)[:300]
        res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/bd-kernel.json','w'),indent=1)
