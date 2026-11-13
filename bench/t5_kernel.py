"""T5 layout: 5 trits/byte (1.6 bit/weight), trit plane i == scale group i of each 640-input tile."""
import json,time,functools
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
K,N,L=5120,17408,8
T=K//640
rng=np.random.default_rng(0)
def prep_x(x):
    alpha=32639.0/jnp.max(jnp.abs(x));q=jnp.round(x*alpha).astype(jnp.int32)
    hi=(q+128)>>8;lo=q-(hi<<8)
    eye=jnp.eye(8,5,dtype=jnp.int32)[None,:,:,None]            # [1,row,g,1]
    def bd(v):return (eye*v.reshape(T,1,5,128)).reshape(T,8,640)
    X=jnp.concatenate([bd(hi),bd(lo)],axis=1).reshape(T*16,640).astype(jnp.int8)
    return X,q.reshape(K//128,128).sum(-1).astype(jnp.float32),alpha
def kernel(X,w,s,*,bn,mode,sdt=None):
    k5,n=w.shape
    def body(xr,wr,sr,yr):
        acc=jnp.zeros((1,bn),jnp.float32)
        for t in range(T):
            wv=wr[128*t:128*t+128,:]
            if mode=='read':
                acc=acc+jnp.sum(wv[:8].astype(jnp.int32),axis=0,keepdims=True).astype(jnp.float32);continue
            a=wv.astype(jnp.int16) if mode!='i32' else wv.astype(jnp.int32)
            planes=[]
            for i in range(5):
                p=a*3 if mode=='mul' else a+(a<<1)
                planes.append((p>>8).astype(jnp.int8));a=p&255
            tt=jnp.concatenate(planes,axis=0)
            p=jax.lax.dot_general(xr[16*t:16*t+16,:],tt,(((1,),(0,)),((),())),preferred_element_type=jnp.int32)
            e=(p[:5]<<8)+p[8:13]
            sv=sr[5*t:5*t+5,:]
            if sdt is not None:sv=sv.astype(jnp.float32)
            acc=acc+jnp.sum(e.astype(jnp.float32)*sv,axis=0,keepdims=True)
        yr[...]=acc
    return pl.pallas_call(body,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((T*16,640),lambda j:(0,0)),pl.BlockSpec((k5,bn),lambda j:(0,j)),pl.BlockSpec((K//128,bn),lambda j:(0,j))],
            out_specs=pl.BlockSpec((1,bn),lambda j:(0,j)),grid=(n//bn,)),
        out_shape=jax.ShapeDtypeStruct((1,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)))(X,w,s)
def gemv(x,w,s,**kw):
    X,xsum,alpha=prep_x(x)
    return (kernel(X,w,s,**kw)[0]-xsum@s.astype(jnp.float32))/alpha
digs=[rng.integers(0,3,size=(K//5,5,N),dtype=np.int32) for _ in range(L)]   # [byte,trit i,n]; k=640*t+128*i+b
def enc(d):
    q=d[:,0]*81+d[:,1]*27+d[:,2]*9+d[:,3]*3+d[:,4]
    return ((q*256+242)//243).astype(np.uint8)
ws=[jnp.asarray(enc(d)) for d in digs]
ss_np=[(rng.random((K//128,N),dtype=np.float32)+.5).astype(np.float16).astype(np.float32) for _ in range(L)]
ss=[jnp.asarray(s) for s in ss_np]
x_np=rng.standard_normal(K).astype(np.float32)
c0=digs[0].reshape(T,128,5,N).transpose(0,2,1,3).reshape(K//128,128,N).astype(np.float64)-1
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
for mode,sdt in (('shadd',None),('mul',None),('i32',None),('read',None),('shadd','f16')):
    for bn in (1088,2176,4352):
        f=functools.partial(gemv,mode=mode,bn=bn,sdt=sdt);r=dict(mode=mode,bn=bn,sdt=sdt)
        sl=ss if sdt is None else [s.astype(jnp.float16) for s in ss]
        try:
            if mode!='read':
                y=np.asarray(jax.jit(f)(x,ws[0],sl[0]));r['rel_l2']=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
            c,ts=timeit(chain(f),x,ws,sl)
            r.update(compile_s=round(c,2),ms=round(float(np.median(ts))/L*1e3,4),min_ms=round(min(ts)/L*1e3,4))
        except Exception as e:r['error']=repr(e)[:400]
        res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/t5-kernel.json','w'),indent=1)
