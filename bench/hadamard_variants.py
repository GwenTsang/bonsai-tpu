"""Per-token Hadamard cost: 256 rotations/token (sizes 5120,6144,5120,17408 per layer). Chained, serial."""
import json,time
import numpy as np, jax, jax.numpy as jnp
from bonsai_tpu.hadamard import fwht
HI=jax.lax.Precision.HIGHEST
def sylvester(n):
    h=np.ones((1,1),np.float32)
    while h.shape[0]<n:h=np.block([[h,h],[h,-h]])
    return h
H=jnp.asarray(sylvester(1024));H32=jnp.asarray(sylvester(32));H8=jnp.asarray(sylvester(1024).astype(np.int8))
def v_butterfly(x):return fwht(x,1024,jnp)
def v_dot_hi(x):return (jnp.matmul(x.reshape(-1,1024),H,precision=HI)*(1024**-0.5)).reshape(-1)
def v_dot_default(x):return (jnp.matmul(x.reshape(-1,1024),H)*(1024**-0.5)).reshape(-1)
def v_kron(x):
    y=x.reshape(-1,32,32)
    y=jnp.einsum('ab,nbc->nac',H32,y,precision=HI);y=jnp.einsum('nac,cd->nad',y,H32,precision=HI)
    return (y*(1024**-0.5)).reshape(-1)
def v_int8(x):
    a=32639.0/jnp.maximum(jnp.max(jnp.abs(x)),1e-30);q=jnp.round(x*a).astype(jnp.int32).reshape(-1,1024)
    hi=(q+128)>>8;lo=q-(hi<<8)
    y=jax.lax.dot_general(jnp.concatenate([hi,lo],0).astype(jnp.int8),H8,(((1,),(0,)),((),())),preferred_element_type=jnp.int32)
    n=q.shape[0]
    return (((y[:n]<<8)+y[n:]).astype(jnp.float32)*(1024**-0.5/a)).reshape(-1)
def v_bf16x3(x):
    # exact-ish: split x into three bf16 pieces, +-1 matrix exact in bf16, f32 accumulate
    xr=x.reshape(-1,1024);a=xr.astype(jnp.bfloat16);r=xr-a.astype(jnp.float32);b=r.astype(jnp.bfloat16);c=(r-b.astype(jnp.float32)).astype(jnp.bfloat16)
    y=jax.lax.dot_general(jnp.concatenate([a,b,c],0),H.astype(jnp.bfloat16),(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
    n=xr.shape[0]
    return ((y[:n]+y[n:2*n]+y[2*n:])*(1024**-0.5)).reshape(-1)
sizes=[5120,6144,5120,17408]*64
rng=np.random.default_rng(0)
xs={k:jnp.asarray(rng.standard_normal(k).astype(np.float32)) for k in (5120,6144,17408)}
def chain(f):
    @jax.jit
    def g(xs):
        acc=jnp.zeros(());out=None
        for k in sizes:
            y=f(xs[k]+acc*1e-20);acc=acc+y[0]
        return acc
    return g
res=[]
ref=np.asarray(fwht(np.asarray(xs[17408]).astype(np.float64),1024))
for name,f in (('butterfly',v_butterfly),('dot_highest',v_dot_hi),('dot_default',v_dot_default),('kron32_highest',v_kron),('int8_hilo',v_int8),('bf16x3',v_bf16x3)):
    r=dict(variant=name)
    try:
        y=np.asarray(jax.jit(f)(xs[17408]));r['rel_l2']=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
        g=chain(f);t=time.perf_counter();g(xs).block_until_ready();r['compile_s']=round(time.perf_counter()-t,1);ts=[]
        for _ in range(15):
            t=time.perf_counter();g(xs).block_until_ready();ts.append(time.perf_counter()-t)
        r['ms_per_token']=round(float(np.median(ts))*1e3,3)
    except Exception as e:r['error']=repr(e)[:300]
    res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/hadamard-variants.json','w'),indent=1)
