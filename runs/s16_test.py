# int16-coded scales: correctness and speed on live PARAMS (no reload)
import functools
from bonsai_tpu import kernels as KK
import importlib;importlib.reload(KK)
mats=[PARAMS['head']]+[L[k] for L in PARAMS['layers'] for k in ('qkvz','qkv','out','ffn_gu','ffn_down') if k in L]
def to16(m):
    h=np.asarray(m['s']).astype(np.float16)
    assert (h.astype(np.float32)==np.asarray(m['s'])).all()
    return {'w':m['w'],'s':jnp.asarray(h.view(np.int16))}
m0=mats[5];m16=to16(m0)
x=jnp.asarray(np.random.default_rng(0).standard_normal(m0['w'].shape[0]*4).astype(np.float32))
try:
    ya=jax.jit(KK.gemv)(x,m0);yb=jax.jit(KK.gemv)(x,m16)
    print('gemv max abs diff',float(jnp.max(jnp.abs(ya-yb))),'identical',bool((ya==yb).all()),flush=True)
    xm=jnp.asarray(np.random.default_rng(1).standard_normal((64,m0['w'].shape[0]*4)).astype(np.float32))
    ga=jax.jit(KK.gemm)(xm,m0);gb=jax.jit(KK.gemm)(xm,m16)
    print('gemm identical',bool((ga==gb).all()),flush=True)
    sub=mats[1:41]
    sub16=[to16(m) for m in sub]
    xs={k:jnp.asarray(np.random.default_rng(0).standard_normal(k).astype(np.float32)) for k in (5120,6144,17408)}
    @jax.jit
    def chain(ms,xs):
        acc=jnp.zeros(8)
        for m in ms:
            acc=acc+KK.gemv(xs[m['w'].shape[0]*4]+acc[0]*1e-20,m)[:8]
        return acc
    for label,ms in (('f32 scales',sub),('int16 scales',sub16),('f32 scales',sub),('int16 scales',sub16)):
        chain(ms,xs).block_until_ready();ts=[]
        for _ in range(20):
            t=time.perf_counter();chain(ms,xs).block_until_ready();ts.append(time.perf_counter()-t)
        print(label,'40 matrices ms',round(float(np.median(ts))*1e3,3),flush=True)
except Exception as e:
    print('FAILED',repr(e)[:1500],flush=True)
