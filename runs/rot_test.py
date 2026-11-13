exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.hadamard import fwht
rng=np.random.default_rng(0)
x=rng.standard_normal((512,17408)).astype(np.float32)*np.exp(rng.standard_normal((512,17408))).astype(np.float32)
ref=fwht(x.astype(np.float64),1024)
f=jax.jit(lambda v:M.rotate(v,1.0));y=np.asarray(f(jnp.asarray(x)))
print('rotate batched rel err',float(np.linalg.norm(y-ref)/np.linalg.norm(ref)))
xs=jnp.asarray(x);f(xs).block_until_ready();ts=[]
for _ in range(10):
    t=time.perf_counter();f(xs).block_until_ready();ts.append(time.perf_counter()-t)
print('rotate 512x17408 ms',np.median(ts)*1e3)
