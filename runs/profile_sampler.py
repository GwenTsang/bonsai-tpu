exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import engine as EG
def timeit(fn,*a,n=30):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return float(np.median(ts)*1e3)
lg=jnp.asarray(np.random.default_rng(0).standard_normal(248320).astype(np.float32))
k=jax.random.PRNGKey(0)
a=(jnp.float32(.7),jnp.int32(20),jnp.float32(.95),jnp.float32(0))
v1,i1=jax.jit(lambda l:jax.lax.top_k(l,64))(lg);v2,i2=jax.jit(EG.top_candidates)(lg)
print('top-k exact match',bool((np.asarray(i1)==np.asarray(i2)).all()))
print('full top_k ms',timeit(jax.jit(lambda l:jax.lax.top_k(l,64)),lg))
print('two-level top_k ms',timeit(jax.jit(EG.top_candidates),lg))
print('sample_token ms',timeit(jax.jit(EG.sample_token),lg,k,*a))
print('argmax ms',timeit(jax.jit(jnp.argmax),lg))
step=jax.jit(functools.partial(M.decode_step,CFG,act_len=1024),donate_argnums=(1,))
st=M.init_state(CFG,1024)
for i in range(3):l,st=step(PARAMS,st,jnp.int32(100+i))
ts=[]
for i in range(20):
    t=time.perf_counter();l,st=step(PARAMS,st,jnp.int32(100+i));l.block_until_ready();ts.append(time.perf_counter()-t)
print('single step ms',np.median(ts)*1e3)
blk=jax.jit(functools.partial(EG.decode_block,CFG,1024,16),donate_argnums=(1,))
stop=jnp.asarray([-1],jnp.int32)
r=blk(PARAMS,st,jnp.int32(5),k,jnp.int32(16),*a,stop);st=r[2];jax.block_until_ready(r)
ts=[]
for i in range(8):
    t=time.perf_counter();r=blk(PARAMS,st,jnp.int32(5),k,jnp.int32(16),*a,stop);st=r[2];jax.block_until_ready(r);ts.append((time.perf_counter()-t)/16)
print('block-16 ms/token',np.median(ts)*1e3,'tok/s',1e3/(np.median(ts)*1e3))
