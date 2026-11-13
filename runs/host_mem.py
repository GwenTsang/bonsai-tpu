exec(open('/content/runs/load.py').read(),globals())
import functools,os,gc
from bonsai_tpu import engine as EG
def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'):return round(int(l.split()[1])/2**20,2)
print('rss after load GiB',rss(),flush=True)
k=jax.random.PRNGKey(0);a=(jnp.float32(0.),jnp.int32(20),jnp.float32(.95),jnp.float32(0));stop=jnp.asarray([-1],jnp.int32)
st=M.init_state(CFG,4096,jnp.bfloat16,slack=512)
for act in (1024,2048):
    blk=jax.jit(functools.partial(EG.decode_block,CFG,act,8),donate_argnums=(1,))
    t=time.perf_counter();r=blk(PARAMS,st,jnp.int32(5),k,jnp.int32(8),*a,stop);st=r[2];jax.block_until_ready(r)
    print('block act',act,'compile s',round(time.perf_counter()-t,1),'rss',rss(),flush=True)
for m in (64,512):
    pre=jax.jit(functools.partial(M.prefill_chunk,CFG,act_len=1024),donate_argnums=(1,))
    t=time.perf_counter();l,st=pre(PARAMS,st,jnp.zeros(m,jnp.int32),jnp.int32(m));l.block_until_ready()
    print('prefill',m,'compile s',round(time.perf_counter()-t,1),'rss',rss(),flush=True)
del blk,pre,r,l;gc.collect();print('after del fns',rss(),flush=True)
jax.clear_caches();gc.collect();print('after clear_caches',rss(),flush=True)
