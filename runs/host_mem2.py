exec(open('/content/runs/load.py').read(),globals())
import functools,os,gc,ctypes
from bonsai_tpu import engine as EG
def rss():
    d={}
    for l in open('/proc/self/status'):
        if l.startswith(('VmRSS','RssAnon','RssFile')):d[l.split(':')[0]]=round(int(l.split()[1])/2**20,2)
    return d
print('start',rss(),flush=True)
ctypes.CDLL('libc.so.6').malloc_trim(0);print('after malloc_trim',rss(),flush=True)
k=jax.random.PRNGKey(0);a=(jnp.float32(0.),jnp.int32(20),jnp.float32(.95),jnp.float32(0));stop=jnp.asarray([-1],jnp.int32)
st=M.init_state(CFG,4096,jnp.bfloat16,slack=512)
for rep in range(3):
    blk=jax.jit(functools.partial(EG.decode_block,CFG,1024,8),donate_argnums=(1,))
    t=time.perf_counter();r=blk(PARAMS,st,jnp.int32(5),k,jnp.int32(8),*a,stop);st=r[2];jax.block_until_ready(r)
    print('compile',rep,round(time.perf_counter()-t,1),'s',rss(),flush=True)
    del blk,r;jax.clear_caches();gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
    print('  cleared+trim',rss(),flush=True)
