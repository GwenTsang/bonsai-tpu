exec(open('/content/runs/load.py').read(),globals())
import functools,subprocess
from bonsai_tpu import engine as EG
jax.config.update('jax_compilation_cache_dir','/content/jax-cache')
jax.config.update('jax_persistent_cache_min_compile_time_secs',1.0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes',0)
k=jax.random.PRNGKey(0);a=(jnp.float32(0.),jnp.int32(20),jnp.float32(.95),jnp.float32(0));stop=jnp.asarray([-1],jnp.int32)
for rep in range(2):
    fn=jax.jit(functools.partial(EG.decode_block,CFG,8),donate_argnums=(1,))
    st=M.init_state(CFG,4096,jnp.bfloat16)
    t=time.perf_counter();r=fn(PARAMS,st,jnp.int32(5),k,jnp.int32(8),*a,stop);jax.block_until_ready(r)
    print('rep',rep,'first-call s',round(time.perf_counter()-t,1),flush=True)
    del fn,r,st;jax.clear_caches()
print(subprocess.run('ls -la /content/jax-cache | head; du -sh /content/jax-cache',shell=True,capture_output=True,text=True).stdout)
