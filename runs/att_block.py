exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import engine as EG
k=jax.random.PRNGKey(0);a=(jnp.float32(0.),jnp.int32(20),jnp.float32(.95),jnp.float32(0));stop=jnp.asarray([-1],jnp.int32)
res=[]
for blk_size in (1024,2048,4096,8192):
    M.ATT_BLOCK=blk_size
    for dt in (jnp.bfloat16,):
        fn=jax.jit(functools.partial(EG.decode_block,CFG,16),donate_argnums=(1,))
        row=dict(att_block=blk_size,kv=str(jnp.dtype(dt)))
        for pos in (100,8000,60000):
            st=M.init_state(CFG,65536,dt);st['pos']=jnp.int32(pos)
            r=fn(PARAMS,st,jnp.int32(5),k,jnp.int32(16),*a,stop);st=r[2];jax.block_until_ready(r);ts=[]
            for i in range(5):
                t=time.perf_counter();r=fn(PARAMS,st,jnp.int32(5),k,jnp.int32(16),*a,stop);st=r[2];jax.block_until_ready(r);ts.append((time.perf_counter()-t)/16*1e3)
            row['ms_at_%d'%pos]=round(float(np.median(ts)),3)
            del st,r
        res.append(row);print(json.dumps(row),flush=True)
    del fn;jax.clear_caches()
M.ATT_BLOCK=4096
json.dump(res,open('/content/bonsai-results/att-block.json','w'),indent=1)
