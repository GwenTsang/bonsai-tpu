exec(open('/content/runs/load.py').read(),globals())
import functools
rng=np.random.default_rng(0)
res={}
for Mt in (512,64):
    pre=jax.jit(functools.partial(M.prefill_chunk,CFG),donate_argnums=(1,))
    toks=jnp.asarray(rng.integers(0,200000,size=Mt).astype(np.int32))
    st=M.init_state(CFG,4096,jnp.bfloat16,slack=512);l,st=pre(PARAMS,st,toks,jnp.int32(Mt));l.block_until_ready();ts=[]
    for i in range(7):
        st['pos']=jnp.int32(0)
        t=time.perf_counter();l,st=pre(PARAMS,st,toks,jnp.int32(Mt));l.block_until_ready();ts.append(time.perf_counter()-t)
    r=float(np.median(ts)*1e3);res[Mt]=r;print('prefill chunk',Mt,round(r,1),'ms',round(Mt/r*1e3),'tok/s',flush=True);del st,pre
json.dump(res,open('/content/bonsai-results/prefill-speed.json','w'))
