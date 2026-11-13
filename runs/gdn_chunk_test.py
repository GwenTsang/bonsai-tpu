exec(open('/content/runs/load.py').read(),globals())
import functools
rng=np.random.default_rng(0)
def timeit(fn,*a,n=10):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return round(float(np.median(ts)*1e3),3)
res=[]
for T in (512,200):
    q=rng.standard_normal((T,48,128)).astype(np.float32);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    k=rng.standard_normal((T,48,128)).astype(np.float32);k[1:]+=0.5*k[:-1];k/=np.linalg.norm(k,axis=-1,keepdims=True)
    v=rng.standard_normal((T,48,128)).astype(np.float32)
    gate=-np.abs(rng.standard_normal((T,48))).astype(np.float32)*0.05;beta=rng.random((T,48)).astype(np.float32)
    s0=rng.standard_normal((48,128,128)).astype(np.float32)*0.1
    a=[jnp.asarray(x) for x in (q,k,v,gate,beta,s0)]
    o1,s1=jax.jit(M.gdn_scan)(*a)
    r=dict(T=T,scan_ms=timeit(jax.jit(M.gdn_scan),*a))
    for C in (16,32,64,128):
        f=jax.jit(functools.partial(M.gdn_chunked,chunk=C))
        o2,s2=f(*a)
        r['c%d'%C]=dict(ms=timeit(f,*a),out_rel=float(jnp.linalg.norm(o2-o1)/jnp.linalg.norm(o1)),state_rel=float(jnp.linalg.norm(s2-s1)/jnp.linalg.norm(s1)))
    res.append(r);print(json.dumps(r),flush=True)
json.dump(res,open('/content/bonsai-results/gdn-chunked.json','w'),indent=1)
