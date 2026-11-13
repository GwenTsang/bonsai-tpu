exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import kernels as KK
def timeit(fn,*a,n=20):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return float(np.median(ts)*1e3)
mats=[]
for L in PARAMS['layers']:
    for k in ('qkvz','qkv','out','ffn_gu','ffn_down'):
        if k in L:mats.append(L[k])
mats.append(PARAMS['head'])
xs={k:jnp.asarray(np.random.default_rng(0).standard_normal(k).astype(np.float32)) for k in (5120,6144,17408)}
preps={k:KK.prep_gemv(v) for k,v in xs.items()}
@jax.jit
def kernels_only(mats,preps):
    acc=jnp.zeros(8)
    for m in mats:
        xb,xsum,alpha=preps[m['w'].shape[0]*4]
        acc=acc+KK._gemv_kernel(xb,xsum,m['w'],m['s'],KK.pick_bn(m['w'].shape[1]))[0,:8]
    return acc
@jax.jit
def kernels_prep(mats,xs):
    acc=jnp.zeros(8)
    for m in mats:
        x=xs[m['w'].shape[0]*4]+acc[0]*1e-20
        acc=acc+KK.gemv(x,m)[:8]
    return acc
@jax.jit
def kernels_prep_rot(mats,xs,signs):
    acc=jnp.zeros(8)
    for m in mats:
        k=m['w'].shape[0]*4
        x=xs[k]+acc[0]*1e-20
        acc=acc+KK.gemv(M.rotate(x,signs[k]),m)[:8]
    return acc
res={}
res['kernels_only_ms']=timeit(kernels_only,mats,preps)
res['kernels_plus_prep_ms']=timeit(kernels_prep,mats,xs)
res['kernels_prep_rotate_ms']=timeit(kernels_prep_rot,mats,xs,PARAMS['signs'])
tot=sum(m['w'].size+m['s'].size*4 for m in mats)
res['matrix_bytes']=int(tot);res['n_calls']=len(mats);res['GBps_kernels_only']=tot/res['kernels_only_ms']/1e6
step=jax.jit(functools.partial(M.decode_step,CFG,act_len=1024),donate_argnums=(1,))
st=M.init_state(CFG,1024)
for i in range(3):l,st=step(PARAMS,st,jnp.int32(100+i))
ts=[]
for i in range(20):
    t=time.perf_counter();l,st=step(PARAMS,st,jnp.int32(100+i));l.block_until_ready();ts.append(time.perf_counter()-t)
res['full_step_ms']=float(np.median(ts)*1e3)
print(json.dumps(res,indent=1))
json.dump(res,open('/content/bonsai-results/profile-decode.json','w'),indent=1)
