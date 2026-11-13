exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import kernels as KK
rng=np.random.default_rng(0);Mt=512
def timeit(fn,*a,n=10):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return round(float(np.median(ts)*1e3),3)
L=PARAMS['layers'][0];LA=PARAMS['layers'][3];signs=PARAMS['signs']
h=jnp.asarray(rng.standard_normal((Mt,5120)).astype(np.float32))
act=jnp.asarray(rng.standard_normal((Mt,17408)).astype(np.float32))
toks=jnp.asarray(rng.integers(0,200000,size=Mt).astype(np.int32))
cs=jnp.zeros((3,10240));gs=jnp.zeros((48,128,128))
res={}
res['embed_many']=timeit(jax.jit(M.embed_many),PARAMS,toks)
res['rms_norm']=timeit(jax.jit(M.rms_norm),h,L['attn_norm'])
res['rotate_5120']=timeit(jax.jit(lambda x:M.rotate(x,signs[5120])),h)
res['rotate_17408']=timeit(jax.jit(lambda x:M.rotate(x,signs[17408])),act)
res['gemm_qkvz']=timeit(jax.jit(lambda x:KK.gemm(x,L['qkvz'])),h)
res['gemm_ffn_gu']=timeit(jax.jit(lambda x:KK.gemm(x,L['ffn_gu'])),h)
res['gemm_ffn_down']=timeit(jax.jit(lambda x:KK.gemm(x,L['ffn_down'])),act)
res['ffn_many']=timeit(jax.jit(lambda x:M.ffn_many(CFG,L,signs,x)),h)
res['recurrent_mixer_many']=timeit(jax.jit(lambda x,c,g:M.recurrent_mixer_many(CFG,L,signs,x,c,g)),h,cs,gs)
q=jnp.asarray(rng.standard_normal((Mt,48,128)).astype(np.float32));gb=jnp.asarray(rng.random((Mt,48)).astype(np.float32))
res['gdn_scan']=timeit(jax.jit(M.gdn_scan),q,q,q,-gb,gb,gs)
kc=jnp.zeros((4,1024,256));vc=jnp.zeros((4,1024,256))
res['attention_mixer_many']=timeit(jax.jit(lambda x,k,v:M.attention_mixer_many(CFG,LA,signs,x,k,v,jnp.int32(0),1024)),h,kc,vc)
res['alpha_beta']=timeit(jax.jit(lambda x:(jnp.matmul(x,L['alpha'].T,precision=M.HI),jnp.matmul(x,L['beta'].T,precision=M.HI))),h)
res['dispatch_noop']=timeit(jax.jit(lambda x:x[0,:8]+1),h)
for k,v in res.items():print(k,v,'ms',flush=True)
json.dump(res,open('/content/bonsai-results/profile-prefill-parts.json','w'),indent=1)
