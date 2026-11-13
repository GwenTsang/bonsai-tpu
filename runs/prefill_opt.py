exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import kernels as KK
from bonsai_tpu.hadamard import fwht, split_bf16, _sylvester
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
rng=np.random.default_rng(0);Mt=512
def timeit(fn,*a,n=10):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return round(float(np.median(ts)*1e3),3)
res={}
x=(rng.standard_normal((Mt,17408))*np.exp(rng.standard_normal((Mt,17408)))).astype(np.float32)
ref=fwht(x.astype(np.float64),1024);xs=jnp.asarray(x)
H128=jnp.asarray(_sylvester(128),dtype=jnp.bfloat16)
def rot_h8h128(v):
    m,k=v.shape
    a,b=split_bf16(v.reshape(-1,128),jnp,jax.lax)
    y=jax.lax.dot_general(jnp.concatenate([a,b],0),H128,(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
    r=a.shape[0];y=(y[:r]+y[r:]).reshape(m,k//1024,8,128)
    for s in (1,2,4):
        z=y.reshape(m,k//1024,8//(2*s),2,s,128);p,q=z[:,:,:,0],z[:,:,:,1]
        y=jnp.stack((p+q,p-q),axis=3).reshape(m,k//1024,8,128)
    return (y*(1024**-0.5)).reshape(m,k)
def rot_single(v):
    h=jnp.asarray(_sylvester(1024),dtype=jnp.bfloat16)
    y=jax.lax.dot_general(v.reshape(-1,1024).astype(jnp.bfloat16),h,(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
    return (y*(1024**-0.5)).reshape(v.shape)
for name,f in (('split2_1024',lambda v:M.rotate(v,1.0)),('h8_x_h128',rot_h8h128),('single_bf16',rot_single)):
    jf=jax.jit(f);y=np.asarray(jf(xs))
    res['rot_'+name]=dict(ms=timeit(jf,xs),rel=float(np.linalg.norm(y-ref)/np.linalg.norm(ref)));print('rot',name,res['rot_'+name],flush=True)
# whole prefill chunk and ablations
def bench(label,n=5):
    pre=jax.jit(functools.partial(M.prefill_chunk,CFG),donate_argnums=(1,))
    toks=jnp.asarray(rng.integers(0,200000,size=Mt).astype(np.int32))
    st=M.init_state(CFG,4096,jnp.bfloat16,slack=512);l,st=pre(PARAMS,st,toks,jnp.int32(Mt));l.block_until_ready();ts=[]
    for i in range(n):
        st['pos']=jnp.int32(0)
        t=time.perf_counter();l,st=pre(PARAMS,st,toks,jnp.int32(Mt));l.block_until_ready();ts.append(time.perf_counter()-t)
    r=float(np.median(ts)*1e3);print(label,round(r,1),'ms',round(Mt/r*1e3),'tok/s',flush=True);del st;return r
res['full']=bench('full 512')
og=M.gdn_chunked;M.gdn_chunked=lambda q,k,v,gate,beta,state,chunk=32:(v*1.0,state)
res['no_gdn']=bench('no gdn')
orot=M.rotate;M.rotate=lambda x,s,gdn=False:x
res['no_gdn_no_rot']=bench('no gdn, no rotate')
oa=M.attention_mixer_many
def att_none(cfg,L,signs,h,kc,vc,pos):
    q=KK.gemm(h,L['qkv']);return KK.gemm(q[:,:6144],L['out']),kc,vc
M.attention_mixer_many=att_none
res['no_gdn_no_rot_no_att']=bench('no gdn, no rotate, no attention math')
M.attention_mixer_many=oa;M.rotate=orot;M.gdn_chunked=og
json.dump(res,open('/content/bonsai-results/prefill-opt.json','w'),indent=1)
