exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import kernels as KK
real_gemm=KK.gemm
def fake_gemm(x,mat,bm_max=256):
    n=mat['w'].shape[1];m=x.shape[0]
    return jnp.resize(x,(m,n))*0.01
rng=np.random.default_rng(0)
def bench(label,Mtok,act_len=1024,n=5):
    pre=jax.jit(functools.partial(M.prefill_chunk,CFG,act_len=act_len),donate_argnums=(1,))
    toks=jnp.asarray(rng.integers(0,200000,size=Mtok).astype(np.int32))
    st=M.init_state(CFG,act_len);l,st=pre(PARAMS,st,toks);l.block_until_ready();ts=[]
    for i in range(n):
        st=M.init_state(CFG,act_len);jax.block_until_ready(st)
        t=time.perf_counter();l,st=pre(PARAMS,st,toks);l.block_until_ready();ts.append(time.perf_counter()-t)
    r=float(np.median(ts)*1e3);print(label,'M=%d'%Mtok,round(r,1),'ms',round(Mtok/r*1e3),'tok/s',flush=True);return r
res={}
for Mtok in (128,512):
    res['full_%d'%Mtok]=bench('full',Mtok)
    KK.gemm=fake_gemm
    res['nokernel_%d'%Mtok]=bench('no gemm',Mtok)
    og=M.gdn_scan
    M.gdn_scan=lambda q,k,v,gate,beta,state:(v*1.0,state)
    res['nokernel_nogdn_%d'%Mtok]=bench('no gemm, no gdn scan',Mtok)
    orot=M.rotate;M.rotate=lambda x,s,gdn=False:x
    res['nokernel_nogdn_norot_%d'%Mtok]=bench('no gemm, no gdn, no rotate',Mtok)
    oa=M.attention_mixer_many
    def att_none(cfg,L,signs,h,kc,vc,pos,act_len):
        q=KK.gemm(h,L['qkv']);return KK.gemm(q[:,:6144],L['out']),kc,vc
    M.attention_mixer_many=att_none
    res['plus_noattn_%d'%Mtok]=bench('... no attention math',Mtok)
    M.attention_mixer_many=oa;M.rotate=orot;M.gdn_scan=og;KK.gemm=real_gemm
json.dump(res,open('/content/bonsai-results/profile-prefill.json','w'),indent=1)
