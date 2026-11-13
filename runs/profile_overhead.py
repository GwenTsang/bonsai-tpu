exec(open('/content/runs/load.py').read(),globals())
import functools
from bonsai_tpu import kernels as KK
real_gemv=KK.gemv
def fake_gemv(x,mat,prepared=None):
    n=mat['w'].shape[1]
    return jnp.resize(x,(n,))*0.01
def bench(label,act_len=1024,n=20):
    step=jax.jit(functools.partial(M.decode_step,CFG,act_len=act_len),donate_argnums=(1,))
    st=M.init_state(CFG,act_len)
    for i in range(3):l,st=step(PARAMS,st,jnp.int32(100+i))
    ts=[]
    for i in range(n):
        t=time.perf_counter();l,st=step(PARAMS,st,jnp.int32(100+i));l.block_until_ready();ts.append(time.perf_counter()-t)
    r=float(np.median(ts)*1e3);print(label,round(r,3),'ms',flush=True);return r
res={}
res['full']=bench('full')
KK.gemv=fake_gemv
res['no_kernels']=bench('no kernels (all else)')
orig_rot=M.rotate;M.rotate=lambda x,s,gdn=False:x
res['no_kernels_no_rotate']=bench('  - rotate')
orig_rec=M.recurrent_mixer
def rec_nogdn(cfg,L,signs,h,cs,gs):
    c=cfg.gdn_inner+2*cfg.gdn_k_heads*cfg.gdn_dk
    qkvz=KK.gemv(h,L['qkvz']);return KK.gemv(qkvz[c:],L['out']),cs,gs
M.recurrent_mixer=rec_nogdn
res['no_recurrent_math']=bench('  - recurrent math (conv/gdn/alpha/beta)')
orig_att=M.attention_mixer
def att_none(cfg,L,signs,h,kc,vc,pos,act_len):
    q=KK.gemv(h,L['qkv']);return KK.gemv(q[:6144],L['out']),kc,vc
M.attention_mixer=att_none
res['no_attention_math']=bench('  - attention math')
orig_embed=M.embed;M.embed=lambda params,token:jnp.ones(5120)*token
res['no_embed']=bench('  - embed')
M.embed=orig_embed;M.attention_mixer=orig_att;M.recurrent_mixer=orig_rec;M.rotate=orig_rot;KK.gemv=real_gemv
json.dump(res,open('/content/bonsai-results/profile-overhead.json','w'),indent=1)
