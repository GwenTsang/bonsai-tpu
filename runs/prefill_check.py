exec(open('/content/runs/load.py').read(),globals())
import functools
ref=np.fromfile('/content/reference-capital.f32','<f4').reshape(8,-1)
prompt=[760,6511,314,9338,369];greedy=[11751,13,198,760,6511,314,9564,369]
N_CTX=1024
def cmp(l,r):
    l=np.asarray(l);p=np.exp(r-r.max());p/=p.sum();lq=l-l.max();lq=lq-np.log(np.exp(lq).sum())
    return dict(argmax=int(l.argmax()),ref_argmax=int(r.argmax()),rel_l2=float(np.linalg.norm(l-r)/np.linalg.norm(r)),kl=float(np.sum(p*(np.log(p+1e-30)-lq))))
step=jax.jit(functools.partial(M.decode_step,CFG,act_len=N_CTX),donate_argnums=(1,))
pre=jax.jit(functools.partial(M.prefill_chunk,CFG,act_len=N_CTX),donate_argnums=(1,))
out={}
# A: prefill 5-token prompt in one batch, then decode greedily and compare to CPU reference at each step
state=M.init_state(CFG,N_CTX,slack=256)
t=time.perf_counter();logits,state=pre(PARAMS,state,jnp.asarray(prompt+[0]*59,jnp.int32),jnp.int32(5));logits.block_until_ready();print('prefill5 compile+run s',round(time.perf_counter()-t,1),flush=True)
rows=[]
for i in range(8):
    rows.append(cmp(logits,ref[i]));print('prefill-path',json.dumps(rows[-1]),flush=True)
    logits,state=step(PARAMS,state,jnp.int32(greedy[i]))
out['prefill_then_decode_vs_reference']=rows
# B: long sequence, batched prefill vs token-by-token decode (same tokens): final logits + states
rng=np.random.default_rng(1);T=200
toks=np.concatenate([prompt,greedy,rng.integers(0,200000,size=T-13)]).astype(np.int32)
toks_pad=np.concatenate([toks,np.zeros(256-T,np.int32)])
sa=M.init_state(CFG,N_CTX)
for tkn in toks:la,sa=step(PARAMS,sa,jnp.int32(tkn))
toks_seq=toks;toks=toks_pad
sb=M.init_state(CFG,N_CTX,slack=256)
t=time.perf_counter();lb,sb=pre(PARAMS,sb,jnp.asarray(toks),jnp.int32(T));lb.block_until_ready();c=time.perf_counter()-t
r=cmp(lb,np.asarray(la))
r['gdn_state_rel']=float(max(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(np.asarray(a)) for a,b in zip(sa['gdn'],sb['gdn'])))
r['k_cache_rel']=float(max(np.linalg.norm(np.asarray(a)[:,:T]-np.asarray(b)[:,:T])/np.linalg.norm(np.asarray(a)) for a,b in zip(sa['k'],sb['k'])))
r['conv_rel']=float(max(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(np.asarray(a)) for a,b in zip(sa['conv'],sb['conv'])))
print('padded-batch(200 of 256)-vs-sequential',json.dumps(r),'compile_s',round(c,1),flush=True);out['batched_vs_sequential_256']=r
ts=[]
for _ in range(5):
    s=M.init_state(CFG,N_CTX,slack=256);t=time.perf_counter();l,s=pre(PARAMS,s,jnp.asarray(toks),jnp.int32(T));l.block_until_ready();ts.append(time.perf_counter()-t)
print('prefill 200(of 256) tok: median s',np.median(ts),flush=True)
assert int(sb['pos'])==T;out['prefill256_s']=ts
json.dump(out,open('/content/bonsai-results/prefill-check.json','w'),indent=1)
