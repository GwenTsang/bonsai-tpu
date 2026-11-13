exec(open('/content/runs/load.py').read(),globals())
import functools
ref=np.fromfile('/content/reference-capital.f32','<f4').reshape(8,-1)
prompt=[760,6511,314,9338,369];greedy=[11751,13,198,760,6511,314,9564,369]
N_CTX=64
step=jax.jit(functools.partial(M.decode_step,CFG),donate_argnums=(1,))
state=M.init_state(CFG,N_CTX)
t=time.perf_counter()
for tok in prompt:
    logits,state=step(PARAMS,state,jnp.int32(tok))
logits.block_until_ready();print('prompt_s(incl compile)',round(time.perf_counter()-t,1),flush=True)
out=[]
for i in range(8):
    l=np.asarray(logits);r=ref[i]
    p=np.exp(r-r.max());p/=p.sum();lq=l-l.max();lq=lq-np.log(np.exp(lq).sum())
    kl=float(np.sum(p*(np.log(p+1e-30)-lq)))
    row=dict(step=i,argmax=int(l.argmax()),ref_argmax=int(r.argmax()),rel_l2=float(np.linalg.norm(l-r)/np.linalg.norm(r)),max_abs=float(np.abs(l-r).max()),kl=kl,top5_match=bool((np.argsort(-l)[:5]==np.argsort(-r)[:5]).all()))
    out.append(row);print(json.dumps(row),flush=True)
    logits,state=step(PARAMS,state,jnp.int32(greedy[i]))
ts=[]
for i in range(30):
    t=time.perf_counter();logits,state=step(PARAMS,state,jnp.int32(greedy[i%8]));logits.block_until_ready();ts.append(time.perf_counter()-t)
print('step_ms median',np.median(ts)*1e3,'min',min(ts)*1e3,'tok/s',1/np.median(ts),flush=True)
json.dump(dict(logits=out,step_ms=ts),open('/content/bonsai-results/first-logits.json','w'),indent=1)
