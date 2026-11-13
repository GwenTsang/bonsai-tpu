# Teacher-forced comparison with the CPU oracle on the representative prompt set, for three numeric paths.
exec(open('/content/runs/load.py').read(),globals())
import functools,hashlib,shutil
from pathlib import Path
name='eval-ref.npz'
man=json.loads(Path('/content/'+name+'.manifest.json').read_text())
with open('/content/'+name,'wb') as dest:
    for part in man['parts']:
        with open('/content/'+part,'rb') as src:shutil.copyfileobj(src,dest)
assert hashlib.file_digest(open('/content/'+name,'rb'),'sha256').hexdigest()==man['sha256']
ref=np.load('/content/'+name)
cases=sorted({k.split('::')[0] for k in ref.files})
step=jax.jit(functools.partial(M.decode_step,CFG),donate_argnums=(1,))
pre512=jax.jit(functools.partial(M.prefill_chunk,CFG),donate_argnums=(1,))
def metrics(l,r):
    l=np.asarray(l,np.float64);r=r.astype(np.float64)
    lp=l-l.max();lp-=np.log(np.exp(lp).sum());rp=r-r.max();rp-=np.log(np.exp(rp).sum())
    return dict(kl=float(np.sum(np.exp(rp)*(rp-lp))),top1=int(l.argmax()==r.argmax()),rel=float(np.linalg.norm(l-r)/np.linalg.norm(r)))
def run(case,path):
    ids=ref[case+'::prompt'].tolist();greedy=ref[case+'::greedy'].tolist();rl=ref[case+'::logits']
    kv=jnp.float32 if path=='decode_f32kv' else jnp.bfloat16
    st=M.init_state(CFG,2048,kv,slack=512)
    if path=='prefill_bf16kv':
        i=0
        while i<len(ids):
            n=min(512,len(ids)-i);ch=ids[i:i+n]+[0]*(512-n)
            lg,st=pre512(PARAMS,st,jnp.asarray(ch,jnp.int32),jnp.int32(n));i+=n
    else:
        for t in ids:lg,st=step(PARAMS,st,jnp.int32(t))
    rows=[]
    for s in range(rl.shape[0]):
        rows.append(metrics(lg,rl[s]))
        lg,st=step(PARAMS,st,jnp.int32(greedy[s]))
    del st
    return rows
out={}
for path in ('decode_f32kv','decode_bf16kv','prefill_bf16kv'):
    agg=[]
    for case in cases:
        rows=run(case,path);out[path+'/'+case]=rows;agg+=rows
        print(path,case,'mean_kl %.2e max_kl %.2e top1 %d/%d mean_rel %.4f'%(np.mean([r['kl'] for r in rows]),max(r['kl'] for r in rows),sum(r['top1'] for r in rows),len(rows),np.mean([r['rel'] for r in rows])),flush=True)
    print('==',path,'ALL mean_kl %.2e max_kl %.2e top1 %d/%d'%(np.mean([r['kl'] for r in agg]),max(r['kl'] for r in agg),sum(r['top1'] for r in agg),len(agg)),flush=True)
json.dump(out,open('/content/bonsai-results/eval-reference.json','w'))
