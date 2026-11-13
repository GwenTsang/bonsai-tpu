exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.engine import Engine
import os
NCTX=int(os.environ.get('BONSAI_NCTX','8192'));KV=os.environ.get('BONSAI_KV','float32')
LENS=[int(x) for x in os.environ.get('BONSAI_LENS','512,2048,8192').split(',')]
E=Engine(None,n_ctx=NCTX,kv_dtype=KV,preloaded=(G,CFG,PARAMS),block=16)
rng=np.random.default_rng(0)
text_ids=E.tokenizer.encode(open('/content/tests/data/long_text.txt').read())
print('corpus tokens',len(text_ids),flush=True)
res=dict(n_ctx=NCTX,kv=KV,rows=[])
for Lp in LENS:
    ids=(text_ids*(Lp//len(text_ids)+1))[:Lp-80]
    row=dict(prompt_tokens=len(ids),prefill_s=[],decode_tok_s=[])
    for trial in range(3):
        E.reset();jax.block_until_ready(E.state)
        t=time.perf_counter();lg=E.feed(ids);lg.block_until_ready();dt=time.perf_counter()-t
        row['prefill_s'].append(dt)
        t=time.perf_counter();n=0
        for tok in E.generate_ids(ids+[int(np.asarray(lg).argmax())],max_new_tokens=65,temperature=0.0,stop_ids=[-1]):
            n+=1
            if n==1:t=time.perf_counter()   # first yield follows a 1-token feed; time the block loop only
        row['decode_tok_s'].append((n-1)/(time.perf_counter()-t))
    row['prefill_tok_s_warm']=len(ids)/min(row['prefill_s'][1:]);row['first_prefill_s_incl_compile']=row['prefill_s'][0]
    row['hbm_in_use_gib']=E.memory()['bytes_in_use']/2**30
    res['rows'].append(row);print(json.dumps({k:(round(v,3) if isinstance(v,float) else [round(x,3) for x in v] if isinstance(v,list) else v) for k,v in row.items()}),flush=True)
res['hbm']=E.memory()
json.dump(res,open('/content/bonsai-results/bench-context-%s-%d.json'%(KV,NCTX),'w'),indent=1)
