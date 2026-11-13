# Headline benchmark: warm decode / prefill / TTFT with variability, one JSON.
exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.engine import Engine
import os,subprocess
NCTX=int(os.environ.get('BONSAI_NCTX','65536'))
E=Engine(None,n_ctx=NCTX,kv_dtype='bfloat16',preloaded=(G,CFG,PARAMS))
res=dict(model='Ternary-Bonsai-2-27B PTQ1_0 (rev 839bd57)',weight_format='B4 2-bit + int16 half scales (from PTQ1_0)',n_ctx=NCTX,kv='bfloat16',batch=1,block=E.block)
res['warmup_s']=round(E.warmup(),1);res['load_s']=round(LOAD_S,1)
tk=E.tokenizer
msgs=[{'role':'user','content':'Write a detailed essay about the history of the Roman Empire, covering its founding, expansion, golden age, crises and fall.'}]
ids=tk.encode(tk.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False))
E.speculate=False
dec=[];ttft=[]
for trial in range(5):
    E.reset();t=time.perf_counter();stamps=[]
    for tok in E.generate_ids(ids,max_new_tokens=512,temperature=0.0,stop_ids=[-1]):stamps.append(time.perf_counter()-t)
    ttft.append(stamps[0]);dec.append((len(stamps)-64)/(stamps[-1]-stamps[63]))
res['decode_tok_s_greedy']=dict(trials=dec,median=float(np.median(dec)),min=min(dec),max=max(dec),tokens=512)
res['ttft_s_37tok_prompt']=dict(trials=ttft,median=float(np.median(ttft)))
dec=[]
for trial in range(3):
    E.reset();t=time.perf_counter();stamps=[]
    for tok in E.generate_ids(ids,max_new_tokens=512,temperature=0.7,top_k=20,top_p=0.95,seed=trial,stop_ids=[-1]):stamps.append(time.perf_counter()-t)
    dec.append((len(stamps)-64)/(stamps[-1]-stamps[63]))
res['decode_tok_s_sampling']=dict(trials=dec,median=float(np.median(dec)))
E.speculate=True
text_ids=tk.encode(open('/content/tests/data/long_text.txt').read())
res['prefill']=[]
for Lp in (512,2048,8192,16384,32768):
    ids2=(text_ids*(Lp//len(text_ids)+1))[:Lp];ts=[];d=[]
    for trial in range(3):
        E.reset();jax.block_until_ready(E.state);t=time.perf_counter();lg=E.feed(ids2);lg.block_until_ready();ts.append(time.perf_counter()-t)
        E.speculate=False;n=0
        for tok in E.generate_ids(ids2+[5],max_new_tokens=130,temperature=0.0,stop_ids=[-1]):
            n+=1
            if n==2:t=time.perf_counter()
        d.append((n-2)/(time.perf_counter()-t));E.speculate=True
    row=dict(prompt_tokens=Lp,prefill_s=ts,prefill_tok_s=Lp/float(np.median(ts)),decode_tok_s_after=float(np.median(d)))
    res['prefill'].append(row);print(json.dumps(row),flush=True)
res['memory']=E.memory_report()
res['commit']=os.environ.get('BONSAI_COMMIT','')
print(json.dumps({k:v for k,v in res.items() if k not in ('prefill','memory')}),flush=True)
json.dump(res,open('/content/bonsai-results/final-bench.json','w'),indent=1)
