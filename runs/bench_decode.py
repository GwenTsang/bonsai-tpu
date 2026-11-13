exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.engine import Engine
import os
BLOCK=int(os.environ.get('BONSAI_BLOCK','8'));NCTX=int(os.environ.get('BONSAI_NCTX','4096'));KV=os.environ.get('BONSAI_KV','float32')
E=Engine(None,n_ctx=NCTX,kv_dtype=KV,preloaded=(G,CFG,PARAMS),block=BLOCK)
msgs=[{'role':'user','content':'Write a detailed essay about the history of the Roman Empire, covering its founding, expansion, golden age, crises and fall.'}]
prompt=E.tokenizer.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False)
ids=E.tokenizer.encode(prompt)
res=dict(block=BLOCK,n_ctx=NCTX,kv=KV,trials=[])
for trial in range(4):
    E.reset();t=time.perf_counter();stamps=[];toks=[]
    for tok in E.generate_ids(ids,max_new_tokens=512,temperature=0.0):
        stamps.append(time.perf_counter()-t);toks.append(tok)
    st=dict(E.stats);n=len(toks)
    # steady state: exclude first block (may include compilation on trial 0)
    k=BLOCK+1
    st['ttft_s']=stamps[0];st['steady_tok_s']=(n-k)/(stamps[-1]-stamps[k-1]) if n>k+8 else None
    st['overall_decode_tok_s']=st['completion_tokens']/st['decode_s']
    res['trials'].append(st);print(json.dumps({a:(round(b,4) if isinstance(b,float) else b) for a,b in st.items()}),flush=True)
print(E.tokenizer.decode(toks)[:300].replace('\n',' '))
res['hbm']=E.memory();print(res['hbm'])
json.dump(res,open('/content/bonsai-results/bench-decode-b%d-%s.json'%(BLOCK,KV),'w'),indent=1)
