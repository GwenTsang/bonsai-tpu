exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.engine import Engine
from bonsai_tpu import engine as EG
E=Engine(None,n_ctx=16384,kv_dtype='bfloat16',preloaded=(G,CFG,PARAMS),block=8)
print('warmup s',round(E.warmup(),1),flush=True)
code=open('/content/bonsai_tpu/server.py').read()[:6000]
tasks={
 'copy_edit':[{'role':'user','content':'Here is a Python file:\n\n```python\n'+code+'\n```\n\nRewrite the whole file exactly as it is, but rename the function `_sampling` to `sampling_params` everywhere. Output only the code block.'}],
 'essay':[{'role':'user','content':'Write a 300-word essay about tide pools.'}],
 'summarize':[{'role':'user','content':'Here is a Python file:\n\n```python\n'+code+'\n```\n\nExplain in detail what each function does, quoting the relevant lines.'}],
}
res={}
for name,msgs in tasks.items():
    ids=E.tokenizer.encode(E.tokenizer.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False))
    outs={}
    for spec in (False,True):
        E.speculate=spec;E.spec_stats={'rounds':0,'drafted':0,'accepted':0,'misses':0}
        E.reset();t=time.perf_counter();toks=list(E.generate_ids(ids,max_new_tokens=1500,temperature=0.0))
        st=dict(E.stats);outs[spec]=toks
        row=dict(task=name,spec=spec,prompt=len(ids),gen=len(toks),decode_s=round(st['decode_s'],2),tok_s=round(len(toks)/st['decode_s'],1),**E.spec_stats)
        res[name+('_spec' if spec else '_base')]=row;print(json.dumps(row),flush=True)
    a,b=outs[False],outs[True];same=sum(1 for x,y in zip(a,b) if x==y)
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    print('   identical prefix',first,'of',len(a),len(b),'| tail:',repr(E.tokenizer.decode(b[-40:])),flush=True)
    res[name+'_identical_prefix']=first
json.dump(res,open('/content/bonsai-results/speculation.json','w'),indent=1)
