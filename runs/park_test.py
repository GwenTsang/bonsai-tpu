exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu.engine import Engine
E=Engine(None,n_ctx=32768,kv_dtype='bfloat16',preloaded=(G,CFG,PARAMS))
print('warmup s',round(E.warmup(),1),flush=True)
tk=E.tokenizer
text=open('/content/tests/data/long_text.txt').read()
def chat(msgs):return tk.encode(tk.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False))
main=[{'role':'system','content':'You are an archivist.'},{'role':'user','content':text[:110000]+'\n\nWhich instrument count is mentioned (the number of copper instruments)? One short sentence.'}]
side=[{'role':'user','content':'Write a 4-word title for a chat about tide tables.'}]
res=[]
def run(label,msgs,n=24):
    ids=chat(msgs);t=time.perf_counter();out=tk.decode(list(E.generate_ids(ids,max_new_tokens=n,temperature=0.0)))
    st={k:(round(v,3) if isinstance(v,float) else v) for k,v in E.stats.items()};st.update(label=label,wall=round(time.perf_counter()-t,3),prompt=len(ids),parked_sessions=len(E.parked),parked_gib=round(sum(p['bytes'] for p in E.parked)/2**30,2))
    res.append(st);print(json.dumps(st),'|',repr(out[:80]),flush=True);return out
a1=run('main first',main)
run('side request',side)
a2=run('main again (restored from host RAM)',main)
run('side again',side)
main2=main+[{'role':'assistant','content':a1.strip()},{'role':'user','content':'And which lamps are mentioned? One short sentence.'}]
run('main continued',main2)
print('same answer after restore:',a1==a2,flush=True)
json.dump(res,open('/content/bonsai-results/parking.json','w'),indent=1)
