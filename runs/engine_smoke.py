exec(open('/content/runs/load.py').read())
from bonsai_tpu.engine import Engine
E=Engine(None,n_ctx=4096,preloaded=(G,CFG,PARAMS))
tk=E.tokenizer
cases=[l.rstrip('\n').replace('\\n','\n') for l in open('/content/tests/data/tokenizer_cases.txt')]
exp=json.load(open('/content/tests/data/tokenizer_expected.json'))
ok=0
for c,e in zip(cases,exp):
    got=tk.encode(c)
    if got==e and tk.decode(got,skip_special=False)==c:ok+=1
    else:print('TOKENIZER MISMATCH',repr(c[:60]),got[:20],e[:20],repr(tk.decode(got,skip_special=False)[:60]),flush=True)
print('tokenizer cases ok',ok,'/',len(cases),flush=True)
t=time.perf_counter();txt=''.join(E.generate('The capital of France is',max_new_tokens=24));print(repr(txt),round(time.perf_counter()-t,1),'s',E.stats,flush=True)
for msgs,kw in (([{'role':'user','content':'What is 84 * 3 / 2? Answer briefly.'}],{}),
                ([{'role':'user','content':'Write a Python function that checks whether a string is a palindrome.'}],{}),
                ([{'role':'user','content':'What is the capital of Australia?'}],{'enable_thinking':True})):
    t=time.perf_counter();txt=''.join(E.chat(msgs,max_new_tokens=200,**kw));dt=time.perf_counter()-t
    print('---',msgs[0]['content'],kw,'\n'+txt,'\n',{k:(round(v,3) if isinstance(v,float) else v) for k,v in E.stats.items()},'decode tok/s',round(E.stats['completion_tokens']/max(E.stats['decode_s'],1e-9),1),flush=True)
