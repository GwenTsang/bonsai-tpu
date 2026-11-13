# End-to-end agentic test on the runtime: server thread + HTTP client; tool loop, prefix reuse, checkpoints.
exec(open('/content/runs/load.py').read(),globals())
import threading,urllib.request
from bonsai_tpu.engine import Engine
from bonsai_tpu.server import serve
E=Engine(None,n_ctx=32768,kv_dtype='bfloat16',preloaded=(G,CFG,PARAMS),block=8)
print('warmup s',round(E.warmup(),1),flush=True)
import socketserver;socketserver.TCPServer.allow_reuse_address=True
PORT=8000
for PORT in range(8000,8010):
    try:HTTPD=serve(E,'127.0.0.1',PORT);break
    except OSError:pass
print('port',PORT,flush=True);threading.Thread(target=HTTPD.serve_forever,daemon=True).start()
def post(body,path='/v1/chat/completions'):
    t=time.perf_counter()
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:%d'%PORT+''+path,json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=900)
    d=json.loads(r.read());d['_wall']=round(time.perf_counter()-t,3);return d
tools=[{'type':'function','function':{'name':'read_file','description':'Read a text file and return its contents.','parameters':{'type':'object','properties':{'path':{'type':'string','description':'Absolute path'}},'required':['path']}}},
       {'type':'function','function':{'name':'run_python','description':'Run a Python expression and return the result.','parameters':{'type':'object','properties':{'expression':{'type':'string'}},'required':['expression']}}}]
files={'/data/config.txt':'threshold=37\nscale=12\nname=bonsai\n'+('# filler line for context\n'*400)}
msgs=[{'role':'system','content':'You are a careful coding agent. Use the tools when needed.'},
      {'role':'user','content':'Read /data/config.txt, then compute threshold*scale with the python tool, and tell me the result.'}]
log=[]
for turn in range(6):
    d=post({'model':'x','messages':msgs,'tools':tools,'temperature':0,'max_tokens':400})
    m=d['choices'][0]['message'];tm=d['timings']
    row=dict(turn=turn,finish=d['choices'][0]['finish_reason'],prompt=d['usage']['prompt_tokens'],cached=d['usage']['prompt_tokens_details']['cached_tokens'],fed=tm['prompt_n'],prefill_ms=round(tm['prompt_ms'],1),gen=tm['predicted_n'],tok_s=round(tm.get('predicted_per_second',0),1),wall=d['_wall'])
    log.append(row);print(json.dumps(row),flush=True)
    print('   content:',repr(m.get('content','')[:200]),'calls:',[(c['function']['name'],c['function']['arguments']) for c in m.get('tool_calls',[])],flush=True)
    msgs.append(m)
    if not m.get('tool_calls'):break
    for c in m['tool_calls']:
        a=json.loads(c['function']['arguments'])
        if c['function']['name']=='read_file':out=files.get(a.get('path'),'ERROR: no such file')
        else:
            try:out=str(eval(a.get('expression','')))
            except Exception as e:out='ERROR: '+repr(e)
        msgs.append({'role':'tool','tool_call_id':c['id'],'content':out})
# same conversation resent unchanged + new user turn (typical agent follow-up)
msgs.append({'role':'user','content':'Now also add 5 to that result. Answer with just the number.'})
d=post({'model':'x','messages':msgs,'tools':tools,'temperature':0,'max_tokens':200});tm=d['timings']
print('follow-up',json.dumps(dict(prompt=d['usage']['prompt_tokens'],cached=tm['cache_n'],fed=tm['prompt_n'],prefill_ms=round(tm['prompt_ms'],1),wall=d['_wall'])),repr(d['choices'][0]['message'].get('content'))[:200],[(c['function']['name'],c['function']['arguments']) for c in d['choices'][0]['message'].get('tool_calls',[])],flush=True)
# edit in the middle of history (forces checkpoint rewind)
msgs2=[dict(m) for m in msgs];msgs2[-1]={'role':'user','content':'Actually, what was the name field in the file?'}
d=post({'model':'x','messages':msgs2,'tools':tools,'temperature':0,'max_tokens':200});tm=d['timings']
print('edited-last-turn',json.dumps(dict(prompt=d['usage']['prompt_tokens'],cached=tm['cache_n'],fed=tm['prompt_n'],prefill_ms=round(tm['prompt_ms'],1),wall=d['_wall'])),repr(d['choices'][0]['message'].get('content'))[:200],flush=True)
# streaming + thinking
req=urllib.request.Request('http://127.0.0.1:%d'%PORT+'/v1/chat/completions',json.dumps({'messages':[{'role':'user','content':'Is 91 prime? Answer in one sentence.'}],'stream':True,'enable_thinking':True,'temperature':0,'max_tokens':300}).encode(),{'Content-Type':'application/json'})
t=time.perf_counter();first=None;rs='';cs=''
for line in urllib.request.urlopen(req,timeout=900):
    line=line.decode().strip()
    if not line.startswith('data: ') or line=='data: [DONE]':continue
    ch=json.loads(line[6:])['choices'][0].get('delta',{})
    if (ch.get('content') or ch.get('reasoning_content')) and first is None:first=time.perf_counter()-t
    rs+=ch.get('reasoning_content') or '';cs+=ch.get('content') or ''
print('stream ttft_s',round(first,3),'reasoning:',repr(rs[:150]),'content:',repr(cs[:200]),flush=True)
print(json.dumps(json.loads(urllib.request.urlopen('http://127.0.0.1:%d'%PORT+'/metrics').read()),default=str)[:600],flush=True)
json.dump(log,open('/content/bonsai-results/agent-loop.json','w'),indent=1)
