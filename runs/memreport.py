# Uses the live server engine `E` (does not reload anything).
import importlib,bonsai_tpu.engine as _EG
from bonsai_tpu.engine import Engine as _Eng
E.memory_report=_Eng.memory_report.__get__(E)
r=E.memory_report()
print(json.dumps({k:(round(v/2**30,3) if isinstance(v,int) else {a:round(b/2**30,3) for a,b in v.items()}) for k,v in r.items()},indent=1),flush=True)
print('n_ctx',E.n_ctx,'kv',E.kv_dtype,'checkpoints',[s['pos'] for s in E.snapshots],'programs',list(E._programs),flush=True)
json.dump(r,open('/content/bonsai-results/memory-report.json','w'),indent=1)
