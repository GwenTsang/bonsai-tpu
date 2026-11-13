# Shared preamble for development runs inside the persistent TPU kernel: free what earlier runs left behind,
# then load the model once per kernel lifetime (globals persist across runs).
import time,json,gc,ctypes,numpy as np,jax,jax.numpy as jnp
from bonsai_tpu.gguf import GGUF
from bonsai_tpu import model as M
for _o in [o for o in gc.get_objects() if type(o).__name__=='ThreadingHTTPServer']:   # orphaned dev servers pin HBM
    try:_o.shutdown();_o.server_close();_o.RequestHandlerClass=None
    except Exception as _e:print('server shutdown',_e)
_o=None
for _n in [n for n in list(globals()) if n not in ('PARAMS','CFG','G','LOAD_S','RELOAD','M','GGUF') and not n.startswith('_') and type(globals()[n]).__name__ not in ('module','function','type','builtin_function_or_method')]:
    del globals()[_n]
if 'PARAMS' in globals() and (PARAMS is None or globals().get('RELOAD') or PARAMS['head']['s'].dtype!=jnp.int16):
    del PARAMS                       # stale resident format
gc.collect()
jax.clear_caches()                   # each cached TPU executable pins ~1.1 GiB of host RAM
ctypes.CDLL('libc.so.6').malloc_trim(0)
if 'PARAMS' not in globals():
    _t=time.perf_counter()
    from bonsai_tpu.defaults import resolve_model
    G=GGUF(resolve_model())
    CFG,PARAMS=M.load_params(G,log=None)
    jax.block_until_ready(PARAMS)
    LOAD_S=time.perf_counter()-_t
    print('LOAD_SECONDS',round(LOAD_S,1),flush=True)
    RELOAD=False
else:
    CFG=M.Config(G.metadata)
print('HBM',{k:round(v/2**30,3) for k,v in jax.devices()[0].memory_stats().items() if 'bytes' in k},flush=True)
