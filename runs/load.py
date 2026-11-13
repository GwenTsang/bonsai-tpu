# Shared preamble: load the model once per kernel lifetime (globals persist across colab exec calls).
import time,json,numpy as np,jax,jax.numpy as jnp
from bonsai_tpu.gguf import GGUF
from bonsai_tpu import model as M
if 'PARAMS' not in globals() or globals().get('RELOAD'):
    t=time.perf_counter()
    G=GGUF('/content/weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
    CFG,PARAMS=M.load_params(G)
    jax.block_until_ready(PARAMS)
    LOAD_S=time.perf_counter()-t
    print('LOAD_SECONDS',round(LOAD_S,1),flush=True)
    RELOAD=False
else:
    CFG=M.Config(G.metadata)
import gc
for _n in [n for n in list(globals()) if n not in ('PARAMS','CFG','G','LOAD_S','RELOAD','M','GGUF') and not n.startswith('_') and type(globals()[n]).__name__ not in ('module','function','type','builtin_function_or_method')]:
    del globals()[_n]
gc.collect()
jax.clear_caches()   # each cached TPU executable pins ~1.1 GiB of host RAM
import ctypes;ctypes.CDLL('libc.so.6').malloc_trim(0)
print('HBM',{k:round(v/2**30,3) for k,v in jax.devices()[0].memory_stats().items() if 'bytes' in k},flush=True)
