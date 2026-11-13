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
print('HBM',{k:round(v/2**30,3) for k,v in jax.devices()[0].memory_stats().items() if 'bytes' in k},flush=True)
