"""Run actual-weight codec and Hadamard test vectors on the attached TPU."""
import tarfile, sys, json, time
with tarfile.open('/content/bonsai-kernels.tar.gz') as f:f.extractall('/content',filter='data')
sys.path.insert(0,'/content')
import numpy as np
import jax
import jax.numpy as jnp
from bonsai_tpu.quantization import decode_codes
from bonsai_tpu.hadamard import activation_transform
vectors=np.load('/content/real-weight-vectors.npz')
results=[]
for key in vectors.files:
    if not key.endswith('_payload'):continue
    prefix=key[:-8]
    p=jax.device_put(vectors[key]);s=jax.device_put(vectors[prefix+'_scales'])
    k=vectors[prefix+'_decoded'].shape[-1]
    f=jax.jit(lambda p,s:decode_codes(p,143,jnp).astype(jnp.float32)*s[:,None])
    got=np.asarray(f(p,s)).reshape(4,k)
    np.testing.assert_array_equal(got,vectors[prefix+'_decoded'])
    x=jax.device_put(vectors[prefix+'_x']);signs=jax.device_put(vectors[prefix+'_signs'])
    gdn='ssm_out' in prefix
    h=jax.jit(lambda x,s:activation_transform(x,s,gdn=gdn,xp=jnp))
    transformed=np.asarray(h(x,signs))
    error=float(np.max(np.abs(transformed-vectors[prefix+'_transformed'])))
    np.testing.assert_allclose(transformed,vectors[prefix+'_transformed'],atol=2e-6,rtol=1e-5)
    results.append(dict(tensor=prefix,decode_max_error=0.,hadamard_max_error=error,device=str(p.device)))
    print(json.dumps(results[-1]),flush=True)
with open('/content/bonsai-results/tpu-codecs.json','w') as f:json.dump(results,f,indent=2)
print('TPU_CODECS_COMPLETE',flush=True)
