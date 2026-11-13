"""Run via colab exec -s bonsai-tpu -f scripts/hardware_baseline.py."""
import json, os, platform, time, statistics
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import SingleDeviceSharding

def main():
    device = jax.devices('tpu')[0]
    result = dict(python=platform.python_version(), jax=jax.__version__,
                  devices=[str(d) for d in jax.devices()], device_kind=device.device_kind,
                  cpu_count=os.cpu_count(), memory_before=device.memory_stats(), measurements=[])
    print(json.dumps(result), flush=True)
    def bench(name, fn, n=7, **meta):
        start=time.perf_counter(); out=fn(); jax.block_until_ready(out)
        cold=time.perf_counter()-start
        samples=[]
        for _ in range(n):
            start=time.perf_counter(); out=fn(); jax.block_until_ready(out)
            samples.append(time.perf_counter()-start)
        row=dict(name=name,cold_seconds=cold,seconds=samples,median_seconds=statistics.median(samples),**meta)
        result['measurements'].append(row); print(json.dumps(row),flush=True)
        return out
    for dtype in (jnp.bfloat16,jnp.int8):
        a=jax.device_put(np.ones((2048,2048),np.int8),device).astype(dtype)
        f=jax.jit(lambda x,y: jax.lax.dot_general(x,y,(((1,),(0,)),((),())),preferred_element_type=jnp.int32 if dtype==jnp.int8 else jnp.float32))
        out=bench('matmul_'+str(dtype),lambda:f(a,a),shape=[2048]*3)
        assert float(out[0,0])==2048
        del a,out,f
    dev=SingleDeviceSharding(device,memory_kind='device')
    host=SingleDeviceSharding(device,memory_kind='pinned_host')
    for size in (16*1024**2,128*1024**2):
        a=jax.device_put(np.arange(size//4,dtype=np.float32),host)
        b=bench('pinned_host_to_device',lambda:jax.device_put(a,dev),bytes=size)
        c=bench('device_to_pinned_host',lambda:jax.device_put(b,host),bytes=size)
        del a,b,c
    result['memory_after']=device.memory_stats()
    result['host_meminfo']=open('/proc/meminfo').read()
    os.makedirs('/content/bonsai-results',exist_ok=True)
    with open('/content/bonsai-results/hardware-baseline.json','w') as f:json.dump(result,f,indent=2)
    print('BASELINE_COMPLETE',flush=True)
main()
