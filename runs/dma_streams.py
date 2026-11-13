exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu import kernels as KK
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
def read_kernel(w,bn,streams,rows):
    k4,n=w.shape;nsteps=k4//(rows*streams)
    def body(*refs):
        ins=refs[:streams];yr=refs[streams];acc=refs[streams+1]
        @pl.when(pl.program_id(1)==0)
        def _():acc[...]=jnp.zeros((1,bn),jnp.float32)
        a=acc[...]
        for r in ins:a=a+jnp.sum(r[...][:8].astype(jnp.int32),axis=0,keepdims=True).astype(jnp.float32)
        acc[...]=a
        @pl.when(pl.program_id(1)==nsteps-1)
        def _():yr[...]=acc[...]
    specs=[pl.BlockSpec((rows,bn),(lambda j,g,i=i:(streams*g+i,j))) for i in range(streams)]
    return pl.pallas_call(body,grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,in_specs=specs,
        out_specs=pl.BlockSpec((1,bn),lambda j,g:(0,j)),scratch_shapes=[pltpu.VMEM((1,bn),jnp.float32)],grid=(n//bn,nsteps)),
        out_shape=jax.ShapeDtypeStruct((1,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')))(*([w]*streams))
mats=[]
for L in PARAMS['layers']:
    for k in ('qkvz','qkv','out','ffn_gu','ffn_down'):
        if k in L:mats.append(L[k])
mats.append(PARAMS['head'])
def timeit(fn,*a,n=15):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return float(np.median(ts)*1e3)
res={};wb=sum(m['w'].size for m in mats)
for streams,rows,target in ((1,128,4352),(2,128,4352),(1,256,4352),(1,64,4352),(2,64,4352),(4,64,4352),(1,256,2176),(2,128,2176),(2,128,8704)):
    try:
        @jax.jit
        def f(mats):
            acc=jnp.zeros(8)
            for m in mats:acc=acc+read_kernel(m['w'],KK.pick_bn(m['w'].shape[1],target),streams,rows)[0,:8]
            return acc
        t=timeit(f,mats);res['streams%d_rows%d_bn%d'%(streams,rows,target)]=t;print(streams,rows,target,round(t,3),'ms',round(wb/t/1e6),'GB/s (weights only)',flush=True)
    except Exception as e:print(streams,rows,target,'ERR',repr(e)[:200],flush=True)
json.dump(res,open('/content/bonsai-results/dma-streams.json','w'),indent=1)
