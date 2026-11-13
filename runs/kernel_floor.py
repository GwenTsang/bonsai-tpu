exec(open('/content/runs/load.py').read(),globals())
from bonsai_tpu import kernels as KK
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
def read_kernel(w,s,bn,what):
    k4,n=w.shape;nt=k4*4//1024
    def body(wr,sr,yr,acc):
        @pl.when(pl.program_id(1)==0)
        def _():acc[...]=jnp.zeros((1,bn),jnp.float32)
        a=acc[...]
        if 'w' in what:a=a+jnp.sum(wr[...][:8].astype(jnp.int32),axis=0,keepdims=True).astype(jnp.float32)
        if 's' in what:a=a+jnp.sum(sr[...],axis=0,keepdims=True)
        acc[...]=a
        @pl.when(pl.program_id(1)==nt-1)
        def _():yr[...]=acc[...]
    return pl.pallas_call(body,grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((256,bn),lambda j,g:(g,j)),pl.BlockSpec((8,bn),lambda j,g:(g,j))],
        out_specs=pl.BlockSpec((1,bn),lambda j,g:(0,j)),scratch_shapes=[pltpu.VMEM((1,bn),jnp.float32)],grid=(n//bn,nt)),
        out_shape=jax.ShapeDtypeStruct((1,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')))(w,s)
mats=[]
for L in PARAMS['layers']:
    for k in ('qkvz','qkv','out','ffn_gu','ffn_down'):
        if k in L:mats.append(L[k])
mats.append(PARAMS['head'])
xs={k:jnp.asarray(np.random.default_rng(0).standard_normal(k).astype(np.float32)) for k in (5120,6144,17408)}
def timeit(fn,*a,n=20):
    r=fn(*a);jax.block_until_ready(r);ts=[]
    for _ in range(n):
        t=time.perf_counter();r=fn(*a);jax.block_until_ready(r);ts.append(time.perf_counter()-t)
    return float(np.median(ts)*1e3)
res={}
for what in ('w','s','ws'):
    for target in (2176,4352,8704):
        @jax.jit
        def f(mats):
            acc=jnp.zeros(8)
            for m in mats:   # serial dependency through a tiny perturbation of scales is avoided; chain via acc only
                acc=acc+read_kernel(m['w'],m['s'],KK.pick_bn(m['w'].shape[1],target),what)[0,:8]
            return acc
        res['read_%s_bn%d'%(what,target)]=timeit(f,mats);print(what,target,res['read_%s_bn%d'%(what,target)],flush=True)
for target in (1088,2176,4352,8704):
    @jax.jit
    def g(mats,xs):
        acc=jnp.zeros(8)
        for m in mats:
            x=xs[m['w'].shape[0]*4]+acc[0]*1e-20
            xb,xsum,alpha=KK.prep_gemv(x)
            acc=acc+KK._gemv_kernel(xb,xsum,m['w'],m['s'],KK.pick_bn(m['w'].shape[1],target))[0,:8]
        return acc
    res['gemv_serial_bn%d'%target]=timeit(g,mats,xs);print('gemv serial',target,res['gemv_serial_bn%d'%target],flush=True)
json.dump(res,open('/content/bonsai-results/kernel-floor.json','w'),indent=1)
