"""Experimental tiled PQ2 matmul with local BF16 reconstruction in VMEM."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def pack_words(codes):
    """[N,K] ternary codes -> [K/16,N] uint32 words, losslessly."""
    n,k=codes.shape
    if k%128 or not np.isin(codes,[-1,0,1]).all():
        raise ValueError('Expected ternary matrix with K divisible by 128')
    d=(codes.astype(np.int32)+1).astype(np.uint32).reshape(n,k//16,16)
    w=np.zeros((n,k//16),np.uint32)
    for i in range(16):w |= d[...,i] << (2*i)
    return np.ascontiguousarray(w.T)


def packed_matmul(x, words, scales, *, bm=32, bk=1024, bn=256):
    """x[M,K], words[K/16,N], scales[K/128,N]; x is already rotated."""
    m,k=x.shape;n=words.shape[1]
    if k%bk or n%bn or bk%1024:
        raise ValueError('Tiles must divide K,N; scale tiling requires K tiles divisible by 1024')
    padded_m=(m+bm-1)//bm*bm
    x=jnp.pad(x,((0,padded_m-m),(0,0))).astype(jnp.bfloat16)
    def body(xr,wr,sr,yr,acc):
        @pl.when(pl.program_id(2)==0)
        def init():acc[...] = jnp.zeros((bm,bn),jnp.float32)
        shifts=jnp.arange(16,dtype=jnp.uint32)*2
        q=((wr[...][:,None,:]>>shifts[None,:,None])&3).astype(jnp.int32)-1
        q=q.reshape(bk//128,128,bn).astype(jnp.float32)
        w=(q*sr[...][:,None,:]).reshape(bk,bn).astype(jnp.bfloat16)
        acc[...] += jax.lax.dot_general(xr[...],w,(((1,),(0,)),((),())),preferred_element_type=jnp.float32)
        @pl.when(pl.program_id(2)==k//bk-1)
        def store():yr[...] = acc[...]
    y=pl.pallas_call(body,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm,bk),lambda i,j,g:(i,g)),
                      pl.BlockSpec((bk//16,bn),lambda i,j,g:(g,j)),
                      pl.BlockSpec((bk//128,bn),lambda i,j,g:(g,j))],
            out_specs=pl.BlockSpec((bm,bn),lambda i,j,g:(i,j)),
            scratch_shapes=[pltpu.VMEM((bm,bn),jnp.float32)],
            grid=(padded_m//bm,n//bn,k//bk)),
        out_shape=jax.ShapeDtypeStruct((padded_m,n),jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel','arbitrary')))(x,words,scales)
    return y[:m]
