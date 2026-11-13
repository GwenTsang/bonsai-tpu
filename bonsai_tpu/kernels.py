"""TPU kernels over the engine's resident "B4" ternary layout.

B4 layout for a matrix W[N, K] (N outputs, K inputs, K % 1024 == 0, scale groups of 128 inputs):
  w: uint8 [K/4, N].  Byte row r = 32*g + i; bit pair j (bits 2j, 2j+1) holds code(k, n) = t + 1 in {0,1,2}
     for input k = 128*g + 32*j + i.  One whole-tile shift+mask therefore yields 4 contiguous sub-blocks.
  s: float32 [K/128, N] group scales.
All unpacking runs in 8-bit vector lanes (bitwise ops only; Mosaic has no i8 arithmetic), the MXU consumes
int8 directly, and per-group partial sums stay exact integers until the single float scale multiply.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BK = 1024          # inputs per K tile
NG = BK // 128     # scale groups per K tile
XMAX = 32639.0     # largest |q| such that the rounded high byte stays within int8


def quantize16(x):
    """float [..., K] -> int32 q with |q| <= XMAX, plus the per-row multiplier alpha (q ~= x * alpha)."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    alpha = XMAX / jnp.maximum(amax, 1e-30)
    return jnp.round(x * alpha).astype(jnp.int32), alpha


def prep_gemv(x):
    """x f32 [K] -> block-diagonal int8 operand [(K/BK)*2*NG, BK], per-group sums f32 [K/128], alpha.

    Row g of each tile carries only scale-group g (high bytes in rows 0..NG-1, low bytes in NG..2NG-1), so
    one MXU product returns every group's partial sum separately.  Columns follow the (j, g, i) order that
    the kernel's concatenated shift planes produce.
    """
    k = x.shape[0]
    nt = k // BK
    q, alpha = quantize16(x)
    hi = (q + 128) >> 8
    lo = q - (hi << 8)
    eye = jnp.eye(NG, dtype=jnp.int32)[None, :, None, :, None]

    def bd(v):
        v = v.reshape(nt, NG, 4, 32).transpose(0, 2, 1, 3)       # [tile, j, g, i]
        return (eye * v[:, None]).reshape(nt, NG, BK)
    xb = jnp.concatenate([bd(hi), bd(lo)], axis=1).reshape(nt * 2 * NG, BK).astype(jnp.int8)
    xsum = q.reshape(k // 128, 128).sum(-1).astype(jnp.float32)
    return xb, xsum, alpha[0]


def _gemv_kernel(xb, w, s, bn):
    k4, n = w.shape
    nt = k4 * 4 // BK

    def body(xr, wr, sr, yr, acc):
        @pl.when(pl.program_id(1) == 0)
        def _():
            acc[...] = jnp.zeros((1, bn), jnp.float32)
        wv = wr[...]
        t = jnp.concatenate([wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6], axis=0).astype(jnp.int8)
        p = jax.lax.dot_general(xr[...], t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
        exact = (p[:NG] << 8) + p[NG:]
        acc[...] += jnp.sum(exact.astype(jnp.float32) * sr[...], axis=0, keepdims=True)

        @pl.when(pl.program_id(1) == nt - 1)
        def _():
            yr[...] = acc[...]

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((2 * NG, BK), lambda j, g: (g, 0)),
                      pl.BlockSpec((BK // 4, bn), lambda j, g: (g, j)),
                      pl.BlockSpec((NG, bn), lambda j, g: (g, j))],
            out_specs=pl.BlockSpec((1, bn), lambda j, g: (0, j)),
            scratch_shapes=[pltpu.VMEM((1, bn), jnp.float32)],
            grid=(n // bn, nt)),
        out_shape=jax.ShapeDtypeStruct((1, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary')))(xb, w, s)


def pick_bn(n, target=4352):
    """Largest multiple of 128 dividing n that does not exceed target."""
    best = None
    for bn in range(128, min(n, target) + 1, 128):
        if n % bn == 0:
            best = bn
    if best is None:
        raise ValueError('N=%d has no legal lane tile' % n)
    return best


def gemv(x, mat, prepared=None):
    """y[N] = W x for one activation vector; `prepared` reuses prep_gemv(x) across matrices sharing x."""
    xb, xsum, alpha = prepared if prepared is not None else prep_gemv(x)
    w, s = mat['w'], mat['s']
    y = _gemv_kernel(xb, w, s, pick_bn(w.shape[1]))[0]
    # codes are t+1: remove the +1 offset, sum_g s[g,n] * sum_{k in g} q[k]
    return (y - jnp.matmul(xsum, s, precision=jax.lax.Precision.HIGHEST)) / alpha
