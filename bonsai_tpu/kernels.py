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
    # per-group sum of q: removes the +1 code offset exactly, in integers, inside the kernel
    xsum = jnp.broadcast_to(q.reshape(k // 128, 128).sum(-1)[:, None], (k // 128, 128))
    return xb, xsum, alpha[0]


def _gemv_kernel(xb, xsum, w, s, bn):
    k4, n = w.shape
    nt = k4 * 4 // BK

    def body(xr, or_, wr, sr, yr, acc):
        @pl.when(pl.program_id(1) == 0)
        def _():
            acc[...] = jnp.zeros((1, bn), jnp.float32)
        wv = wr[...]
        t = jnp.concatenate([wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6], axis=0).astype(jnp.int8)
        p = jax.lax.dot_general(xr[...], t, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
        exact = (p[:NG] << 8) + p[NG:] - or_[...][:, :1]
        acc[...] += jnp.sum(exact.astype(jnp.float32) * sr[...], axis=0, keepdims=True)

        @pl.when(pl.program_id(1) == nt - 1)
        def _():
            yr[...] = acc[...]

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((2 * NG, BK), lambda j, g: (g, 0)),
                      pl.BlockSpec((NG, 128), lambda j, g: (g, 0)),
                      pl.BlockSpec((BK // 4, bn), lambda j, g: (g, j)),
                      pl.BlockSpec((NG, bn), lambda j, g: (g, j))],
            out_specs=pl.BlockSpec((1, bn), lambda j, g: (0, j)),
            scratch_shapes=[pltpu.VMEM((1, bn), jnp.float32)],
            grid=(n // bn, nt)),
        out_shape=jax.ShapeDtypeStruct((1, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary')))(xb, xsum, w, s)


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
    return _gemv_kernel(xb, xsum, w, s, pick_bn(w.shape[1]))[0] / alpha


def _gemm_kernel(x, w, s, bm, bn):
    m, k = x.shape
    n = w.shape[1]
    nt = k // BK
    q = BK // 4

    def body(xr, wr, sr, yr, acc):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc[...] = jnp.zeros((bm, bn), jnp.float32)
        wv, sv, xv = wr[...], sr[...], xr[...]
        srep = jnp.concatenate([jnp.broadcast_to(sv[g:g + 1], (32, bn)) for g in range(NG)], axis=0)
        tot = acc[...]
        for j, plane in enumerate((wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6)):
            wj = ((plane.astype(jnp.int32).astype(jnp.float32) - 1.0) * srep).astype(jnp.bfloat16)
            tot = tot + jax.lax.dot_general(xv[:, j * q:(j + 1) * q], wj, (((1,), (0,)), ((), ())),
                                            preferred_element_type=jnp.float32)
        acc[...] = tot

        @pl.when(pl.program_id(2) == nt - 1)
        def _():
            yr[...] = acc[...]

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm, BK), lambda i, j, g: (i, g)),
                      pl.BlockSpec((q, bn), lambda i, j, g: (g, j)),
                      pl.BlockSpec((NG, bn), lambda i, j, g: (g, j))],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, g: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, nt)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel', 'arbitrary')))(x, w, s)


def gemm(x, mat, bm_max=256):
    """y[M, N] = x[M, K] W^T for prompt processing.  Weights are reconstructed as BF16 (scale folded) in
    VMEM; activations are BF16; accumulation is FP32."""
    m, k = x.shape
    w, s = mat['w'], mat['s']
    bm = min(bm_max, -(-m // 8) * 8)
    mp = -(-m // bm) * bm
    xp = x.reshape(m, k // BK, NG, 4, 32).transpose(0, 1, 3, 2, 4).reshape(m, k).astype(jnp.bfloat16)
    if mp != m:
        xp = jnp.pad(xp, ((0, mp - m), (0, 0)))
    return _gemm_kernel(xp, w, s, bm, pick_bn(w.shape[1], 2176))[:m]


def _gemm_perm():
    """Column order of the GEMM kernel's shift planes: position (j, g, i) holds input 128g + 32j + i."""
    import numpy as np
    return np.arange(BK).reshape(NG, 4, 32).transpose(1, 0, 2).reshape(-1)


def rotate_for_gemm(x, signs, bm_max=256):
    """Fused prefill pre-processing in VMEM: sign flip, normalized 1024-block Hadamard (two exact BF16 MXU
    passes on a bf16-split activation), permutation into the GEMM kernel's column order and the BF16 cast.
    x f32 [M, K] -> bf16 [M', K] with M' = M rounded up to the row tile (feed to gemm_rotated)."""
    import numpy as np
    from .hadamard import _H1024
    m, k = x.shape
    bm = min(bm_max, -(-m // 16) * 16)
    mp = -(-m // bm) * bm
    if mp != m:
        x = jnp.pad(x, ((0, mp - m), (0, 0)))
    h = jnp.asarray(_H1024[:, _gemm_perm()] * np.float32(1024 ** -0.5), dtype=jnp.bfloat16)   # +-1/32: exact

    def body(xr, sr, hr, yr):
        xs = xr[...] * sr[...]
        a = xs.astype(jnp.bfloat16)
        b = (xs - a.astype(jnp.float32)).astype(jnp.bfloat16)
        hv = hr[...]
        dims = (((1,), (0,)), ((), ()))
        y = (jax.lax.dot_general(a, hv, dims, preferred_element_type=jnp.float32)
             + jax.lax.dot_general(b, hv, dims, preferred_element_type=jnp.float32))
        yr[...] = y.astype(jnp.bfloat16)

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm, BK), lambda i, g: (i, g)),
                      pl.BlockSpec((1, BK), lambda i, g: (0, g)),
                      pl.BlockSpec((BK, BK), lambda i, g: (0, 0))],
            out_specs=pl.BlockSpec((bm, BK), lambda i, g: (i, g)),
            grid=(mp // bm, k // BK)),
        out_shape=jax.ShapeDtypeStruct((mp, k), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel')))(
            x, signs.reshape(1, k), h)


def gemm_rotated(xp, mat, m, bm_max=256):
    """y[m, N] for activations already produced by rotate_for_gemm (padded, permuted, BF16)."""
    w, s = mat['w'], mat['s']
    bm = min(bm_max, xp.shape[0])
    return _gemm_kernel(xp, w, s, bm, pick_bn(w.shape[1], 2176))[:m]
