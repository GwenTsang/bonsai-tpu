"""TPU kernels over the engine's resident "B4" ternary layout.

B4 layout for a matrix W[N, K] (N outputs, K inputs, K % 1024 == 0, scale groups of 128 inputs):
  w: uint8 [K/4, N].  Byte row r = 32*g + i; bit pair j (bits 2j, 2j+1) holds code(k, n) = t + 1 in {0,1,2}
     for input k = 128*g + 32*j + i.  One whole-tile shift+mask therefore yields 4 contiguous sub-blocks.
  s: int16 [K/128, N] group scales as raw IEEE-half bit patterns (decoded in-kernel; FP32 also accepted).
All unpacking runs in 8-bit vector lanes (bitwise ops only; Mosaic has no i8 arithmetic), the MXU consumes
int8 directly, and per-group partial sums stay exact integers until the single float scale multiply.

P8 layout, for the few matrices a release stores as Q2_K / Q3_K instead of ternary (dict key 'p'):
  p: uint8 [K, N].  Byte row k holds code(k, n) + P8_BIAS, where weight = s * code and the code already
     contains the K-quant sub-block multiplier (pack.p8_from_kquant).  Inputs keep their natural order.
  s: float32 [K/128, N] signed scales (each 256-input superblock scale stored twice).
  c: optional float32 [N, K/16] offsets applied to sums of 16 consecutive inputs (Q2_K minima).
The kernels split each byte into two nibbles with the same shift/mask ops and otherwise run the B4 code path.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BK = 1024          # inputs per K tile
NG = BK // 128     # scale groups per K tile
XMAX = 32639.0     # largest |q| such that the rounded high byte stays within int8
P8_BIAS = 124      # P8 stored byte = code + P8_BIAS; K-quant codes lie in [-124, 128]
INTERPRET = False  # tests set this to run the kernels through the Pallas interpreter on CPU
HI = jax.lax.Precision.HIGHEST


def _scales_f32(sv):
    """Group scales as FP32 inside a kernel.  Scales may be stored as raw IEEE half bit patterns (int16):
    Mosaic cannot load f16, so the half (in [0, 128), checked at load) is decoded with integer ops; every half is
    an integer multiple of 2^-24, which FP32 represents exactly."""
    if sv.dtype != jnp.int16:
        return sv
    u = sv.astype(jnp.int32) & 0x7FFF
    e = u >> 10
    m = u & 1023
    i = jnp.where(e == 0, m, (m | 1024) << jnp.maximum(e - 1, 0))
    return i.astype(jnp.float32) * (2.0 ** -24)


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


def prep_gemv8(x):
    """prep_gemv for a P8 matrix: same block-diagonal operand with inputs in natural order, and the
    per-group sums pre-multiplied by the byte bias so the kernel's subtraction leaves sum(code * q)."""
    k = x.shape[0]
    nt = k // BK
    q, alpha = quantize16(x)
    hi = (q + 128) >> 8
    lo = q - (hi << 8)
    eye = jnp.eye(NG, dtype=jnp.int32)[None, :, :, None]

    def bd(v):
        return (eye * v.reshape(nt, 1, NG, 128)).reshape(nt, NG, BK)
    xb = jnp.concatenate([bd(hi), bd(lo)], axis=1).reshape(nt * 2 * NG, BK).astype(jnp.int8)
    xsum = jnp.broadcast_to((P8_BIAS * q.reshape(k // 128, 128).sum(-1))[:, None], (k // 128, 128))
    return xb, xsum, alpha[0]


def _gemv_kernel(xb, xsum, w, s, bn, p8=False):
    """B4: w uint8 [K/4, N].  P8 (p8=True): w uint8 [K, N]; integer sums stay below 2^31
    (128 inputs * 255 * XMAX = 1.07e9)."""
    rows = BK if p8 else BK // 4
    n = w.shape[1]
    nt = w.shape[0] // rows
    dims = (((1,), (0,)), ((), ()))

    def body(xr, or_, wr, sr, yr, acc):
        @pl.when(pl.program_id(1) == 0)
        def _():
            acc[...] = jnp.zeros((1, bn), jnp.float32)
        wv = wr[...]
        if p8:
            xv = xr[...]
            p = jax.lax.dot_general(xv, (wv & 15).astype(jnp.int8), dims, preferred_element_type=jnp.int32) \
                + (jax.lax.dot_general(xv, (wv >> 4).astype(jnp.int8), dims,
                                       preferred_element_type=jnp.int32) << 4)
        else:
            t = jnp.concatenate([wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6], axis=0).astype(jnp.int8)
            p = jax.lax.dot_general(xr[...], t, dims, preferred_element_type=jnp.int32)
        exact = (p[:NG] << 8) + p[NG:] - or_[...][:, :1]
        acc[...] += jnp.sum(exact.astype(jnp.float32) * _scales_f32(sr[...]), axis=0, keepdims=True)

        @pl.when(pl.program_id(1) == nt - 1)
        def _():
            yr[...] = acc[...]

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((2 * NG, BK), lambda j, g: (g, 0)),
                      pl.BlockSpec((NG, 128), lambda j, g: (g, 0)),
                      pl.BlockSpec((rows, bn), lambda j, g: (g, j)),
                      pl.BlockSpec((NG, bn), lambda j, g: (g, j))],
            out_specs=pl.BlockSpec((1, bn), lambda j, g: (0, j)),
            scratch_shapes=[pltpu.VMEM((1, bn), jnp.float32)],
            grid=(n // bn, nt)),
        out_shape=jax.ShapeDtypeStruct((1, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary')),
        interpret=INTERPRET)(xb, xsum, w, s)


def pick_bn(n, target=4352):
    """Largest multiple of 128 dividing n that does not exceed target."""
    best = None
    for bn in range(128, min(n, target) + 1, 128):
        if n % bn == 0:
            best = bn
    if best is None:
        raise ValueError('N=%d has no legal lane tile' % n)
    return best


def _subblock_offsets(x, mat):
    """P8 'c' term: offsets times the sums of 16 consecutive inputs.  x [..., K] -> [..., N]."""
    xs = x.reshape(x.shape[:-1] + (x.shape[-1] // 16, 16)).sum(-1)
    return jnp.matmul(xs, mat['c'].T, precision=HI)


def gemv(x, mat, prepared=None):
    """y[N] = W x for one activation vector; `prepared` reuses prep_gemv(x) across matrices sharing x."""
    if 'p' in mat:
        xb, xsum, alpha = prepared if prepared is not None else prep_gemv8(x)
        w, s = mat['p'], mat['s']
        y = _gemv_kernel(xb, xsum, w, s, pick_bn(w.shape[1]), p8=True)[0] / alpha
        return y + _subblock_offsets(x, mat) if 'c' in mat else y
    xb, xsum, alpha = prepared if prepared is not None else prep_gemv(x)
    w, s = mat['w'], mat['s']
    return _gemv_kernel(xb, xsum, w, s, pick_bn(w.shape[1]))[0] / alpha


def _gemm_kernel(x, w, s, bm, bn, p8=False):
    m, k = x.shape
    n = w.shape[1]
    nt = k // BK
    q = BK // 4
    rows = BK if p8 else q
    per = 128 if p8 else 32          # byte rows per scale group

    def body(xr, wr, sr, yr, acc):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc[...] = jnp.zeros((bm, bn), jnp.float32)
        wv, sv, xv = wr[...], _scales_f32(sr[...]), xr[...]
        srep = jnp.concatenate([jnp.broadcast_to(sv[g:g + 1], (per, bn)) for g in range(NG)], axis=0)
        tot = acc[...]
        if p8:      # four quarter tiles of 256 inputs keep the temporaries the size of a B4 plane
            lo = (wv & 15).astype(jnp.int32).astype(jnp.float32)
            hi = (wv >> 4).astype(jnp.int32).astype(jnp.float32)
            planes = [(lo[j * q:(j + 1) * q] + 16.0 * hi[j * q:(j + 1) * q] - float(P8_BIAS))
                      * srep[j * q:(j + 1) * q] for j in range(4)]
        else:
            planes = [(plane.astype(jnp.int32).astype(jnp.float32) - 1.0) * srep
                      for plane in (wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6)]
        for j, plane in enumerate(planes):
            tot = tot + jax.lax.dot_general(xv[:, j * q:(j + 1) * q], plane.astype(jnp.bfloat16),
                                            (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        acc[...] = tot

        @pl.when(pl.program_id(2) == nt - 1)
        def _():
            yr[...] = acc[...]

    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm, BK), lambda i, j, g: (i, g)),
                      pl.BlockSpec((rows, bn), lambda i, j, g: (g, j)),
                      pl.BlockSpec((NG, bn), lambda i, j, g: (g, j))],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, g: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, nt)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel', 'arbitrary')),
        interpret=INTERPRET)(x, w, s)


def gemm(x, mat, bm_max=256):
    """y[M, N] = x[M, K] W^T for prompt processing.  Weights are reconstructed as BF16 (scale folded) in
    VMEM; activations are BF16; accumulation is FP32."""
    m, k = x.shape
    bm = min(bm_max, -(-m // 8) * 8)
    mp = -(-m // bm) * bm
    if 'p' in mat:
        xp = x.astype(jnp.bfloat16)
    else:
        xp = x.reshape(m, k // BK, NG, 4, 32).transpose(0, 1, 3, 2, 4).reshape(m, k).astype(jnp.bfloat16)
    if mp != m:
        xp = jnp.pad(xp, ((0, mp - m), (0, 0)))
    return gemm_rotated(xp, mat, m, bm_max)


def _gemm_perm():
    """Column order of the GEMM kernel's shift planes: position (j, g, i) holds input 128g + 32j + i."""
    import numpy as np
    return np.arange(BK).reshape(NG, 4, 32).transpose(1, 0, 2).reshape(-1)


def rotate_for_gemm(x, signs, bm_max=256, natural=False):
    """Fused prefill pre-processing in VMEM: sign flip, normalized 1024-block Hadamard (two exact BF16 MXU
    passes on a bf16-split activation), permutation into the GEMM kernel's column order and the BF16 cast.
    x f32 [M, K] -> bf16 [M', K] with M' = M rounded up to the row tile (feed to gemm_rotated).
    natural=True keeps the inputs in their natural order, as P8 matrices expect."""
    import numpy as np
    from .hadamard import _H1024
    m, k = x.shape
    bm = min(bm_max, -(-m // 16) * 16)
    mp = -(-m // bm) * bm
    if mp != m:
        x = jnp.pad(x, ((0, mp - m), (0, 0)))
    cols = slice(None) if natural else _gemm_perm()
    h = jnp.asarray(_H1024[:, cols] * np.float32(1024 ** -0.5), dtype=jnp.bfloat16)   # +-1/32: exact

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
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel')),
        interpret=INTERPRET)(x, signs.reshape(1, k), h)


def gemm_rotated(xp, mat, m, bm_max=256):
    """y[m, N] for activations already produced by rotate_for_gemm (padded, BF16; permuted for B4 matrices,
    natural order for P8 ones)."""
    bm = min(bm_max, xp.shape[0])
    if 'p' in mat:
        w, s = mat['p'], mat['s']
        y = _gemm_kernel(xp, w, s, bm, pick_bn(w.shape[1], 2176), p8=True)[:m]
        return y + _subblock_offsets(xp[:m].astype(jnp.float32), mat) if 'c' in mat else y
    w, s = mat['w'], mat['s']
    bm = min(bm_max, xp.shape[0])
    return _gemm_kernel(xp, w, s, bm, pick_bn(w.shape[1], 2176))[:m]
