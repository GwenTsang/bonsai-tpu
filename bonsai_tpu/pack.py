"""On-device conversion of released blocks into the resident layouts (see kernels.py): PTQ1_0 -> B4 (2-bit
ternary), Q2_K / Q3_K -> P8 (one byte per weight)."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from .kernels import P8_BIAS
from .quantization import decode_codes, kquant_codes, kquant_scales, Q2_K


@jax.jit
def _ptq1_to_b4(payload):
    """payload uint8 [n, G, 26] -> uint8 [G*32, n] with byte (g,i) packing k = 128g + 32j + i at bits 2j."""
    n, g, _ = payload.shape
    c = (decode_codes(payload, 143, jnp) + 1).astype(jnp.uint8)      # [n, G, 128] in {0,1,2}
    c = c.reshape(n, g, 4, 32)
    b = c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
    return b.reshape(n, g * 32).T


def b4_from_ptq1(payload, scales, chunk=12416):
    """Host PTQ1 views (payload [N,G,26] u8, scales [N,G] f16) -> device dict(w u8 [K/4,N], s i16 [G,N])."""
    n = payload.shape[0]
    parts = [_ptq1_to_b4(jnp.asarray(payload[i:i + chunk])) for i in range(0, n, chunk)]
    w = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
    if (scales < 0).any() or not np.isfinite(scales).all():
        raise ValueError('unsupported group scales (negative or non-finite)')
    # scales stay in their released 16-bit form (raw IEEE-half bits); kernels decode them with integer ops
    if scales.max() < 64:        # the kernels' integer half decode covers [0, 128)
        s = jnp.asarray(np.ascontiguousarray(scales.T).astype(np.float16).view(np.int16))
    else:
        s = jnp.asarray(np.ascontiguousarray(scales.T).astype(np.float32))
    return {'w': w, 's': s}


@functools.partial(jax.jit, static_argnums=1)
def _kquant_to_p8(blocks, kind):
    """superblocks uint8 [n, B, bytes] -> uint8 [B*256, n] of code + P8_BIAS."""
    n, b, _ = blocks.shape
    codes, _ = kquant_codes(blocks, kind, jnp)
    return (codes + P8_BIAS).astype(jnp.uint8).reshape(n, b * 256).T


def p8_from_kquant(kind, blocks, chunk=640):
    """Host K-quant view (superblocks [N, K/256, bytes] u8) -> device dict(p u8 [K, N], s f32 [K/128, N]) and,
    for Q2_K, c f32 [N, K/16]: the per-sub-block offsets -dmin*min that multiply sub-block input sums.

    The sub-block multipliers are folded into the integer codes (weight = d * code), so the kernels only
    need one signed scale per 256 inputs; it is stored once per 128 so they share the B4 group machinery."""
    n = blocks.shape[0]
    parts = [_kquant_to_p8(jnp.asarray(blocks[i:i + chunk]), kind) for i in range(0, n, chunk)]
    d, dmin = kquant_scales(blocks, kind)
    if not (np.isfinite(d).all() and (dmin is None or np.isfinite(dmin).all())):
        raise ValueError('non-finite K-quant scales')
    mat = {'p': parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1),
           's': jnp.asarray(np.repeat(np.ascontiguousarray(d.T), 2, axis=0))}
    if kind == Q2_K:
        mins = np.asarray(blocks[..., :16]) >> 4                                          # [N, B, 16]
        mat['c'] = jnp.asarray((-dmin[..., None] * mins.astype(np.float32)).reshape(n, -1))
    return mat


def p8_to_dense(mat):
    """Reference decode of a P8 matrix to float32 [N, K] (tests and small validation only)."""
    p = np.asarray(mat['p']).astype(np.float32) - P8_BIAS
    w = (p * np.repeat(np.asarray(mat['s']), 128, axis=0)).T
    if 'c' in mat:
        w = w + np.repeat(np.asarray(mat['c']), 16, axis=1)
    return w


def concat_outputs(mats):
    """Fuse matrices that consume the same input into one wider matrix (outputs concatenated)."""
    return {'w': jnp.concatenate([m['w'] for m in mats], axis=1),
            's': jnp.concatenate([m['s'] for m in mats], axis=1)}


def b4_to_dense(mat):
    """Reference decode of a B4 matrix to float32 [N, K] (tests and small validation only)."""
    w = np.asarray(mat['w'])
    s = np.asarray(mat['s'])
    s = s.view(np.float16).astype(np.float32) if s.dtype == np.int16 else s
    g = s.shape[0]
    n = w.shape[1]
    b = w.reshape(g, 32, n)
    c = np.stack([(b >> (2 * j)) & 3 for j in range(4)], axis=1).astype(np.float32) - 1   # [g, j, i, n]
    return (c.reshape(g, 128, n) * s[:, None, :]).reshape(g * 128, n).T
