"""Exact on-disk codecs matching Prism llama.cpp ggml-quants.c.

PTQ1_0 stores its scale at the end; PQ2_0 stores it at the start.
Functions return blocks of 128 values without changing tensor axis order.

Q2_K / Q3_K (stock llama.cpp K-quants, superblocks of 256 weights in 16 sub-blocks of 16) appear in releases
whose edited matrices are no longer ternary (e.g. the abliterated Huihui build); see kquant_codes.
"""
import numpy as np

Q2_K, Q3_K = 10, 11
KQUANT_BYTES = {Q2_K: 84, Q3_K: 110}      # bytes per 256-weight superblock


def kquant_codes(blocks, kind, xp=np):
    """K-quant superblocks uint8 [..., bytes] -> (codes int16 [..., 256], mins uint8 [..., 16] or None).

    weight[k] = d * codes[k] - dmin * mins[k // 16], where the half-precision d (and dmin for Q2_K) are read
    by kquant_scales.  Q3_K: codes = (scale6 - 32) * (q3 - 4) in [-124, 128], no mins.  Q2_K: codes =
    scale4 * q2 in [0, 45].  Element order inside a superblock is 128*half + 32*j + l, taken from bit pair j
    of byte 32*half + l, exactly as dequantize_row_q{2,3}_K walks it.  xp may be numpy or jax.numpy.
    """
    lead = blocks.shape[:-1]
    shifts = (xp.arange(4, dtype=xp.uint8) * 2).reshape(4, 1)
    if kind == Q3_K:
        hm, qs, sc = blocks[..., :32], blocks[..., 32:96], blocks[..., 96:108]
        low = xp.concatenate([sc[..., 0:8] & 15, sc[..., 0:8] >> 4], axis=-1)
        top = sc[..., 8:12]
        high = xp.concatenate([(top >> n) & 3 for n in (0, 2, 4, 6)], axis=-1)
        mult = (low | (high << 4)).astype(xp.int16) - 32                                  # [..., 16]
        q = (qs.reshape(*lead, 2, 1, 32) >> shifts) & 3                                   # [..., half, j, l]
        bit = xp.arange(8, dtype=xp.uint8).reshape(2, 4, 1)
        q = q + (((hm.reshape(*lead, 1, 1, 32) >> bit) & 1) << 2)
        codes = (q.astype(xp.int16) - 4).reshape(*lead, 16, 16) * mult[..., None]
        return codes.reshape(*lead, 256), None
    if kind == Q2_K:
        sc, qs = blocks[..., :16], blocks[..., 16:80]
        q = (qs.reshape(*lead, 2, 1, 32) >> shifts) & 3
        codes = q.astype(xp.int16).reshape(*lead, 16, 16) * (sc & 15).astype(xp.int16)[..., None]
        return codes.reshape(*lead, 256), sc >> 4
    raise ValueError(f'Unsupported K-quant type: {kind}')


def kquant_scales(blocks, kind):
    """Host-side superblock scales as float32: (d [...], dmin [...] or None)."""
    at = {Q3_K: 108, Q2_K: 80}[kind]
    half = lambda a: np.ascontiguousarray(a).view('<f2').reshape(blocks.shape[:-1]).astype(np.float32)
    return half(blocks[..., at:at + 2]), (half(blocks[..., 82:84]) if kind == Q2_K else None)


def dequantize_kquant(raw, kind):
    """Reference float decode: bytes -> float32 [superblocks, 256]."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, KQUANT_BYTES[kind])
    codes, mins = kquant_codes(blocks, kind)
    d, dmin = kquant_scales(blocks, kind)
    out = d[:, None] * codes.astype(np.float32)
    if mins is not None:
        out = out - np.repeat(dmin[:, None] * mins.astype(np.float32), 16, axis=1)
    return out


def split_blocks(raw, kind):
    size = {143: 28, 142: 34}[kind]
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    if kind == 143:
        payload, scale_bytes = blocks[:, :26], blocks[:, 26:28]
    else:
        payload, scale_bytes = blocks[:, 2:], blocks[:, :2]
    scales = np.ascontiguousarray(scale_bytes).view('<f2').reshape(-1).astype(np.float32)
    return payload.copy(), scales


def decode_codes(payload, kind, xp=np):
    """Decode uint8 payload to int8 codes; xp may be numpy or jax.numpy."""
    if kind == 142:
        shifts = xp.arange(4, dtype=xp.uint8) * 2
        return (((payload[..., :, None] >> shifts) & 3).reshape(*payload.shape[:-1], 128).astype(xp.int8) - 1)
    if kind != 143:
        raise ValueError(f'Unsupported ternary type: {kind}')
    parts = []
    for start, count, trits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
        powers = xp.asarray([3**i for i in range(trits)], dtype=xp.uint16)
        v = payload[..., None, start:start+count].astype(xp.uint16) * powers[:, None]
        codes = (((v & 255) * 3) >> 8).astype(xp.int8) - 1
        parts.append(codes.reshape(*payload.shape[:-1], count * trits))
    return xp.concatenate(parts, axis=-1)


def dequantize(raw, kind):
    payload, scales = split_blocks(raw, kind)
    return decode_codes(payload, kind).astype(np.float32) * scales[:, None]


def encode_ptq1(codes, scales):
    """Lossless packing of already-ternary blocks; does not quantize floats."""
    codes = np.asarray(codes)
    if codes.shape[-1] != 128 or not np.isin(codes, [-1, 0, 1]).all():
        raise ValueError('Expected blocks of 128 ternary codes')
    digits = codes.astype(np.uint16) + 1
    out = np.empty((*codes.shape[:-1], 28), np.uint8)
    offset = 0
    for start, count, trits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
        d = digits[..., offset:offset+count*trits].reshape(*digits.shape[:-1], trits, count)
        q = np.zeros((*digits.shape[:-1], count), np.uint16)
        for n in range(trits):
            q = q * 3 + d[..., n, :]
        if trits == 4:
            q *= 3
        out[..., start:start+count] = (q * 256 + 242) // 243
        offset += count * trits
    scale = np.broadcast_to(np.asarray(scales, dtype='<f2'), codes.shape[:-1]).copy()
    out[..., 26:] = scale.reshape(-1).view(np.uint8).reshape(*codes.shape[:-1], 2)
    return out
