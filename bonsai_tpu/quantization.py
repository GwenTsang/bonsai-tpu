"""Exact on-disk codecs matching Prism llama.cpp ggml-quants.c.

PTQ1_0 stores its scale at the end; PQ2_0 stores it at the start.
Functions return blocks of 128 values without changing tensor axis order.
"""
import numpy as np


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
