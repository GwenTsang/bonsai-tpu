"""Prism activation basis changes; all feature dimensions are last."""
import numpy as np


def fwht(x, block_size=1024, xp=np):
    if block_size <= 0 or block_size & (block_size-1) or x.shape[-1] % block_size:
        raise ValueError('Hadamard blocks must be powers of two dividing input width')
    shape = x.shape
    y = x.astype(xp.float32).reshape(*shape[:-1], -1, block_size)
    stride = 1
    while stride < block_size:
        z = y.reshape(*y.shape[:-1], -1, 2, stride)
        a, b = z[..., 0, :], z[..., 1, :]
        y = xp.stack((a+b, a-b), axis=-2).reshape(y.shape)
        stride *= 2
    return (y * (block_size ** -0.5)).reshape(shape)


def grouped_gdn(x, key_heads=16, value_heads=48, xp=np):
    if value_heads % key_heads or x.shape[-1] % value_heads:
        raise ValueError('Invalid GDN geometry')
    hd = x.shape[-1] // value_heads
    y = x.reshape(*x.shape[:-1], value_heads//key_heads, key_heads, hd)
    return xp.swapaxes(y, -3, -2).reshape(x.shape)


def activation_transform(x, signs, block_size=1024, gdn=False, xp=np):
    if gdn:
        x = grouped_gdn(x, xp=xp)
    return fwht(x * signs, block_size, xp)


def embedding_inverse(x, signs, block_size=1024, xp=np):
    return fwht(x, block_size, xp) * signs


def sign_table(metadata):
    widths = metadata['prism.hadamard.sign_widths']
    values = metadata['prism.hadamard.sign_values']
    if sum(widths) != len(values) or len(set(widths)) != len(widths):
        raise ValueError('Malformed sign vector lengths')
    result = {}
    offset = 0
    for width in widths:
        signs = np.asarray(values[offset:offset+width], dtype=np.float32)
        if not np.isin(signs, [-1, 1]).all():
            raise ValueError('Signs must be +/-1')
        result[width] = signs
        offset += width
    return result


def _sylvester(n):
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


_H32 = _sylvester(32)


def fwht_mxu(x, jnp, precision):
    """Normalized 1024-block Walsh-Hadamard as H32 (x) H32 on the MXU: two exact-sign 32x32 products per
    block at full FP32 precision.  ~10x cheaper per decode token on TPU than ten butterfly stages."""
    shape = x.shape
    y = x.reshape(-1, 32, 32)
    h = jnp.asarray(_H32)
    y = jnp.einsum('ab,nbc->nac', h, y, precision=precision)
    y = jnp.einsum('nac,cd->nad', y, h, precision=precision)
    return (y * (1024 ** -0.5)).reshape(shape)
