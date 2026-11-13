"""On-device conversion of released PTQ1_0 blocks into the resident B4 layout (see kernels.py)."""
import numpy as np
import jax
import jax.numpy as jnp
from .quantization import decode_codes


@jax.jit
def _ptq1_to_b4(payload):
    """payload uint8 [n, G, 26] -> uint8 [G*32, n] with byte (g,i) packing k = 128g + 32j + i at bits 2j."""
    n, g, _ = payload.shape
    c = (decode_codes(payload, 143, jnp) + 1).astype(jnp.uint8)      # [n, G, 128] in {0,1,2}
    c = c.reshape(n, g, 4, 32)
    b = c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
    return b.reshape(n, g * 32).T


def b4_from_ptq1(payload, scales, chunk=12416):
    """Host PTQ1 views (payload [N,G,26] u8, scales [N,G] f16) -> device dict(w u8 [K/4,N], s f32 [G,N])."""
    n = payload.shape[0]
    parts = [_ptq1_to_b4(jnp.asarray(payload[i:i + chunk])) for i in range(0, n, chunk)]
    w = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
    s = jnp.asarray(np.ascontiguousarray(scales.T).astype(np.float32))
    return {'w': w, 's': s}


def concat_outputs(mats):
    """Fuse matrices that consume the same input into one wider matrix (outputs concatenated)."""
    return {'w': jnp.concatenate([m['w'] for m in mats], axis=1),
            's': jnp.concatenate([m['s'] for m in mats], axis=1)}


def b4_to_dense(mat):
    """Reference decode of a B4 matrix to float32 [N, K] (tests and small validation only)."""
    w = np.asarray(mat['w'])
    s = np.asarray(mat['s'])
    g = s.shape[0]
    n = w.shape[1]
    b = w.reshape(g, 32, n)
    c = np.stack([(b >> (2 * j)) & 3 for j in range(4)], axis=1).astype(np.float32) - 1   # [g, j, i, n]
    return (c.reshape(g, 128, n) * s[:, None, :]).reshape(g * 128, n).T
