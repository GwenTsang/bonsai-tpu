"""Ternary Bonsai 2 (qwen35 hybrid GDN / attention) single-token decode graph for one TPU.

All ternary matrices stay resident in HBM in the 2-bit B4 layout and are consumed by the int8 Pallas
GEMV kernel.  Semantics follow PrismML llama.cpp src/models/qwen35.cpp at the pinned revision.
"""
import numpy as np
import jax
import jax.numpy as jnp
from . import kernels
from .hadamard import fwht_batched, fwht_mxu, grouped_gdn, sign_table
from .pack import b4_from_ptq1, concat_outputs
from .quantization import decode_codes

HI = jax.lax.Precision.HIGHEST
EPS = 1e-6


class Config:
    def __init__(self, md):
        a = 'qwen35.'
        if md['general.architecture'] != 'qwen35':
            raise ValueError('Unsupported architecture')
        self.n_layer = md[a + 'block_count']
        self.n_embd = md[a + 'embedding_length']
        self.n_ff = md[a + 'feed_forward_length']
        self.n_head = md[a + 'attention.head_count']
        self.n_head_kv = md[a + 'attention.head_count_kv']
        self.head_dim = md[a + 'attention.key_length']
        self.full_interval = md[a + 'full_attention_interval']
        self.rope_dim = md[a + 'rope.dimension_count']
        self.rope_base = md[a + 'rope.freq_base']
        self.eps = md[a + 'attention.layer_norm_rms_epsilon']
        self.conv_kernel = md[a + 'ssm.conv_kernel']
        self.gdn_dk = md[a + 'ssm.state_size']
        self.gdn_k_heads = md[a + 'ssm.group_count']
        self.gdn_v_heads = md[a + 'ssm.time_step_rank']
        self.gdn_inner = md[a + 'ssm.inner_size']
        self.n_ctx_train = md[a + 'context_length']
        self.vocab = len(md['tokenizer.ggml.tokens'])
        if md['prism.hadamard.block_size'] != 1024 or md['prism.hadamard.sign_mode'] != 'explicit' \
                or not md['prism.hadamard.gdn_v_grouped']:
            raise ValueError('Unsupported Hadamard contract')
        assert self.gdn_inner == self.gdn_v_heads * self.gdn_dk and self.rope_dim == 64

    def is_recurrent(self, i):
        return (i + 1) % self.full_interval != 0


def load_params(gguf, log=print):
    """Read the GGUF and place every tensor on the TPU in its resident format."""
    md = gguf.metadata
    cfg = Config(md)
    folded = set(md['prism.hadamard.weight_names'])
    if md['prism.hadamard.inverse_weight_names'] != ['token_embd.weight']:
        raise ValueError('Unexpected inverse-Hadamard tensor set')

    def tern(name):
        if name not in folded:
            raise ValueError(name + ' is not Hadamard-folded; unsupported contract')
        return b4_from_ptq1(*gguf.ternary_blocks(name))

    def dense(name):
        return jnp.asarray(gguf.dense(name))

    params = {'signs': {w: jnp.asarray(s) for w, s in sign_table(md).items()}}
    payload, scales = gguf.ternary_blocks('token_embd.weight')
    params['embed'] = {'payload': jnp.asarray(payload), 'scales': jnp.asarray(scales.astype(np.float32))}
    params['out_norm'] = dense('output_norm.weight')
    params['head'] = tern('output.weight')
    layers = []
    for i in range(cfg.n_layer):
        p = 'blk.%d.' % i
        L = {'attn_norm': dense(p + 'attn_norm.weight'), 'post_norm': dense(p + 'post_attention_norm.weight'),
             'ffn_gu': concat_outputs([tern(p + 'ffn_gate.weight'), tern(p + 'ffn_up.weight')]),
             'ffn_down': tern(p + 'ffn_down.weight')}
        if cfg.is_recurrent(i):
            L['qkvz'] = concat_outputs([tern(p + 'attn_qkv.weight'), tern(p + 'attn_gate.weight')])
            L['alpha'] = dense(p + 'ssm_alpha.weight')
            L['beta'] = dense(p + 'ssm_beta.weight')
            L['conv'] = dense(p + 'ssm_conv1d.weight')          # [C, 4]
            L['dt'] = dense(p + 'ssm_dt.bias')
            L['a'] = dense(p + 'ssm_a')
            L['ssm_norm'] = dense(p + 'ssm_norm.weight')
            L['out'] = tern(p + 'ssm_out.weight')
        else:
            L['qkv'] = concat_outputs([tern(p + 'attn_q.weight'), tern(p + 'attn_k.weight'),
                                       tern(p + 'attn_v.weight')])
            L['q_norm'] = dense(p + 'attn_q_norm.weight')
            L['k_norm'] = dense(p + 'attn_k_norm.weight')
            L['out'] = tern(p + 'attn_output.weight')
        layers.append(L)
        if log and i % 8 == 7:
            log('loaded layers', i + 1)
    params['layers'] = layers
    return cfg, params


def init_state(cfg, n_ctx, kv_dtype=jnp.float32, slack=0):
    """slack: extra KV rows so a padded prefill chunk may write past the last valid position."""
    n_ctx = -(-(n_ctx + slack) // ATT_BLOCK) * ATT_BLOCK
    n_rec = sum(cfg.is_recurrent(i) for i in range(cfg.n_layer))
    n_full = cfg.n_layer - n_rec
    c = cfg.gdn_inner + 2 * cfg.gdn_k_heads * cfg.gdn_dk
    return {
        'pos': jnp.zeros((), jnp.int32),
        'conv': [jnp.zeros((cfg.conv_kernel - 1, c), jnp.float32) for _ in range(n_rec)],
        'gdn': [jnp.zeros((cfg.gdn_v_heads, cfg.gdn_dk, cfg.gdn_dk), jnp.float32) for _ in range(n_rec)],
        'k': [jnp.zeros((cfg.n_head_kv, n_ctx, cfg.head_dim), kv_dtype) for _ in range(n_full)],
        'v': [jnp.zeros((cfg.n_head_kv, n_ctx, cfg.head_dim), kv_dtype) for _ in range(n_full)],
    }


def rms_norm(x, weight):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * weight


def l2_norm(x):
    return x / jnp.maximum(jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True)), EPS)


def rotate(x, signs, gdn=False):
    """Prism activation transform preceding every folded ternary matrix."""
    if gdn:
        x = grouped_gdn(x, xp=jnp)
    if x.ndim == 2:
        return fwht_batched(x * signs, jnp, jax.lax)
    return fwht_mxu(x * signs, jnp, HI)


def embed(params, token):
    e = params['embed']
    codes = decode_codes(e['payload'][token], 143, jnp).astype(jnp.float32)      # [G, 128]
    x = (codes * e['scales'][token][:, None]).reshape(-1)
    return fwht_mxu(x, jnp, HI) * params['signs'][x.shape[0]]


def rope(x, pos, cfg):
    half = cfg.rope_dim // 2
    freq = cfg.rope_base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    theta = pos.astype(jnp.float32) * freq
    c, s = jnp.cos(theta), jnp.sin(theta)
    a, b = x[..., :half], x[..., half:cfg.rope_dim]
    return jnp.concatenate((a * c - b * s, b * c + a * s, x[..., cfg.rope_dim:]), axis=-1)


def ffn(cfg, L, signs, x):
    gu = kernels.gemv(rotate(x, signs[cfg.n_embd]), L['ffn_gu'])
    act = jax.nn.silu(gu[:cfg.n_ff]) * gu[cfg.n_ff:]
    return kernels.gemv(rotate(act, signs[cfg.n_ff]), L['ffn_down'])


def recurrent_mixer(cfg, L, signs, h, conv_state, gdn_state):
    dk, hk, hv = cfg.gdn_dk, cfg.gdn_k_heads, cfg.gdn_v_heads
    c = cfg.gdn_inner + 2 * hk * dk
    qkvz = kernels.gemv(rotate(h, signs[cfg.n_embd]), L['qkvz'])
    qkv, z = qkvz[:c], qkvz[c:]
    alpha = jnp.matmul(L['alpha'], h, precision=HI)
    beta = jax.nn.sigmoid(jnp.matmul(L['beta'], h, precision=HI))
    window = jnp.concatenate((conv_state, qkv[None, :]), axis=0)                  # [4, C] oldest first
    conv = jax.nn.silu(jnp.sum(window * L['conv'].T, axis=0))
    qk = l2_norm(conv[:2 * hk * dk].reshape(2 * hk, dk))
    rep = hv // hk                                                                # value head h uses key head h % hk
    q, k = jnp.tile(qk[:hk], (rep, 1)), jnp.tile(qk[hk:], (rep, 1))               # [hv, dk]
    v = conv[2 * hk * dk:].reshape(hv, dk)
    gate = L['a'] * jax.nn.softplus(alpha + L['dt'])
    # Gated delta rule with two reads of the old state and one write:
    #   S' = e^g S + delta k^T,  delta = (v - e^g S k) beta,  out = S' q = e^g S q + delta (k.q)
    decay = jnp.exp(gate)[:, None]
    sk = jnp.sum(gdn_state * k[:, None, :], axis=-1)
    sq = jnp.sum(gdn_state * q[:, None, :], axis=-1)
    delta = (v - decay * sk) * beta[:, None]
    s = decay[:, :, None] * gdn_state + delta[:, :, None] * k[:, None, :]
    out = (decay * sq + delta * jnp.sum(k * q, axis=-1, keepdims=True)) * (dk ** -0.5)
    out = (rms_norm(out, L['ssm_norm']) * jax.nn.silu(z.reshape(hv, dk))).reshape(-1)
    y = kernels.gemv(rotate(out, signs[cfg.gdn_inner], gdn=True), L['out'])
    return y, window[1:], s


ATT_BLOCK = 4096      # keys per streaming-attention block; KV caches are allocated in multiples of it


def attend(q, qpos, k_cache, v_cache, scale):
    """Causal attention of q[G, R, hd] (row r sees cache positions <= qpos[r]) over k/v caches [G, S, hd].

    Streams over the cache in ATT_BLOCK-key blocks with an online softmax inside a while_loop whose trip
    count follows the *used* length, so one compiled program serves every context length, work scales with
    the tokens actually present, and temporaries are bounded by the block size instead of the context.
    """
    g, r, hd = q.shape
    f32 = k_cache.dtype == jnp.float32
    qc = q if f32 else q.astype(k_cache.dtype)
    nblk = jnp.max(qpos) // ATT_BLOCK + 1

    def body(c):
        b, mx, den, acc = c
        ks = jax.lax.dynamic_slice(k_cache, (0, b * ATT_BLOCK, 0), (g, ATT_BLOCK, hd))
        vs = jax.lax.dynamic_slice(v_cache, (0, b * ATT_BLOCK, 0), (g, ATT_BLOCK, hd))
        if f32:
            sc = jnp.einsum('grd,gsd->grs', qc, ks, precision=HI)
        else:
            sc = jnp.einsum('grd,gsd->grs', qc, ks, preferred_element_type=jnp.float32)
        idx = b * ATT_BLOCK + jnp.arange(ATT_BLOCK)
        sc = jnp.where(idx[None, None, :] <= qpos[None, :, None], sc * scale, -jnp.inf)
        mn = jnp.maximum(mx, jnp.max(sc, axis=-1))
        p = jnp.exp(sc - mn[..., None])
        carry = jnp.exp(mx - mn)
        if f32:
            pv = jnp.einsum('grs,gsd->grd', p, vs, precision=HI)
        else:
            pv = jnp.einsum('grs,gsd->grd', p.astype(vs.dtype), vs, preferred_element_type=jnp.float32)
        return b + 1, mn, den * carry + jnp.sum(p, axis=-1), acc * carry[..., None] + pv

    init = (jnp.int32(0), jnp.full((g, r), -jnp.inf, jnp.float32), jnp.zeros((g, r), jnp.float32),
            jnp.zeros((g, r, hd), jnp.float32))
    _, _, den, acc = jax.lax.while_loop(lambda c: c[0] < nblk, body, init)
    return acc / den[..., None]


def attention_mixer(cfg, L, signs, h, k_cache, v_cache, pos):
    nh, nkv, hd = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    qkv = kernels.gemv(rotate(h, signs[cfg.n_embd]), L['qkv'])
    qg = qkv[:nh * hd * 2].reshape(nh, 2 * hd)
    q = rope(rms_norm(qg[:, :hd], L['q_norm']), pos, cfg)
    gate = qg[:, hd:].reshape(-1)
    k = rope(rms_norm(qkv[nh * hd * 2:nh * hd * 2 + nkv * hd].reshape(nkv, hd), L['k_norm']), pos, cfg)
    v = qkv[nh * hd * 2 + nkv * hd:].reshape(nkv, hd)
    k_cache = jax.lax.dynamic_update_slice(k_cache, k[:, None, :].astype(k_cache.dtype), (0, pos, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v[:, None, :].astype(v_cache.dtype), (0, pos, 0))
    per = nh // nkv
    out = attend(q.reshape(nkv, per, hd), jnp.full((per,), pos), k_cache, v_cache, hd ** -0.5)
    out = out.reshape(-1) * jax.nn.sigmoid(gate)
    y = kernels.gemv(rotate(out, signs[nh * hd]), L['out'])
    return y, k_cache, v_cache


def decode_step(cfg, params, state, token, trace=None):
    """One autoregressive step: token id -> logits [vocab], updated state."""
    signs = params['signs']
    x = embed(params, token)
    pos = state['pos']
    new = {'pos': pos + 1, 'conv': [], 'gdn': [], 'k': [], 'v': []}
    ir = ia = 0
    for i, L in enumerate(params['layers']):
        h = rms_norm(x, L['attn_norm'])
        if cfg.is_recurrent(i):
            y, cs, gs = recurrent_mixer(cfg, L, signs, h, state['conv'][ir], state['gdn'][ir])
            new['conv'].append(cs)
            new['gdn'].append(gs)
            ir += 1
        else:
            y, kc, vc = attention_mixer(cfg, L, signs, h, state['k'][ia], state['v'][ia], pos)
            new['k'].append(kc)
            new['v'].append(vc)
            ia += 1
        x = x + y
        x = x + ffn(cfg, L, signs, rms_norm(x, L['post_norm']))
        if trace is not None:
            trace.append(x)
    h = rms_norm(x, params['out_norm'])
    logits = kernels.gemv(rotate(h, signs[cfg.n_embd]), params['head'])
    return logits, new


# ---------------------------------------------------------------------------------------------------
# Batched prompt processing (prefill): identical semantics, GEMM kernels, sequence-parallel where possible.

def embed_many(params, tokens):
    e = params['embed']
    codes = decode_codes(e['payload'][tokens], 143, jnp).astype(jnp.float32)      # [M, G, 128]
    x = (codes * e['scales'][tokens][:, :, None]).reshape(tokens.shape[0], -1)
    return fwht_batched(x, jnp, jax.lax) * params['signs'][x.shape[-1]]


def ffn_many(cfg, L, signs, x):
    gu = kernels.gemm(rotate(x, signs[cfg.n_embd]), L['ffn_gu'])
    act = jax.nn.silu(gu[:, :cfg.n_ff]) * gu[:, cfg.n_ff:]
    return kernels.gemm(rotate(act, signs[cfg.n_ff]), L['ffn_down'])


def gdn_scan(q, k, v, gate, beta, state):
    """Sequential gated delta rule. q/k/v [T, hv, dk], gate/beta [T, hv], state [hv, dk, dk]."""
    scale = q.shape[-1] ** -0.5

    def step(s, inp):
        qt, kt, vt, gt, bt = inp
        s = s * jnp.exp(gt)[:, None, None]
        pred = jnp.sum(s * kt[:, None, :], axis=-1)
        s = s + ((vt - pred) * bt[:, None])[:, :, None] * kt[:, None, :]
        return s, jnp.sum(s * qt[:, None, :], axis=-1) * scale
    state, out = jax.lax.scan(step, state, (q, k, v, gate, beta))
    return out, state


def gdn_chunked(q, k, v, gate, beta, state, chunk=64):
    """Chunk-parallel gated delta rule, algebraically identical to gdn_scan.

    Within a chunk (decay a_t = e^{g_t}, G = cumsum g):
        delta_i = beta_i (v_i - e^{G_i} S0 k_i - sum_{j<i} e^{G_i-G_j} (k_j.k_i) delta_j)
    i.e. (I + diag(beta) A) Delta = diag(beta) (V - Gamma K S0^T) with A strictly lower triangular and
    independent of S0, so every chunk's unit-triangular inverse is computed at once and only T/chunk
    sequential steps (each a few small matmuls) remain.
        out_t = e^{G_t} S0 q_t + sum_{i<=t} e^{G_t-G_i} (k_i.q_t) delta_i
        S_end = e^{G_C} S0 + sum_i e^{G_C-G_i} delta_i k_i^T
    """
    t, hv, d = q.shape
    n = -(-t // chunk)
    pad = n * chunk - t

    def prep(x):                      # [T, hv, ...] -> [n, hv, C, ...]; padding is a no-op token
        if pad:
            x = jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1))
        x = x.reshape((n, chunk) + x.shape[1:])
        return jnp.swapaxes(x, 1, 2)
    q, k, v, gate, beta = prep(q), prep(k), prep(v), prep(gate), prep(beta)
    g = jnp.cumsum(gate, axis=-1)                                           # [n, hv, C]
    idx = jnp.arange(chunk)
    lower = idx[:, None] >= idx[None, :]
    ratio = jnp.exp(jnp.where(lower, g[..., :, None] - g[..., None, :], -jnp.inf))   # e^{G_i - G_j}, j <= i
    kk = jnp.einsum('nhid,nhjd->nhij', k, k, precision=HI)
    strict = idx[:, None] > idx[None, :]
    lmat = jnp.eye(chunk, dtype=jnp.float32) + beta[..., :, None] * jnp.where(strict, ratio * kk, 0.0)
    tinv = jax.scipy.linalg.solve_triangular(lmat, jnp.broadcast_to(jnp.eye(chunk, dtype=jnp.float32), lmat.shape),
                                             lower=True, unit_diagonal=True)
    aq = ratio * jnp.einsum('nhtd,nhid->nhti', q, k, precision=HI)          # [t, i], i <= t
    gam = jnp.exp(g)
    wend = jnp.exp(g[..., -1:] - g)                                         # e^{G_C - G_i}

    def step(s, c):
        qc, kc, vc, bc, gc, tc, ac, wc = c
        ks = jnp.einsum('hcd,hvd->hcv', kc, s, precision=HI)
        delta = jnp.einsum('hci,hiv->hcv', tc, bc[..., None] * (vc - gc[..., None] * ks), precision=HI)
        qs = jnp.einsum('hcd,hvd->hcv', qc, s, precision=HI)
        out = gc[..., None] * qs + jnp.einsum('hti,hiv->htv', ac, delta, precision=HI)
        s = gc[:, -1, None, None] * s + jnp.einsum('hcv,hcd->hvd', delta * wc[..., None], kc, precision=HI)
        return s, out
    state, out = jax.lax.scan(step, state, (q, k, v, beta, gam, tinv, aq, wend))
    out = jnp.swapaxes(out, 1, 2).reshape(n * chunk, hv, d)[:t]
    return out * (d ** -0.5), state


def recurrent_mixer_many(cfg, L, signs, h, conv_state, gdn_state, n_valid):
    m = h.shape[0]
    valid = (jnp.arange(m) < n_valid).astype(jnp.float32)
    dk, hk, hv = cfg.gdn_dk, cfg.gdn_k_heads, cfg.gdn_v_heads
    c = cfg.gdn_inner + 2 * hk * dk
    qkvz = kernels.gemm(rotate(h, signs[cfg.n_embd]), L['qkvz'])
    qkv, z = qkvz[:, :c], qkvz[:, c:]
    alpha = jnp.matmul(h, L['alpha'].T, precision=HI)
    beta = jax.nn.sigmoid(jnp.matmul(h, L['beta'].T, precision=HI))
    window = jnp.concatenate((conv_state, qkv), axis=0)                           # [3 + M, C]
    nk = cfg.conv_kernel
    conv = sum(window[i:i + m] * L['conv'][:, i] for i in range(nk))
    conv = jax.nn.silu(conv)
    qk = l2_norm(conv[:, :2 * hk * dk].reshape(m, 2 * hk, dk))
    rep = hv // hk
    q, k = jnp.tile(qk[:, :hk], (1, rep, 1)), jnp.tile(qk[:, hk:], (1, rep, 1))
    v = conv[:, 2 * hk * dk:].reshape(m, hv, dk)
    gate = L['a'] * jax.nn.softplus(alpha + L['dt']) * valid[:, None]
    # padded rows are exact no-ops for the recurrence: no decay, no write
    out, gdn_state = gdn_chunked(q, k * valid[:, None, None], v, gate, beta * valid[:, None], gdn_state, chunk=32)
    out = (rms_norm(out, L['ssm_norm']) * jax.nn.silu(z.reshape(m, hv, dk))).reshape(m, -1)
    y = kernels.gemm(rotate(out, signs[cfg.gdn_inner], gdn=True), L['out'])
    new_conv = jax.lax.dynamic_slice(window, (n_valid, 0), (nk - 1, c))            # last 3 valid inputs
    return y, new_conv, gdn_state


def attention_mixer_many(cfg, L, signs, h, k_cache, v_cache, pos):
    m = h.shape[0]
    nh, nkv, hd = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    positions = pos + jnp.arange(m)
    qkv = kernels.gemm(rotate(h, signs[cfg.n_embd]), L['qkv'])
    qg = qkv[:, :nh * hd * 2].reshape(m, nh, 2 * hd)
    q = rope(rms_norm(qg[..., :hd], L['q_norm']), positions[:, None, None], cfg)
    gate = qg[..., hd:].reshape(m, -1)
    k = rms_norm(qkv[:, nh * hd * 2:nh * hd * 2 + nkv * hd].reshape(m, nkv, hd), L['k_norm'])
    k = rope(k, positions[:, None, None], cfg)
    v = qkv[:, nh * hd * 2 + nkv * hd:].reshape(m, nkv, hd)
    k_cache = jax.lax.dynamic_update_slice(k_cache, k.transpose(1, 0, 2).astype(k_cache.dtype), (0, pos, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, v.transpose(1, 0, 2).astype(v_cache.dtype), (0, pos, 0))
    per = nh // nkv
    qh = q.reshape(m, nkv, per, hd).transpose(1, 0, 2, 3).reshape(nkv, m * per, hd)       # rows (token, head)
    out = attend(qh, jnp.repeat(positions, per), k_cache, v_cache, hd ** -0.5)
    out = out.reshape(nkv, m, per, hd).transpose(1, 0, 2, 3).reshape(m, -1) * jax.nn.sigmoid(gate)
    y = kernels.gemm(rotate(out, signs[nh * hd]), L['out'])
    return y, k_cache, v_cache


def forward_chunk(cfg, params, state, tokens, n_valid):
    """Batched pass over tokens[M] (first n_valid real, rest padding).  Returns final hidden rows [M, d] and
    the new state.  Padding never alters recurrent state; its KV rows land beyond the valid range and are
    overwritten by later tokens (caches carry M rows of slack)."""
    signs = params['signs']
    x = embed_many(params, tokens)
    pos = state['pos']
    new = {'pos': pos + n_valid, 'conv': [], 'gdn': [], 'k': [], 'v': []}
    ir = ia = 0
    for i, L in enumerate(params['layers']):
        h = rms_norm(x, L['attn_norm'])
        if cfg.is_recurrent(i):
            y, cs, gs = recurrent_mixer_many(cfg, L, signs, h, state['conv'][ir], state['gdn'][ir], n_valid)
            new['conv'].append(cs)
            new['gdn'].append(gs)
            ir += 1
        else:
            y, kc, vc = attention_mixer_many(cfg, L, signs, h, state['k'][ia], state['v'][ia], pos)
            new['k'].append(kc)
            new['v'].append(vc)
            ia += 1
        x = x + y
        x = x + ffn_many(cfg, L, signs, rms_norm(x, L['post_norm']))
    return x, new


def prefill_chunk(cfg, params, state, tokens, n_valid):
    """Prompt processing: one compiled program serves every length <= M.  Returns the logits after the
    last valid token and the new state."""
    x, new = forward_chunk(cfg, params, state, tokens, n_valid)
    h = rms_norm(x[n_valid - 1], params['out_norm'])
    return kernels.gemv(rotate(h, params['signs'][cfg.n_embd]), params['head']), new


def verify_chunk(cfg, params, state, tokens, n_valid):
    """Speculative-decoding verification: logits [M, vocab] after every position, and the new state."""
    x, new = forward_chunk(cfg, params, state, tokens, n_valid)
    h = rms_norm(x, params['out_norm'])
    return kernels.gemm(rotate(h, params['signs'][cfg.n_embd]), params['head']), new
