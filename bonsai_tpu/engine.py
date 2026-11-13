"""User-facing inference engine: model loading, prefill/decode scheduling, sampling, streaming, chat."""
import functools
import time
import numpy as np
import jax
import jax.numpy as jnp
from . import model as M
from .gguf import GGUF
from .tokenizer import BonsaiTokenizer, StreamDecoder

PREFILL_BUCKETS = (512, 128, 32, 8)
TOPK = 64


def top_candidates(logits, k=TOPK, width=128):
    """Exact top-k of a long vector without a full sort: every global top-k element lives in one of the k
    rows with the largest row maxima, so only k*width candidates are sorted."""
    rows = logits.reshape(-1, width)
    _, ridx = jax.lax.top_k(jnp.max(rows, axis=1), k)
    vals, cidx = jax.lax.top_k(rows[ridx].reshape(-1), k)
    return vals, ridx[cidx // width] * width + cidx % width


def sample_token(logits, key, temperature, top_k, top_p, min_p):
    """In-graph sampler over the exact top-TOPK candidates; temperature <= 0 selects argmax."""
    vals, idx = top_candidates(logits)
    greedy = idx[0]
    t = jnp.maximum(temperature, 1e-6)
    z = (vals - vals[0]) / t
    rank = jnp.arange(TOPK)
    z = jnp.where(rank < jnp.clip(top_k, 1, TOPK), z, -jnp.inf)
    p = jax.nn.softmax(z)
    cum = jnp.cumsum(p)
    keep = (cum - p < top_p) & (p >= min_p * p[0])
    z = jnp.where(keep | (rank == 0), z, -jnp.inf)
    choice = idx[jax.random.categorical(key, z)]
    return jnp.where(temperature <= 0, greedy, choice).astype(jnp.int32)


def decode_block(cfg, act_len, n_max, params, state, token, key, n, temperature, top_k, top_p, min_p, stop_ids):
    """Up to n (<= n_max) autoregressive steps in ONE device program (no host round trip per token).

    Feeds `token`, samples the next, feeds it, ...  Stops right after sampling a stop id (which is recorded
    but not fed), so the returned state always corresponds exactly to the fed tokens.
    Returns (sampled[n_max], count, state, last_logits).
    """
    def cond(c):
        i, _, _, _, _, done, _ = c
        return (i < n) & ~done

    def body(c):
        i, tok, st, k, out, _, _ = c
        logits, st = M.decode_step(cfg, params, st, tok, act_len)
        k, sub = jax.random.split(k)
        nxt = sample_token(logits, sub, temperature, top_k, top_p, min_p)
        return i + 1, nxt, st, k, out.at[i].set(nxt), jnp.any(nxt == stop_ids), logits

    init = (jnp.int32(0), token, state, key, jnp.full((n_max,), -1, jnp.int32), jnp.bool_(False),
            jnp.zeros((cfg.vocab,), jnp.float32))
    i, _, state, key, out, _, logits = jax.lax.while_loop(cond, body, init)
    return out, i, state, key, logits


class Engine:
    def __init__(self, gguf_path, n_ctx=8192, kv_dtype='float32', log=print, preloaded=None, block=8):
        self.block = int(block)
        self.log = log or (lambda *a, **k: None)
        t = time.perf_counter()
        if preloaded is not None:      # (gguf, cfg, params) already resident, e.g. in a notebook kernel
            self.gguf, self.cfg, self.params = preloaded
        else:
            self.gguf = GGUF(gguf_path)
            self.cfg, self.params = M.load_params(self.gguf, log=None)
        self.tokenizer = BonsaiTokenizer(self.gguf.metadata)
        jax.block_until_ready(self.params)
        self.load_seconds = time.perf_counter() - t
        self.n_ctx = int(n_ctx)
        self.kv_dtype = jnp.dtype(kv_dtype)
        self._step = {}
        self._prefill = {}
        self.reset()
        self.stats = {}

    # -- state ------------------------------------------------------------------------------------
    def reset(self):
        self.state = M.init_state(self.cfg, self.n_ctx, self.kv_dtype)
        self.history = []          # token ids already absorbed into the state
        self.pending_logits = None

    def _act_len(self, upto):
        """Static attention bound: smallest power-of-two bucket (>=1024) covering `upto` positions."""
        n = 1024
        while n < upto:
            n *= 2
        return min(n, self.n_ctx)

    def _step_fn(self, act_len):
        if act_len not in self._step:
            cfg = self.cfg

            def fn(params, state, token, key, temperature, top_k, top_p, min_p):
                logits, state = M.decode_step(cfg, params, state, token, act_len)
                return sample_token(logits, key, temperature, top_k, top_p, min_p), logits, state
            self._step[act_len] = jax.jit(fn, donate_argnums=(1,))
        return self._step[act_len]

    def _block_fn(self, act_len):
        k = ('block', act_len, self.block)
        if k not in self._step:
            self._step[k] = jax.jit(functools.partial(decode_block, self.cfg, act_len, self.block),
                                    donate_argnums=(1,))
        return self._step[k]

    def _prefill_fn(self, m, act_len, want_logits):
        k = (m, act_len, want_logits)
        if k not in self._prefill:
            self._prefill[k] = jax.jit(functools.partial(M.prefill_chunk, self.cfg, act_len=act_len,
                                                         want_logits=want_logits), donate_argnums=(1,))
        return self._prefill[k]

    # -- core -------------------------------------------------------------------------------------
    def feed(self, tokens):
        """Absorb tokens into the state; returns device logits after the last one."""
        tokens = [int(t) for t in tokens]
        if len(self.history) + len(tokens) > self.n_ctx:
            raise ValueError('context overflow: %d + %d > n_ctx=%d' % (len(self.history), len(tokens), self.n_ctx))
        logits = self.pending_logits
        i = 0
        while i < len(tokens):
            rest = len(tokens) - i
            m = next((b for b in PREFILL_BUCKETS if b <= rest), None)
            pos = len(self.history)
            if m is None:
                act = self._act_len(pos + 1)
                _, logits, self.state = self._step_fn(act)(
                    self.params, self.state, jnp.int32(tokens[i]), jax.random.PRNGKey(0),
                    jnp.float32(0), jnp.int32(1), jnp.float32(1), jnp.float32(0))
                m = 1
            else:
                act = self._act_len(pos + m)
                last = (rest == m)
                logits, self.state = self._prefill_fn(m, act, last)(
                    self.params, self.state, jnp.asarray(tokens[i:i + m], jnp.int32))
            self.history.extend(tokens[i:i + m])
            i += m
        self.pending_logits = logits
        return logits

    def generate_ids(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=20, top_p=0.95, min_p=0.0,
                     seed=0, stop_ids=None, reuse_prefix=True):
        """Yield generated token ids.  With reuse_prefix, a prompt extending the absorbed history only
        processes the new suffix (multi-turn chat keeps recurrent state and KV cache)."""
        prompt_ids = [int(t) for t in prompt_ids]
        n = len(self.history)
        if not (reuse_prefix and n and len(prompt_ids) > n and prompt_ids[:n] == self.history):
            self.reset()
            n = 0
        stop = self.tokenizer.eot_ids if stop_ids is None else set(stop_ids)
        t0 = time.perf_counter()
        logits = self.feed(prompt_ids[n:])
        key = jax.random.PRNGKey(seed)
        args = (jnp.float32(temperature), jnp.int32(top_k), jnp.float32(top_p), jnp.float32(min_p))
        key, sub = jax.random.split(key)
        token = int(_sample_jit(logits, sub, *args))
        t1 = time.perf_counter()
        self.stats = {'prompt_tokens': len(prompt_ids) - n, 'reused_tokens': n, 'prefill_s': t1 - t0,
                      'completion_tokens': 0, 'decode_s': 0.0}
        produced = 0
        stop_arr = jnp.asarray(sorted(stop) or [-1], jnp.int32)
        while not (token in stop or produced >= max_new_tokens or len(self.history) >= self.n_ctx):
            yield token
            produced += 1
            # run up to `block` steps on device: feeds `token`, returns the tokens sampled after it
            budget = min(self.block, max_new_tokens - produced + 1, self.n_ctx - len(self.history))
            act = self._act_len(len(self.history) + budget)
            out, cnt, self.state, key, logits = self._block_fn(act)(
                self.params, self.state, jnp.int32(token), key, jnp.int32(budget), *args, stop_arr)
            out = np.asarray(out)[:int(cnt)].tolist()
            self.history.extend([token] + out[:-1])
            self.pending_logits = None            # logits belong to out[-1]'s predecessor; see below
            for t in out[:-1]:
                yield t
                produced += 1
            token = out[-1]
            self._last_logits = logits
            self.stats['completion_tokens'] = produced
            self.stats['decode_s'] = time.perf_counter() - t1
        self.stats['finish_reason'] = 'stop' if token in stop else 'length'

    def generate(self, prompt, **kw):
        """Yield decoded text pieces for a raw text prompt."""
        dec = StreamDecoder(self.tokenizer)
        for tok in self.generate_ids(self.tokenizer.encode(prompt), **kw):
            piece = dec.push(tok)
            if piece:
                yield piece

    def chat(self, messages, enable_thinking=False, tools=None, **kw):
        """Yield text pieces for an OpenAI-style message list using the model's own chat template."""
        tk = {} if tools is None else {'tools': tools}
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    enable_thinking=enable_thinking, **tk)
        yield from self.generate(prompt, **kw)

    def memory(self):
        s = jax.devices()[0].memory_stats()
        return {k: s[k] for k in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit') if k in s}


_sample_jit = jax.jit(sample_token)
