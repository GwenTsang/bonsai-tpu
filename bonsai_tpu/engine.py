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


def sample_token(logits, key, temperature, top_k, top_p, min_p):
    """In-graph sampler over the top-TOPK candidates; temperature <= 0 selects argmax."""
    vals, idx = jax.lax.top_k(logits, TOPK)
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


class Engine:
    def __init__(self, gguf_path, n_ctx=8192, kv_dtype='float32', log=print, preloaded=None):
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
        while True:
            if token in stop or produced >= max_new_tokens or len(self.history) >= self.n_ctx:
                break
            yield token
            produced += 1
            key, sub = jax.random.split(key)
            act = self._act_len(len(self.history) + 1)
            nxt, logits, self.state = self._step_fn(act)(self.params, self.state, jnp.int32(token), sub, *args)
            self.history.append(token)
            self.pending_logits = logits
            token = int(nxt)
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
