"""User-facing inference engine: model loading, prefill/decode scheduling, sampling, streaming, chat."""
import functools
import time
import numpy as np
import jax
import jax.numpy as jnp
from . import model as M
from .gguf import GGUF
from .tokenizer import BonsaiTokenizer, StreamDecoder

PREFILL_BUCKETS = (512, 64)      # compiled chunk sizes; any shorter remainder is padded into one
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


def decode_block(cfg, n_max, params, state, token, key, n, temperature, top_k, top_p, min_p, stop_ids):
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
        logits, st = M.decode_step(cfg, params, st, tok)
        k, sub = jax.random.split(k)
        nxt = sample_token(logits, sub, temperature, top_k, top_p, min_p)
        return i + 1, nxt, st, k, out.at[i].set(nxt), jnp.any(nxt == stop_ids), logits

    init = (jnp.int32(0), token, state, key, jnp.full((n_max,), -1, jnp.int32), jnp.bool_(False),
            jnp.zeros((cfg.vocab,), jnp.float32))
    i, _, state, key, out, _, logits = jax.lax.while_loop(cond, body, init)
    return out, i, state, key, logits


class Engine:
    def __init__(self, gguf_path, n_ctx=8192, kv_dtype='bfloat16', log=print, preloaded=None, block=8, max_programs=12,
                 max_snapshots=6, snapshot_every=4096, compile_cache='~/.cache/bonsai_tpu/xla'):
        self.block = int(block)
        self.log = log or (lambda *a, **k: None)
        if compile_cache:
            # compiled TPU programs persist across restarts: ~35 s compile becomes ~5 s load per program
            import os
            path = os.path.expanduser(compile_cache)
            os.makedirs(path, exist_ok=True)
            jax.config.update('jax_compilation_cache_dir', path)
            jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
            jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
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
        self._programs = {}
        self.max_snapshots = int(max_snapshots)
        self.snapshot_every = int(snapshot_every)
        self.max_programs = int(max_programs)
        self.reset()
        self.stats = {}

    # -- state ------------------------------------------------------------------------------------
    def reset(self):
        self.state = M.init_state(self.cfg, self.n_ctx, self.kv_dtype, slack=PREFILL_BUCKETS[0])
        self.history = []          # token ids already absorbed into the state
        self.pending_logits = None
        self.snapshots = []        # recurrent-state checkpoints, ascending position

    # Agent loops resend the whole conversation with small edits (re-rendered tool calls, stripped
    # reasoning).  The KV cache is position-addressed, so rewinding it is free, but the recurrent (GDN and
    # convolution) state cannot be rolled back; checkpoints of it (~155 MiB each, kept in HBM) let a request
    # resume from the nearest checkpoint at or before the first changed token instead of from zero.
    def snapshot(self):
        pos = len(self.history)
        if pos == 0 or self.max_snapshots <= 0 or any(sn['pos'] == pos for sn in self.snapshots):
            return
        conv, gdn = _copy_tree((self.state['conv'], self.state['gdn']))
        self.snapshots.append({'pos': pos, 'conv': conv, 'gdn': gdn})
        self.snapshots.sort(key=lambda sn: sn['pos'])
        while len(self.snapshots) > self.max_snapshots:
            # keep the two most recent; among the rest drop the one closest to its predecessor
            cand = range(0, len(self.snapshots) - 2)
            gaps = [self.snapshots[i]['pos'] - (self.snapshots[i - 1]['pos'] if i else 0) for i in cand]
            del self.snapshots[int(np.argmin(gaps))]

    def rewind(self, limit):
        """Restore the latest checkpoint with pos <= limit (or reset).  Returns the restored position."""
        self.snapshots = [sn for sn in self.snapshots if sn['pos'] <= limit]
        if not self.snapshots:
            snaps = []
            self.reset()
            self.snapshots = snaps
            return 0
        sn = self.snapshots[-1]
        conv, gdn = _copy_tree((sn['conv'], sn['gdn']))
        self.state = {'pos': jnp.int32(sn['pos']), 'conv': conv, 'gdn': gdn,
                      'k': self.state['k'], 'v': self.state['v']}
        self.history = self.history[:sn['pos']]
        self.pending_logits = None
        return sn['pos']

    def _program(self, key, build):
        """Compiled-program cache.  Every cached TPU executable pins ~1.1 GiB of host RAM, so the cache is
        bounded; exceeding it drops all programs (they recompile on demand)."""
        if key not in self._programs:
            if len(self._programs) >= self.max_programs:
                self._programs.clear()
                jax.clear_caches()
                try:
                    import ctypes
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except OSError:
                    pass
            self._programs[key] = build()
        return self._programs[key]

    def _block_fn(self):
        return self._program('block', lambda: jax.jit(
            functools.partial(decode_block, self.cfg, self.block), donate_argnums=(1,)))

    def _prefill_fn(self, m):
        return self._program(('prefill', m), lambda: jax.jit(
            functools.partial(M.prefill_chunk, self.cfg), donate_argnums=(1,)))

    def warmup(self):
        """Compile every program the engine uses (decode block and both prefill sizes) before serving."""
        t = time.perf_counter()
        self.reset()
        for m in PREFILL_BUCKETS:
            self.feed([0] * m)
        for _ in self.generate_ids([0, 1], max_new_tokens=self.block + 2, stop_ids=[-1]):
            pass
        self.reset()
        return time.perf_counter() - t

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
            m = PREFILL_BUCKETS[0] if rest > PREFILL_BUCKETS[-1] else PREFILL_BUCKETS[-1]
            n = min(m, rest)
            chunk = tokens[i:i + n] + [0] * (m - n)
            logits, self.state = self._prefill_fn(m)(
                self.params, self.state, jnp.asarray(chunk, jnp.int32), jnp.int32(n))
            before = len(self.history)
            self.history.extend(tokens[i:i + n])
            i += n
            if self.snapshot_every and before // self.snapshot_every != len(self.history) // self.snapshot_every:
                self.snapshot()
        self.pending_logits = logits
        return logits

    def generate_ids(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=20, top_p=0.95, min_p=0.0,
                     seed=0, stop_ids=None, reuse_prefix=True):
        """Yield generated token ids.  With reuse_prefix, a prompt extending the absorbed history only
        processes the new suffix (multi-turn chat keeps recurrent state and KV cache)."""
        prompt_ids = [int(t) for t in prompt_ids]
        if not prompt_ids:
            raise ValueError('empty prompt')
        if len(prompt_ids) >= self.n_ctx:
            raise ValueError('prompt has %d tokens; n_ctx is %d' % (len(prompt_ids), self.n_ctx))
        lcp = 0
        if reuse_prefix:
            for a_, b_ in zip(prompt_ids, self.history):
                if a_ != b_:
                    break
                lcp += 1
        n = len(self.history)
        if not (lcp == n and len(prompt_ids) > n):
            # diverged (or nothing new to feed): resume from a checkpoint that still leaves >= 1 token to feed
            n = self.rewind(min(lcp, len(prompt_ids) - 1))
        stop = self.tokenizer.eot_ids if stop_ids is None else set(stop_ids)
        t0 = time.perf_counter()
        logits = self.feed(prompt_ids[n:])
        self.snapshot()            # prompt end: the usual resume point of the next agent turn
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
            out, cnt, self.state, key, logits = self._block_fn()(
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
_copy_tree = jax.jit(lambda t: jax.tree.map(lambda x: x + jnp.zeros((), x.dtype), t))
