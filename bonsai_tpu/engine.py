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
MAX_STOP = 8
SPEC_K = 32          # positions verified per speculative pass (one compiled program)
SPEC_NGRAM = 6       # context n-gram that must match before a continuation is proposed
SPEC_MIN_DRAFT = 6   # shortest draft worth a batched pass
SPEC_BACKOFF = 24    # tokens generated normally after a rejected draft


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
    def __init__(self, gguf_path, n_ctx=8192, kv_dtype='bfloat16', log=print, preloaded=None, block=16, max_programs=12,
                 max_snapshots=6, snapshot_every=4096, compile_cache='~/.cache/bonsai_tpu/xla',
                 speculate=True):
        self.speculate = bool(speculate)
        self.spec_stats = {'rounds': 0, 'drafted': 0, 'accepted': 0, 'misses': 0}
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
        old = getattr(self, 'state', None)
        self.snapshots = []
        if old is None:
            self.state = M.init_state(self.cfg, self.n_ctx, self.kv_dtype, slack=PREFILL_BUCKETS[0])
        else:
            # KV rows are position-addressed and masked by pos: keep the buffers (no second multi-GiB
            # allocation), clear only the recurrent state
            self.state = {'pos': jnp.zeros((), jnp.int32), 'k': old['k'], 'v': old['v'],
                          'conv': [jnp.zeros_like(c) for c in old['conv']],
                          'gdn': [jnp.zeros_like(g) for g in old['gdn']]}
        self.history = []          # token ids already absorbed into the state
        self.pending_logits = None
        self.snapshots = []        # recurrent-state checkpoints, ascending position
        self._ngram, self._ngram_upto = {}, 0

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
        self._ngram, self._ngram_upto = {}, 0
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

    def _verify_fn(self):
        cfg = self.cfg

        def fn(params, state, tokens, n_valid, key, temperature, top_k, top_p, min_p):
            logits, state = M.verify_chunk(cfg, params, state, tokens, n_valid)
            keys = jax.random.split(key, tokens.shape[0])
            return jax.vmap(lambda l, k: sample_token(l, k, temperature, top_k, top_p, min_p))(logits, keys), state
        return self._program('verify', lambda: jax.jit(fn, donate_argnums=(1,)))

    def warmup(self):
        """Compile every program the engine uses (decode block and both prefill sizes) before serving."""
        t = time.perf_counter()
        self.reset()
        for m in PREFILL_BUCKETS:
            self.feed([0] * m)
        for _ in self.generate_ids([0, 1], max_new_tokens=self.block + 2, stop_ids=[-1]):
            pass
        if self.speculate:
            _, self.state = self._verify_fn()(self.params, self.state, jnp.zeros((SPEC_K,), jnp.int32), jnp.int32(1),
                              jax.random.PRNGKey(0), jnp.float32(0), jnp.int32(1), jnp.float32(1), jnp.float32(0))
        self.reset()
        return time.perf_counter() - t

    # -- core -------------------------------------------------------------------------------------
    def feed(self, tokens, snap_at=()):
        """Absorb tokens into the state; returns device logits after the last one.  `snap_at` lists absolute
        positions (message boundaries) at which to checkpoint the recurrent state on the way."""
        tokens = [int(t) for t in tokens]
        base = len(self.history)
        cuts = sorted(p - base for p in set(snap_at) if base < p < base + len(tokens))
        if cuts:
            logits = None
            for a_, b_ in zip([0] + cuts, cuts + [len(tokens)]):
                logits = self.feed(tokens[a_:b_])
                if b_ != len(tokens):
                    self.snapshot()
            return logits
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

    # -- prompt-lookup speculation ----------------------------------------------------------------
    # Agent output copies long spans of its context (file edits, quoted tool output).  When the last
    # SPEC_NGRAM tokens reoccur earlier in the context, the tokens that followed are proposed as a draft and
    # checked in ONE batched pass: decode is HBM-bandwidth-bound, so 32 positions cost about as much as two
    # single steps.  Each position's token is still drawn from the model's own distribution; a draft only
    # guesses that outcome, so the output distribution is unchanged.
    def _index_history(self):
        n = SPEC_NGRAM
        h = self.history
        start = max(self._ngram_upto, n)
        for end in range(start, len(h) + 1):
            self._ngram[tuple(h[end - n:end])] = end          # latest occurrence wins
        self._ngram_upto = max(start, len(h) + 1)

    def _draft(self, token, limit):
        h = self.history
        if len(h) + 1 < SPEC_NGRAM + 1:
            return []
        key = tuple(h[len(h) - SPEC_NGRAM + 1:] + [token])
        end = self._ngram.get(key)
        if end is None:
            return []
        return h[end:end + min(SPEC_K - 1, limit)]

    def generate_ids(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=20, top_p=0.95, min_p=0.0,
                     seed=0, stop_ids=None, reuse_prefix=True, snap_at=()):
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
        logits = self.feed(prompt_ids[n:], snap_at)
        self.snapshot()            # prompt end: the usual resume point of the next agent turn
        key = jax.random.PRNGKey(seed)
        args = (jnp.float32(temperature), jnp.int32(top_k), jnp.float32(top_p), jnp.float32(min_p))
        key, sub = jax.random.split(key)
        token = int(_sample_jit(logits, sub, *args))
        t1 = time.perf_counter()
        self.stats = {'prompt_tokens': len(prompt_ids) - n, 'reused_tokens': n, 'prefill_s': t1 - t0,
                      'completion_tokens': 0, 'decode_s': 0.0}
        produced = 0
        blocks_done = 0
        cooldown = 0
        if len(stop) > MAX_STOP:
            raise ValueError('at most %d stop ids' % MAX_STOP)
        stop_arr = jnp.asarray((sorted(stop) + [-1] * MAX_STOP)[:MAX_STOP], jnp.int32)   # fixed shape: one program
        while not (token in stop or produced >= max_new_tokens or len(self.history) >= self.n_ctx):
            yield token
            produced += 1
            # run up to `block` steps on device: feeds `token`, returns the tokens sampled after it
            if self.speculate and cooldown <= 0:
                self._index_history()
                room = min(max_new_tokens - produced, self.n_ctx - len(self.history) - 1)
                draft = self._draft(token, room)
                if len(draft) >= SPEC_MIN_DRAFT:
                    n_in = 1 + len(draft)
                    old_pos = len(self.history)
                    saved = _copy_tree((self.state['conv'], self.state['gdn']))
                    chunk = jnp.asarray([token] + draft + [0] * (SPEC_K - n_in), jnp.int32)
                    key, sub = jax.random.split(key)
                    preds, self.state = self._verify_fn()(self.params, self.state, chunk, jnp.int32(n_in), sub, *args)
                    preds = np.asarray(preds)[:n_in].tolist()
                    acc = 0
                    while acc < len(draft) and preds[acc] == draft[acc] and preds[acc] not in stop:
                        acc += 1
                    self.spec_stats['rounds'] += 1
                    self.spec_stats['drafted'] += len(draft)
                    self.spec_stats['accepted'] += acc
                    if acc + 1 < n_in:
                        # the state absorbed rejected draft tokens: restore and (if anything was accepted)
                        # re-absorb exactly token + accepted drafts
                        self.state = {'pos': jnp.int32(old_pos), 'conv': saved[0], 'gdn': saved[1],
                                      'k': self.state['k'], 'v': self.state['v']}
                        if acc == 0:
                            self.spec_stats['misses'] += 1
                            cooldown = SPEC_BACKOFF
                        else:
                            _, self.state = self._verify_fn()(self.params, self.state, chunk, jnp.int32(acc + 1),
                                                              sub, *args)
                    if acc > 0:
                        self.history.extend([token] + draft[:acc])
                        for t in draft[:acc]:
                            yield t
                            produced += 1
                        token = preds[acc]
                        self.pending_logits = None
                        self.stats['completion_tokens'] = produced
                        self.stats['decode_s'] = time.perf_counter() - t1
                        continue
            # ramp 1, 2, 4, ... so the first tokens stream immediately, then amortise dispatch
            budget = min(self.block, 1 << min(blocks_done, 10), max_new_tokens - produced + 1,
                         self.n_ctx - len(self.history))
            blocks_done += 1
            out, cnt, self.state, key, logits = self._block_fn()(
                self.params, self.state, jnp.int32(token), key, jnp.int32(budget), *args, stop_arr)
            out = np.asarray(out)[:int(cnt)].tolist()
            cooldown -= len(out)
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
