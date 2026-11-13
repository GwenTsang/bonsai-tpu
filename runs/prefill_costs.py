"""Calibrate existing executables; no new TPU program shapes are introduced."""
exec(open('/content/runs/load.py').read(), globals())
from bonsai_tpu.engine import Engine
import hashlib
import os

E = Engine(n_ctx=65536, preloaded=(G, CFG, PARAMS), max_snapshots=0,
           snapshot_every=0, speculate=False, park_bytes=0)
print('warmup_s', E.warmup(), flush=True)
text = E.tokenizer.encode(open('/content/tests/data/long_text.txt').read())
tokens = (text * (65536 // len(text) + 1))[:65536]
rows = []
for context in (0, 8192, 32768, 61440):
    # Populate real KV rows for the context. Repeated timing passes reuse their allocation;
    # recurrence is deliberately not compared in this microbenchmark.
    if len(E.history) < context:
        E.feed(tokens[len(E.history):context]).block_until_ready()
    for size in (64, 512, 1):
        times = []
        for trial in range(4):
            E.state['pos'] = jnp.int32(context)
            jax.block_until_ready(E.state)
            start = time.perf_counter()
            if size == 1:
                _, _, E.state, _, logits = E._block_fn()(
                    PARAMS, E.state, jnp.int32(tokens[context]), jax.random.PRNGKey(0), jnp.int32(1),
                    jnp.float32(0), jnp.int32(1), jnp.float32(1), jnp.float32(0),
                    jnp.full((8,), -1, jnp.int32))
            else:
                logits, E.state = E._prefill_fn(size)(
                    PARAMS, E.state, jnp.asarray(tokens[context:context + size], jnp.int32), jnp.int32(size))
            jax.block_until_ready((logits, E.state))
            if trial:
                times.append(time.perf_counter() - start)
        row = dict(context=context, size=size, seconds=times, median_s=float(np.median(times)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    E.state['pos'] = jnp.int32(context)
res = dict(rows=rows, n_ctx=E.n_ctx, kv=str(E.kv_dtype), device=str(jax.devices()[0]),
           jax_version=jax.__version__, commit=os.environ.get('BONSAI_COMMIT', ''),
           engine_sha256=hashlib.sha256(open('/content/bonsai_tpu/engine.py', 'rb').read()).hexdigest())
json.dump(res, open('/content/bonsai-results/prefill-costs.json', 'w'), indent=2)
