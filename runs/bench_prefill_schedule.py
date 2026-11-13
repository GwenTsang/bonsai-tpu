"""Paired, warm old/new feed benchmarks and TPU correctness checks for proposition 3.

Run with scripts/run.sh runs/bench_prefill_schedule.py. Timings exclude identical restoration of the
cached prefix, include host scheduling/dispatch, and synchronize all returned device buffers.
"""
exec(open('/content/runs/load.py').read(), globals())
import hashlib
import os
import shutil
from pathlib import Path
import bonsai_tpu.engine as em
from bonsai_tpu.engine import Engine, _copy_tree


def old_plan(count, position=0):
    plan = []
    while count:
        size = 512 if count > 64 else 64
        plan.append(size)
        count -= min(count, size)
    return plan


def compare_logits(candidate, reference):
    c, r = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    lc, lr = c - c.max(), r - r.max()
    lc -= np.log(np.exp(lc).sum())
    lr -= np.log(np.exp(lr).sum())
    return dict(kl=float(np.sum(np.exp(lr) * (lr - lc))),
                rel_l2=float(np.linalg.norm(c - r) / np.linalg.norm(r)),
                top1_equal=bool(c.argmax() == r.argmax()))


def relative_state(candidate, reference):
    num = sum(np.sum((np.asarray(c, np.float64) - r.astype(np.float64)) ** 2)
              for c, r in zip(candidate, reference))
    den = sum(np.sum(r.astype(np.float64) ** 2) for r in reference)
    return float(np.sqrt(num / max(den, 1e-30)))


E = Engine(n_ctx=65536, preloaded=(G, CFG, PARAMS), max_snapshots=0,
           snapshot_every=0, speculate=False, park_bytes=0)
print('warmup_s', E.warmup(), flush=True)
adaptive_plan = em.prefill_plan
text = E.tokenizer.encode(open('/content/tests/data/long_text.txt').read())
tokens = (text * (65536 // len(text) + 1))[:65536]
res = dict(commit=os.environ.get('BONSAI_COMMIT', ''), jax_version=jax.__version__,
           engine_sha256=hashlib.sha256(Path('/content/bonsai_tpu/engine.py').read_bytes()).hexdigest(),
           model_revision='839bd57', n_ctx=E.n_ctx, kv=str(E.kv_dtype), block=E.block,
           speculate=False, max_snapshots=0, costs=em.PREFILL_COSTS, trials=3, rows=[])
for context in (0, 8192, 32768, 61440):
    E.reset()
    if context:
        E.feed(tokens[:context]).block_until_ready()
    base = _copy_tree((E.state['conv'], E.state['gdn']))
    jax.block_until_ready(base)

    def restore():
        conv, gdn = _copy_tree(base)
        E.state = dict(pos=jnp.int32(context), conv=conv, gdn=gdn, k=E.state['k'], v=E.state['v'])
        E.history = list(tokens[:context])
        E.pending_logits = None
        E.snapshots = []
        jax.block_until_ready(E.state)

    for length in (1, 32, 64, 65, 128, 192, 256, 320, 512, 513, 577, 1024):
        suffix = tokens[context:context + length]
        samples = {'old': [], 'adaptive': []}
        # Alternate execution order to reduce systematic thermal/order bias.
        for trial in range(3):
            order = ('old', 'adaptive') if trial % 2 == 0 else ('adaptive', 'old')
            for policy in order:
                restore()
                em.prefill_plan = old_plan if policy == 'old' else adaptive_plan
                start = time.perf_counter()
                logits = E.feed(suffix)
                jax.block_until_ready((logits, E.state))
                samples[policy].append(time.perf_counter() - start)
                assert E.history == tokens[:context + length]
                assert int(E.state['pos']) == context + length
                if trial == 2:
                    if policy == 'old':
                        old_logits = np.asarray(logits)
                        old_rec = {k: [np.asarray(a) for a in E.state[k]] for k in ('conv', 'gdn')}
                        old_kv = {k: [np.asarray(a[:, context:context + length]) for a in E.state[k]]
                                  for k in ('k', 'v')}
                    else:
                        numeric = compare_logits(logits, old_logits)
                        numeric.update({k + '_rel': relative_state(E.state[k], old_rec[k])
                                        for k in ('conv', 'gdn')})
                        numeric.update({k + '_rel': relative_state(
                            [a[:, context:context + length] for a in E.state[k]], old_kv[k])
                            for k in ('k', 'v')})
        old_s, new_s = (float(np.median(samples[k])) for k in ('old', 'adaptive'))
        old_chunks, new_chunks = old_plan(length), adaptive_plan(length, context)
        row = dict(context=context, suffix_tokens=length, old_s=samples['old'], adaptive_s=samples['adaptive'],
                   old_median_s=old_s, adaptive_median_s=new_s, speedup=old_s / new_s,
                   latency_saved_ms=1000 * (old_s - new_s), old_chunks=old_chunks, adaptive_chunks=new_chunks,
                   old_padded_positions=sum(old_chunks), adaptive_padded_positions=sum(new_chunks),
                   numeric=numeric)
        res['rows'].append(row)
        print(json.dumps(row), flush=True)
        assert numeric['kl'] < 1e-3 and numeric['rel_l2'] < 0.01, row
        assert max(numeric[k + '_rel'] for k in ('conv', 'gdn', 'k', 'v')) < 0.03, row
    em.prefill_plan = adaptive_plan
    del base, old_rec, old_kv

# A real checkpoint rewind/replay across short message segments.
E.reset()
E.max_snapshots = 3
prompt = tokens[:322]
original = np.asarray(E.feed(prompt, snap_at=(65, 193)))
assert [s['pos'] for s in E.snapshots] == [65, 193]
assert E.rewind(100) == 65
replayed = np.asarray(E.feed(prompt[65:], snap_at=(193,)))
res['checkpoint_replay'] = compare_logits(replayed, original)
assert np.array_equal(original, replayed)
E.max_snapshots = 0

# Reuse the existing CPU-oracle data and decode block, without adding a compiled executable.
name = '/content/eval-ref.npz'
manifest_path = Path(name + '.manifest.json')
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    with open(name, 'wb') as dst:
        for part in manifest['parts']:
            with open('/content/' + part, 'rb') as src:
                shutil.copyfileobj(src, dst)
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == manifest['sha256']
    ref = np.load(name)
    res['reference'] = []
    for case in sorted({k.split('::')[0] for k in ref.files}):
        prompt = ref[case + '::prompt'].tolist()
        forced = ref[case + '::greedy'].tolist()
        for policy in ('old', 'adaptive'):
            E.reset()
            em.prefill_plan = old_plan if policy == 'old' else adaptive_plan
            logits = E.feed(prompt)
            metrics = []
            for i, target in enumerate(ref[case + '::logits']):
                metrics.append(compare_logits(logits, target))
                logits = E._feed_one(forced[i])
                E.history.append(forced[i])
            row = dict(case=case, policy=policy, steps=len(metrics),
                       top1=sum(m['top1_equal'] for m in metrics),
                       mean_kl=float(np.mean([m['kl'] for m in metrics])),
                       max_kl=max(m['kl'] for m in metrics))
            res['reference'].append(row)
            print('reference', json.dumps(row), flush=True)
    for policy in ('old', 'adaptive'):
        rows = [r for r in res['reference'] if r['policy'] == policy]
        assert sum(r['top1'] for r in rows) >= 318
        assert np.mean([r['mean_kl'] for r in rows]) < 1e-4
else:
    raise FileNotFoundError('CPU oracle manifest needed for numeric validation')

# Actual TTFT and subsequent greedy generation, including snapshots and prefix-reuse scheduling.
em.prefill_plan = adaptive_plan
res['generation'] = []
E.max_snapshots = 3
for length in (1, 65, 128, 320):
    ids = tokens[:length]
    outputs, ttfts = {}, {'old': [], 'adaptive': []}
    for trial in range(3):
        for policy in ('old', 'adaptive'):
            E.reset()
            em.prefill_plan = old_plan if policy == 'old' else adaptive_plan
            out = []
            start = time.perf_counter()
            for token in E.generate_ids(ids, max_new_tokens=32, stop_ids=[-1], reuse_prefix=False):
                if not out:
                    ttfts[policy].append(time.perf_counter() - start)
                out.append(token)
            outputs[policy] = out
    row = dict(prompt_tokens=length, ttft_s=ttfts, greedy_equal=outputs['old'] == outputs['adaptive'],
               outputs=outputs)
    if not row['greedy_equal']:
        # Changing BF16 chunk boundaries can flip near ties. Compare distributions at the first
        # mismatch under the SAME generated prefix, rather than comparing already-divergent text.
        mismatch = next(i for i, (a, b) in enumerate(zip(outputs['old'], outputs['adaptive'])) if a != b)
        pair = {}
        for policy in ('old', 'adaptive'):
            E.reset()
            em.prefill_plan = old_plan if policy == 'old' else adaptive_plan
            lg = E.feed(ids)
            for tok in outputs['old'][:mismatch]:
                lg = E._feed_one(tok)
                E.history.append(tok)
            pair[policy] = np.asarray(lg)
        row['first_mismatch'] = mismatch
        row['shared_prefix_metrics'] = compare_logits(pair['adaptive'], pair['old'])
        row['top_logits'] = {policy: [(int(t), float(lg[t])) for t in np.argsort(lg)[-5:][::-1]]
                             for policy, lg in pair.items()}
        assert row['shared_prefix_metrics']['kl'] < 1e-3
        assert row['shared_prefix_metrics']['rel_l2'] < 0.01
    res['generation'].append(row)
    print('generation', json.dumps(row), flush=True)
em.prefill_plan = adaptive_plan
res['validated'] = True
json.dump(res, open('/content/bonsai-results/prefill-schedule.json', 'w'), indent=2)
print('PREFILL_SCHEDULE_VALIDATED', flush=True)
