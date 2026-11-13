"""Host scheduler invariants; the model's numeric paths are compared on TPU."""
from functools import lru_cache

import numpy as np
import pytest

from bonsai_tpu.engine import Engine, prefill_plan


def test_planner_minimizes_cost_and_covers_only_real_tokens():
    # Independent exhaustive oracle for small token counts, including padding and a one-token tail.
    costs = ((0, {1: 2., 64: 5., 512: 26.}),)

    @lru_cache(None)
    def oracle(n):
        if n == 0:
            return 0.
        return min([costs[0][1][m] + oracle(max(0, n - m)) for m in (64, 512)]
                   + ([2.] if n == 1 else []))

    for n in range(1100):
        plan = prefill_plan(n, costs=costs)
        assert sum(costs[0][1][m] for m in plan) == oracle(n)
        assert not plan or sum(plan[:-1]) < n <= sum(plan)
        if 1 in plan:
            assert plan[-1] == 1 and sum(plan) == n


def test_planner_accounts_for_context_and_interpolates():
    costs = ((0, {1: 2., 64: 5., 512: 24.}),
             (8192, {1: 3., 64: 6., 512: 43.}))
    assert prefill_plan(320, 0, costs) == [512]
    assert prefill_plan(320, 4096, costs) == [64] * 5
    assert prefill_plan(320, 100000, costs) == [64] * 5
    assert prefill_plan(65, 0, costs) == [64, 1]
    assert prefill_plan(512, 0, costs) == [512]
    assert prefill_plan(0) == []
    with pytest.raises(ValueError):
        prefill_plan(-1)


def mock_engine():
    e = Engine.__new__(Engine)
    e.history = []
    e.state = {'pos': 0, 'absorbed': []}
    e.pending_logits = None
    e.n_ctx = 2048
    e.snapshot_every = 0
    e.calls, e.boundaries = [], []

    def apply(tokens, valid, size):
        real = [int(x) for x in tokens[:valid]]
        e.calls.append((size, valid))
        e.state = {'pos': e.state['pos'] + valid, 'absorbed': e.state['absorbed'] + real}
        return ('logits', real[-1])

    def prefill(size):
        def run(params, state, tokens, valid):
            return apply(np.asarray(tokens), int(valid), size), e.state
        return run

    e.params = None
    e._prefill_fn = prefill
    e._feed_one = lambda token: apply([token], 1, 1)
    e.snapshot = lambda: e.boundaries.append(len(e.history))
    return e


def test_feed_preserves_history_padding_logits_and_message_boundaries():
    e = mock_engine()
    tokens = list(range(322))
    logits = e.feed(tokens, snap_at=(65, 193))
    assert e.history == tokens == e.state['absorbed']
    assert e.state['pos'] == len(tokens)
    assert e.boundaries == [65, 193]
    assert e.calls == [(64, 64), (1, 1), (64, 64), (64, 64), (64, 64), (64, 64), (1, 1)]
    assert logits == e.pending_logits == ('logits', tokens[-1])
    assert e.feed([]) == logits
    assert e.calls[-1] == (1, 1)


def test_feed_tail_and_periodic_snapshot_near_context_limit():
    e = mock_engine()
    e.n_ctx = 577
    e.snapshot_every = 512
    e.feed(list(range(577)))
    assert e.calls == [(512, 512), (64, 64), (1, 1)]
    assert e.boundaries == [512]
    assert e.history == e.state['absorbed'] == list(range(577))
    with pytest.raises(ValueError, match='context overflow'):
        e.feed([1])
    assert e.state['pos'] == 577
