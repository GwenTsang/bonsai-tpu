"""CPU-side tests of engine building blocks (no TPU needed): B4 packing, kernel algebra, GDN, Hadamard,
sampler, tokenizer.  TPU kernel execution is validated on-device by runs/tpu_tests.py."""
import json
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from bonsai_tpu import kernels, model as M
from bonsai_tpu.hadamard import fwht, fwht_batched, fwht_mxu
from bonsai_tpu.pack import b4_from_ptq1, b4_to_dense
from bonsai_tpu.quantization import encode_ptq1

HI = jax.lax.Precision.HIGHEST


def random_matrix(rng, n, k):
    codes = rng.integers(-1, 2, size=(n, k // 128, 128)).astype(np.int8)
    scales = (rng.random((n, k // 128)) + 0.25).astype(np.float16)
    blocks = encode_ptq1(codes, scales)
    dense = (codes.astype(np.float32) * scales.astype(np.float32)[..., None]).reshape(n, k)
    return blocks[..., :26], scales, dense


def test_b4_packing_roundtrip():
    rng = np.random.default_rng(0)
    payload, scales, dense = random_matrix(rng, 256, 2048)
    mat = b4_from_ptq1(payload, scales, chunk=100)       # exercises chunked conversion
    assert mat['w'].shape == (512, 256) and mat['w'].dtype == jnp.uint8
    np.testing.assert_array_equal(b4_to_dense(mat), dense)


def test_block_diagonal_gemv_algebra():
    """Emulate the Pallas GEMV kernel in NumPy integers and compare with the float64 product."""
    rng = np.random.default_rng(1)
    n, k = 128, 2048
    payload, scales, dense = random_matrix(rng, n, k)
    mat = b4_from_ptq1(payload, scales)
    x = rng.standard_normal(k).astype(np.float32)
    xb, xsum, alpha = (np.asarray(v) for v in kernels.prep_gemv(jnp.asarray(x)))
    w, s = np.asarray(mat['w']), np.asarray(mat['s']).view(np.float16).astype(np.float32)
    ng, bk = kernels.NG, kernels.BK
    y = np.zeros(n, np.float64)
    for t in range(k // bk):
        wv = w[t * bk // 4:(t + 1) * bk // 4]
        planes = np.concatenate([wv & 3, (wv >> 2) & 3, (wv >> 4) & 3, wv >> 6], axis=0).astype(np.int64)
        p = xb[t * 2 * ng:(t + 1) * 2 * ng].astype(np.int64) @ planes
        exact = (p[:ng] << 8) + p[ng:] - xsum[t * ng:(t + 1) * ng, :1]
        y += (exact * s[t * ng:(t + 1) * ng].astype(np.float64)).sum(0)
    ref = dense.astype(np.float64) @ x.astype(np.float64)
    assert np.linalg.norm(y / alpha - ref) / np.linalg.norm(ref) < 1e-4       # 16-bit activations
    q = np.round(x * alpha)
    np.testing.assert_allclose(y, dense.astype(np.float64) @ q, rtol=1e-9, atol=1e-6)   # integer-exact part


def test_half_scale_decode_matches_ieee():
    """Integer decode of raw half bits used inside the kernels, over every half in [0, 128)."""
    bits = np.arange(0, np.float16(128).view(np.int16), dtype=np.int16)
    got = np.asarray(kernels._scales_f32(jnp.asarray(bits)))
    np.testing.assert_array_equal(got, bits.view(np.float16).astype(np.float32))


def test_gemm_column_order_matches_planes():
    """The GEMM wrapper permutes activations to the (tile, j, g, i) order of the kernel's shift planes."""
    k = 2048
    x = np.arange(k, dtype=np.float32)[None]
    xp = x.reshape(1, k // 1024, 8, 4, 32).transpose(0, 1, 3, 2, 4).reshape(k)
    for t in range(2):
        for j in range(4):
            for g in range(8):
                for i in (0, 31):
                    assert xp[t * 1024 + j * 256 + g * 32 + i] == t * 1024 + 128 * g + 32 * j + i


def test_gdn_chunked_matches_scan():
    rng = np.random.default_rng(2)
    t, hv, d = 77, 6, 16
    q = rng.standard_normal((t, hv, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k = rng.standard_normal((t, hv, d)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((t, hv, d)).astype(np.float32)
    gate = -np.abs(rng.standard_normal((t, hv))).astype(np.float32) * 0.1
    beta = rng.random((t, hv)).astype(np.float32)
    s0 = rng.standard_normal((hv, d, d)).astype(np.float32) * 0.1
    o1, s1 = M.gdn_scan(q, k, v, gate, beta, s0)
    for chunk in (8, 32, 128):
        o2, s2 = M.gdn_chunked(*(jnp.asarray(a) for a in (q, k, v, gate, beta, s0)), chunk=chunk)
        np.testing.assert_allclose(o2, o1, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(s2, s1, rtol=2e-4, atol=2e-5)


def test_hadamard_variants_agree():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((3, 2048)) * np.exp(rng.standard_normal((3, 2048)))).astype(np.float32)
    ref = fwht(x.astype(np.float64), 1024)
    np.testing.assert_allclose(fwht_mxu(jnp.asarray(x[0]), jnp, HI), ref[0], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fwht_batched(jnp.asarray(x), jnp, jax.lax), ref, rtol=1e-3, atol=1e-3)
    err = np.linalg.norm(np.asarray(fwht_batched(jnp.asarray(x), jnp, jax.lax)) - ref) / np.linalg.norm(ref)
    assert err < 5e-5


def test_two_level_topk_is_exact():
    from bonsai_tpu.engine import top_candidates, TOPK
    rng = np.random.default_rng(4)
    for _ in range(5):
        logits = jnp.asarray(rng.standard_normal(248320).astype(np.float32))
        v, i = top_candidates(logits)
        rv, ri = jax.lax.top_k(logits, TOPK)
        np.testing.assert_array_equal(i, ri)
        np.testing.assert_array_equal(v, rv)


def test_sampler_modes():
    from bonsai_tpu.engine import sample_token
    logits = jnp.zeros(248320).at[1234].set(5.0).at[77].set(4.0)
    key = jax.random.PRNGKey(0)
    assert int(sample_token(logits, key, 0.0, 20, 0.95, 0.0)) == 1234                      # greedy
    assert int(sample_token(logits, key, 1.0, 1, 1.0, 0.0)) == 1234                        # top_k = 1
    draws = {int(sample_token(logits, jax.random.PRNGKey(s), 1.0, 2, 1.0, 0.0)) for s in range(40)}
    assert draws == {1234, 77}                                                             # top_k = 2
    draws = {int(sample_token(logits, jax.random.PRNGKey(s), 1.0, 20, 0.5, 0.0)) for s in range(40)}
    assert draws == {1234}                                                                 # nucleus cut


@pytest.fixture(scope='module')
def tokenizer():
    path = Path('weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
    if not path.exists():
        pytest.skip('released GGUF not present locally')
    from bonsai_tpu.gguf import GGUF
    from bonsai_tpu.tokenizer import BonsaiTokenizer
    return BonsaiTokenizer(GGUF(path).metadata)


def test_tokenizer_matches_llama_cpp(tokenizer):
    cases = [l.rstrip('\n').replace('\\n', '\n') for l in open('tests/data/tokenizer_cases.txt')]
    expected = json.load(open('tests/data/tokenizer_expected.json'))
    for text, ids in zip(cases, expected):
        assert tokenizer.encode(text) == ids
        assert tokenizer.decode(ids, skip_special=False) == text


def test_chat_template_and_stream_decoder(tokenizer):
    from bonsai_tpu.tokenizer import StreamDecoder
    p = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hi'}], enable_thinking=False)
    assert p == '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    text = 'Ünïcödé 日本語 🎉 ok'
    dec = StreamDecoder(tokenizer)
    assert ''.join(dec.push(t) for t in tokenizer.encode(text)) == text


def test_tool_call_parsing_and_splitter():
    from bonsai_tpu.server import OutputSplitter, parse_tool_calls, normalize_messages
    raw = ('Let me think.\n</think>\n\nI will check.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n'
           '/tmp/a.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>')
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'}}}}}]
    sp = OutputSplitter(True, True, [])
    events = []
    for i in range(0, len(raw), 3):                       # arbitrary chunk boundaries
        events += sp.push(raw[i:i + 3])
    events += sp.push('', final=True)
    assert ''.join(t for k, t in events if k == 'reasoning').strip() == 'Let me think.'
    assert ''.join(t for k, t in events if k == 'content') == 'I will check.'
    reasoning, content, tool = sp.result()
    calls = parse_tool_calls(tool, tools)
    assert len(calls) == 1 and calls[0]['function']['name'] == 'read_file'
    assert json.loads(calls[0]['function']['arguments']) == {'path': '/tmp/a.txt', 'limit': 20}
    # stop strings never leak, even when split across pieces
    sp = OutputSplitter(False, False, ['STOP'])
    out = ''.join(t for p in ('abc S', 'TO', 'P def') for k, t in sp.push(p))
    assert out == 'abc ' and sp.stopped
    msgs = normalize_messages([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': None,
                               'tool_calls': [{'id': 'x', 'type': 'function', 'function': {
                                   'name': 'f', 'arguments': '{"a": 1}'}}]}])
    assert msgs[1]['content'] == '' and msgs[1]['tool_calls'][0]['function']['arguments'] == {'a': 1}


def test_tool_roundtrip_through_template(tokenizer):
    """A parsed tool call re-rendered by the chat template reproduces the model-native text."""
    from bonsai_tpu.server import normalize_messages
    tools = [{'type': 'function', 'function': {'name': 'f', 'description': 'd', 'parameters': {
        'type': 'object', 'properties': {'a': {'type': 'integer'}}}}}]
    msgs = normalize_messages([
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
            'name': 'f', 'arguments': '{"a": 1}'}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'result!'}])
    p = tokenizer.apply_chat_template(msgs, tools=tools, enable_thinking=False)
    assert '<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>' in p
    assert '<tool_response>\nresult!\n</tool_response>' in p


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_streaming_attention_matches_dense(dtype):
    rng = np.random.default_rng(5)
    g, r, hd, s = 2, 5, 16, 3 * M.ATT_BLOCK
    q = rng.standard_normal((g, r, hd)).astype(np.float32)
    k = rng.standard_normal((g, s, hd)).astype(np.float32).astype(dtype)
    v = rng.standard_normal((g, s, hd)).astype(np.float32).astype(dtype)
    qpos = np.array([0, 7, M.ATT_BLOCK - 1, M.ATT_BLOCK, 2 * M.ATT_BLOCK + 17])
    out = np.asarray(M.attend(jnp.asarray(q), jnp.asarray(qpos), jnp.asarray(k), jnp.asarray(v), 0.25))
    qd = q.astype(dtype).astype(np.float64)
    sc = np.einsum('grd,gsd->grs', qd, k.astype(np.float64)) * 0.25
    sc = np.where(np.arange(s)[None, None, :] <= qpos[None, :, None], sc, -np.inf)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = np.einsum('grs,gsd->grd', p, v.astype(np.float64))
    np.testing.assert_allclose(out, ref, rtol=2e-2 if dtype == 'bfloat16' else 1e-4, atol=2e-2 if dtype == 'bfloat16' else 1e-5)


def test_segment_cached_chat_tokenization_is_exact(tokenizer):
    msgs = [{'role': 'system', 'content': 'You are an agent. Ünïcödé ok.'},
            {'role': 'user', 'content': 'Read the file  <tool_call> literally?\n\n  spaces  '},
            {'role': 'assistant', 'content': 'def f(x):\n    return x * 2\n'},
            {'role': 'user', 'content': 'thanks'}]
    text = tokenizer.apply_chat_template(msgs, enable_thinking=False)
    for _ in range(2):                       # second pass is served from the cache
        ids, bounds = tokenizer.encode_chat(text)
        assert ids == tokenizer.encode(text)
    assert bounds[0] == 0 and len(bounds) == 5
    assert all(ids[b] == tokenizer.tokens.index('<|im_start|>') for b in bounds)


def test_thinking_parameter_spellings():
    from bonsai_tpu.server import thinking_options as t
    assert t({}, True) == (True, None) and t({}, False) == (False, None)
    assert t({'thinking': False}, True) == (False, None)
    assert t({'thinking': {'type': 'disabled'}}, True) == (False, None)
    assert t({'thinking': {'type': 'enabled'}}, False) == (True, None)
    assert t({'enable_thinking': True, 'reasoning_effort': 'high'}, False) == (True, 'xhigh')
    assert t({'reasoning_effort': 'low'}, False) == (True, 'low')
    assert t({'reasoning_effort': 'none'}, True) == (False, None)
    assert t({'chat_template_kwargs': {'enable_thinking': False}}, True) == (False, None)
    assert t({'thinking': 'medium'}, False) == (True, 'medium')
    assert t({'enable_thinking': False, 'reasoning_effort': 'high'}, True) == (False, None)


def test_default_model_is_the_qwen38_based_release():
    from bonsai_tpu import defaults
    assert defaults.HF_REPO == 'prism-ml/Ternary-Bonsai-2-27B-gguf'
    assert defaults.resolve_model('/some/path.gguf') == '/some/path.gguf'
