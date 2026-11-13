"""Command line: python -m bonsai_tpu.cli {generate,chat,serve,bench} --model <gguf> [...]"""
import argparse
import sys
import time


def main(argv=None):
    p = argparse.ArgumentParser(prog='bonsai_tpu')
    p.add_argument('mode', choices=['generate', 'chat', 'serve', 'bench'])
    p.add_argument('--model', default=None, help='GGUF path (default: download huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF, Ternary-Bonsai-PTQ1_0 file)')
    p.add_argument('--ctx', type=int, default=65536, help='context length in tokens')
    p.add_argument('--kv', default='bfloat16', choices=['float32', 'bfloat16'], help='KV cache element type')
    p.add_argument('--prompt', default=None)
    p.add_argument('--system', default=None)
    p.add_argument('-n', '--max-tokens', type=int, default=512)
    p.add_argument('--temp', type=float, default=None, help='default: model card value (1.0); 0 = greedy')
    p.add_argument('--top-k', type=int, default=None)
    p.add_argument('--top-p', type=float, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--thinking', default='default', choices=['default', 'off', 'low', 'medium', 'xhigh'],
                   help="the model's thinking parameter (default: model default = on, effort xhigh)")
    p.add_argument('--block', type=int, default=8, help='tokens generated per device round trip')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args(argv)

    from .engine import Engine
    print('loading model ...', file=sys.stderr, flush=True)
    eng = Engine(a.model, n_ctx=a.ctx, kv_dtype=a.kv, block=a.block)
    md = eng.gguf.metadata
    samp = dict(max_new_tokens=a.max_tokens, seed=a.seed,
                temperature=md.get('general.sampling.temp', 1.0) if a.temp is None else a.temp,
                top_k=md.get('general.sampling.top_k', 20) if a.top_k is None else a.top_k,
                top_p=md.get('general.sampling.top_p', 0.95) if a.top_p is None else a.top_p)
    print('loaded in %.1f s; HBM %.2f GiB' % (eng.load_seconds, eng.memory()['bytes_in_use'] / 2 ** 30),
          file=sys.stderr, flush=True)

    def report():
        s = eng.stats
        if s.get('decode_s'):
            print('\n[prompt %d tok (%d reused) in %.2f s | %d tok at %.1f tok/s]' % (
                s['prompt_tokens'], s['reused_tokens'], s['prefill_s'], s['completion_tokens'],
                s['completion_tokens'] / s['decode_s']), file=sys.stderr, flush=True)

    think = None if a.thinking == 'default' else a.thinking != 'off'
    effort = a.thinking if a.thinking in ('low', 'medium', 'xhigh') else None
    if a.mode == 'generate':
        for piece in eng.generate(a.prompt or sys.stdin.read(), **samp):
            print(piece, end='', flush=True)
        report()
    elif a.mode == 'chat':
        messages = [{'role': 'system', 'content': a.system}] if a.system else []
        while True:
            try:
                user = input('\n> ')
            except (EOFError, KeyboardInterrupt):
                break
            if user.strip() in ('/exit', '/quit'):
                break
            if user.strip() == '/reset':
                messages = messages[:1] if a.system else []
                eng.reset()
                continue
            messages.append({'role': 'user', 'content': user})
            reply = ''
            for piece in eng.chat(messages, thinking=think, reasoning_effort=effort, **samp):
                reply += piece
                print(piece, end='', flush=True)
            content = reply.split('</think>')[-1].strip() if think is not False else reply.strip()
            messages.append({'role': 'assistant', 'content': content})
            report()
    elif a.mode == 'serve':
        from .server import serve
        httpd = serve(eng, a.host, a.port, default_thinking=think is not False)
        print('serving OpenAI-compatible API on http://%s:%d/v1' % (a.host, a.port), file=sys.stderr, flush=True)
        httpd.serve_forever()
    else:
        ids = eng.tokenizer.encode(a.prompt or 'Write a long story about a lighthouse keeper.')
        for trial in range(3):
            eng.reset()
            t = time.perf_counter()
            n = sum(1 for _ in eng.generate_ids(ids, max_new_tokens=a.max_tokens, temperature=0.0, stop_ids=[-1]))
            s = eng.stats
            print('trial %d: %d tokens, %.1f tok/s decode, prefill %.2f s, total %.2f s' % (
                trial, n, s['completion_tokens'] / s['decode_s'], s['prefill_s'], time.perf_counter() - t), flush=True)


if __name__ == '__main__':
    main()
