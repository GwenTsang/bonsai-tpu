"""Client-side agent-loop benchmark against the OpenAI-compatible server (run through scripts/tunnel.sh).
Each turn resends the whole conversation (as agent harnesses do), streams the reply, executes fake tools that
return ~2K tokens of file text, and records TTFT, wall time, cached vs fed tokens and decode speed."""
import json, random, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
random.seed(0)
words = 'alpha beta gamma delta epsilon river stone engine ledger harbor signal copper tide archive bridge'.split()
def fake_file(i):
    lines = ['# module_%d.py' % i] + ['def fn_%d_%d(x):  # %s' % (i, j, ' '.join(random.choices(words, k=6))) +
                                       '\n    return x * %d + %d' % (j, random.randint(0, 999)) for j in range(60)]
    return '\n'.join(lines) + '\nSECRET_%d = %d\n' % (i, 1000 + 7 * i)
tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file from the repository.',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
msgs = [{'role': 'system', 'content': 'You are a coding agent. To answer you must read files with the read_file tool, one file per step.'},
        {'role': 'user', 'content': 'Read module_0.py through module_%d.py one at a time (one tool call per reply), then tell me the sum of all SECRET_i values.' % (TURNS - 2)}]
rows = []
for turn in range(TURNS):
    body = {'model': 'ternary-bonsai-2-27b', 'messages': msgs, 'tools': tools, 'temperature': 0, 'max_tokens': 600, 'stream': True}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter(); ttft = None; content = ''; calls = []; last = None
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        ev = json.loads(line[6:]); d = ev['choices'][0].get('delta', {})
        if ttft is None and (d.get('content') or d.get('tool_calls') or d.get('reasoning_content')):
            ttft = time.perf_counter() - t0
        content += d.get('content') or ''
        calls += d.get('tool_calls') or []
        if 'timings' in ev:
            last = ev
    wall = time.perf_counter() - t0
    tm = last['timings']
    rows.append(dict(turn=turn, prompt=last['usage']['prompt_tokens'], cached=tm['cache_n'], fed=tm['prompt_n'],
                     host_ms=round(tm['host_prepare_ms'], 1), prefill_ms=round(tm['prompt_ms'], 1), gen=tm['predicted_n'],
                     tok_s=round(tm.get('predicted_per_second', 0), 1), ttft_s=round(ttft or wall, 3), wall_s=round(wall, 3),
                     finish=last['choices'][0]['finish_reason']))
    print(json.dumps(rows[-1]), flush=True)
    m = {'role': 'assistant', 'content': content}
    if calls:
        m['tool_calls'] = [{k: c[k] for k in ('id', 'type', 'function')} for c in calls]
    msgs.append(m)
    if not calls:
        print('FINAL:', content[:300].replace('\n', ' '))
        break
    for c in calls:
        path = json.loads(c['function']['arguments']).get('path', '')
        digits = ''.join(ch for ch in path if ch.isdigit())
        msgs.append({'role': 'tool', 'tool_call_id': c['id'], 'content': fake_file(int(digits or 0))})
print('expected sum', sum(1000 + 7 * i for i in range(TURNS - 1)))
tot = lambda k: sum(r[k] for r in rows)
print(json.dumps(dict(turns=len(rows), total_wall_s=round(tot('wall_s'), 2), total_fed=tot('fed'), total_cached=tot('cached'),
                      total_gen=tot('gen'), mean_ttft_s=round(tot('ttft_s') / len(rows), 3))))
json.dump(rows, open(sys.argv[3] if len(sys.argv) > 3 else '/tmp/agent-loop-client.json', 'w'), indent=1)
