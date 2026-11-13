"""Minimal OpenAI-compatible HTTP server (stdlib only): /v1/chat/completions, /v1/completions, /v1/models,
/health, /metrics.  One request at a time (single-user engine); streaming via server-sent events."""
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .defaults import MODEL_NAME
from .tokenizer import StreamDecoder


TOOL_OPEN = '<tool_call>'
_TOOL_RE = re.compile(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', re.S)
_PARAM_RE = re.compile(r'<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>', re.S)
EFFORT = {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'xhigh', 'xhigh': 'xhigh', 'max': 'xhigh'}


def thinking_options(body, default):
    """Resolve the model's `thinking` switch and reasoning effort from the request.

    Accepted spellings (first match wins): `thinking` (bool, "on"/"off", or {"type": "enabled"|"disabled"}),
    `enable_thinking`, `chat_template_kwargs.{enable_thinking, thinking, reasoning_effort}`,
    `reasoning_effort` / `reasoning.effort` ("none"/"off" disables; minimal..max map onto the template's
    low / medium / xhigh).  Returns (enabled: bool, effort or None)."""
    ctk = body.get('chat_template_kwargs') or {}
    effort = body.get('reasoning_effort') or (body.get('reasoning') or {}).get('effort') or ctk.get('reasoning_effort')
    enabled = None
    for v in (body.get('thinking'), body.get('enable_thinking'), ctk.get('enable_thinking'), ctk.get('thinking')):
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get('type', 'enabled')
        if isinstance(v, str):
            if v.lower() in EFFORT and not effort:
                effort = v.lower()
            v = v.lower() not in ('disabled', 'off', 'false', 'none', '0')
        enabled = bool(v)
        break
    if isinstance(effort, str) and effort.lower() in ('none', 'off', 'disabled'):
        enabled, effort = (False if enabled is None else enabled), None
    if enabled is None:
        enabled = True if effort else default
    return enabled, (EFFORT.get(str(effort).lower()) if effort and enabled else None)


def _sampling(body, default_max):
    temp = body.get('temperature')
    return dict(max_new_tokens=int(body.get('max_tokens') or body.get('max_completion_tokens') or default_max),
                temperature=float(1.0 if temp is None else temp), top_p=float(body.get('top_p') or 0.95),
                top_k=int(body.get('top_k') or 20), min_p=float(body.get('min_p') or 0.0),
                seed=int(body.get('seed') or time.time_ns() % (2 ** 31)))


def parse_tool_calls(text, tools=None):
    """Model-native <tool_call><function=..><parameter=..> blocks -> OpenAI tool_calls."""
    schemas = {}
    for t in tools or []:
        f = t.get('function', t)
        schemas[f.get('name')] = (f.get('parameters') or {}).get('properties') or {}
    calls = []
    for name, inner in _TOOL_RE.findall(text):
        name = name.strip()
        args = {}
        for key, raw in _PARAM_RE.findall(inner):
            kind = (schemas.get(name, {}).get(key) or {}).get('type')
            val = raw
            if kind != 'string':
                try:
                    val = json.loads(raw)
                except ValueError:
                    val = raw
                if kind is None and not isinstance(val, (dict, list, bool, int, float)) and val is not None:
                    val = raw
            args[key.strip()] = val
        calls.append({'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
    return calls


def normalize_messages(messages):
    """OpenAI wire format -> what the model's chat template expects."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get('role') == 'developer':
            m['role'] = 'system'
        if m.get('content') is None:
            m['content'] = ''
        if m.get('tool_calls'):
            calls = []
            for c in m['tool_calls']:
                f = dict(c.get('function', c))
                if isinstance(f.get('arguments'), str):
                    try:
                        f['arguments'] = json.loads(f['arguments']) if f['arguments'].strip() else {}
                    except ValueError:
                        f['arguments'] = {'_raw': f['arguments']}
                calls.append({'type': 'function', 'function': f})
            m['tool_calls'] = calls
        out.append(m)
    # the template requires the system message first; merge any later ones into it
    sys_msgs = [m for m in out if m['role'] == 'system']
    if len(sys_msgs) > 1 or (sys_msgs and out[0]['role'] != 'system'):
        def text(c):
            return c if isinstance(c, str) else ''.join(p.get('text', '') for p in c if isinstance(p, dict))
        merged = {'role': 'system', 'content': '\n\n'.join(text(m['content']) for m in sys_msgs)}
        out = [merged] + [m for m in out if m['role'] != 'system']
    return out


class OutputSplitter:
    """Incrementally separates reasoning, visible content and the tool-call tail of the raw model text."""

    def __init__(self, thinking, tools, stops):
        self.thinking, self.tools, self.stops = thinking, tools, stops
        self.text = ''
        self.sent = {'reasoning': 0, 'content': 0}
        self.stopped = False

    def _parts(self):
        t = self.text
        reasoning = ''
        if self.thinking:
            if '</think>' in t:
                reasoning, t = t.split('</think>', 1)
                t = t.lstrip('\n')
            else:
                return t, '', '', False
        tool = ''
        if self.tools and TOOL_OPEN in t:
            i = t.index(TOOL_OPEN)
            t, tool = t[:i], t[i:]
        return reasoning, t, tool, True

    def push(self, piece, final=False):
        """Returns [(kind, delta)] that are safe to emit now."""
        self.text += piece
        reasoning, content, tool, think_done = self._parts()
        for s in self.stops:
            if s in content:
                content, self.stopped, final = content[:content.index(s)], True, True
        hold = 0 if final else max([len(s) for s in self.stops] + [len(TOOL_OPEN) if self.tools else 0, 1]) - 1
        if tool:
            content = content.rstrip('\n')
        out = []
        r_safe = len(reasoning) if (think_done or final) else max(len(reasoning) - len('</think>'), 0)
        if r_safe > self.sent['reasoning']:
            out.append(('reasoning', reasoning[self.sent['reasoning']:r_safe]))
            self.sent['reasoning'] = r_safe
        c_safe = len(content) if (tool or final) else max(len(content) - hold, 0)
        if self.tools and not (tool or final):
            c_safe = len(content[:c_safe].rstrip())          # trailing blank lines may precede a tool call
        if c_safe > self.sent['content']:
            out.append(('content', content[self.sent['content']:c_safe]))
            self.sent['content'] = c_safe
        return out

    def result(self):
        reasoning, content, tool, _ = self._parts()
        for s in self.stops:
            if s in content:
                content = content[:content.index(s)]
        if tool:
            content = content.rstrip('\n')
        return reasoning.strip(), content, tool


def make_handler(engine, model_name, default_thinking=True, default_max_tokens=4096):
    lock = threading.Lock()
    totals = {'requests': 0, 'prompt_tokens': 0, 'reused_tokens': 0, 'completion_tokens': 0, 'prefill_s': 0.0,
              'decode_s': 0.0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse(self, obj):
            self.wfile.write(b'data: ' + json.dumps(obj).encode() + b'\n\n')
            self.wfile.flush()

        def do_GET(self):
            path = self.path.split('?')[0].rstrip('/')
            if path in ('/health', '/v1/health'):
                return self._json(200, {'status': 'ok'})
            if path in ('/v1/models', '/models'):
                return self._json(200, {'object': 'list', 'data': [{
                    'id': model_name, 'object': 'model', 'owned_by': 'prism-ml', 'created': 0,
                    'context_length': engine.n_ctx}]})
            if path == '/metrics':
                m = dict(totals, last=engine.stats, hbm=engine.memory(), n_ctx=engine.n_ctx,
                         context_tokens=len(engine.history), checkpoints=[sn['pos'] for sn in engine.snapshots])
                if totals['decode_s']:
                    m['decode_tokens_per_second'] = totals['completion_tokens'] / totals['decode_s']
                if totals['prefill_s']:
                    m['prefill_tokens_per_second'] = totals['prompt_tokens'] / totals['prefill_s']
                return self._json(200, m)
            self._json(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            t_req = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            except Exception:
                return self._json(400, {'error': {'message': 'invalid JSON'}})
            path = self.path.split('?')[0].rstrip('/')
            chat = path in ('/v1/chat/completions', '/chat/completions')
            if not chat and path not in ('/v1/completions', '/completions'):
                return self._json(404, {'error': {'message': 'not found'}})
            tools = body.get('tools') if chat and body.get('tool_choice') != 'none' else None
            thinking = False
            try:
                if chat:
                    kw = {'tools': tools} if tools else {}
                    thinking, effort = thinking_options(body, default_thinking)
                    totals['last_thinking'] = {'enabled': thinking, 'effort': effort}
                    if effort:
                        kw['reasoning_effort'] = effort
                    for k_ in ('preserve_thinking',):
                        if k_ in (body.get('chat_template_kwargs') or {}):
                            kw[k_] = body['chat_template_kwargs'][k_]
                    prompt = engine.tokenizer.apply_chat_template(
                        normalize_messages(body['messages']), add_generation_prompt=True,
                        enable_thinking=bool(thinking), **kw)
                else:
                    prompt = body['prompt'] if isinstance(body['prompt'], str) else body['prompt'][0]
                ids, bounds = engine.tokenizer.encode_chat(prompt)
                # checkpoint where the conversation proper starts (after system prompt + tool schemas) and at
                # the last message: new sessions and edited last turns then resume without a full prefill
                first = next((b for b in bounds if b > 0), 0) if chat else 0
                snap_at = [p for p in {first, bounds[-2] if len(bounds) > 2 else 0} if p > 0]
                host_ms = 1e3 * (time.perf_counter() - t_req)
                if len(ids) >= engine.n_ctx:
                    raise ValueError('prompt has %d tokens; n_ctx is %d' % (len(ids), engine.n_ctx))
            except Exception as e:
                return self._json(400, {'error': {'message': '%s: %s' % (type(e).__name__, e)}})
            samp = _sampling(body, default_max_tokens)
            samp['max_new_tokens'] = max(1, min(samp['max_new_tokens'], engine.n_ctx - len(ids)))
            stops = body.get('stop') or []
            stops = [stops] if isinstance(stops, str) else list(stops)
            rid = ('chatcmpl-' if chat else 'cmpl-') + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(body.get('stream'))
            base = {'id': rid, 'object': 'chat.completion.chunk' if chat else 'text_completion',
                    'created': created, 'model': model_name}

            def emit(kind, text):
                if chat:
                    delta = {'reasoning_content': text} if kind == 'reasoning' else {'content': text}
                    self._sse(dict(base, choices=[{'index': 0, 'delta': delta, 'finish_reason': None}]))
                elif kind == 'content':
                    self._sse(dict(base, choices=[{'index': 0, 'text': text, 'finish_reason': None}]))

            with lock:
                if stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    if chat:
                        self._sse(dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                                                       'finish_reason': None}]))
                dec = StreamDecoder(engine.tokenizer, skip_special=False)
                split = OutputSplitter(bool(thinking), bool(tools), stops)
                gen = engine.generate_ids(ids, snap_at=snap_at, **samp)
                try:
                    for tok in gen:
                        for kind, text in split.push(dec.push(tok)):
                            if stream:
                                emit(kind, text)
                        if split.stopped:
                            break
                    gen.close()
                    for kind, text in split.push('', final=True):
                        if stream:
                            emit(kind, text)
                except (BrokenPipeError, ConnectionResetError):
                    gen.close()
                    return
                except Exception as e:          # device failure mid-request: donated state may be gone
                    gen.close()
                    engine.hard_reset()
                    totals['errors'] = totals.get('errors', 0) + 1
                    err = {'error': {'message': '%s: %s' % (type(e).__name__, str(e)[:500]), 'type': 'server_error'}}
                    try:
                        if stream:
                            self._sse(err)
                        else:
                            self._json(500, err)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                st = dict(engine.stats)
                reasoning, content, tool_text = split.result()
                calls = parse_tool_calls(tool_text, tools) if tool_text else []
                finish = 'tool_calls' if calls else ('stop' if split.stopped else st.get('finish_reason', 'stop'))
                usage = {'prompt_tokens': len(ids), 'completion_tokens': st.get('completion_tokens', 0),
                         'total_tokens': len(ids) + st.get('completion_tokens', 0),
                         'prompt_tokens_details': {'cached_tokens': st.get('reused_tokens', 0)}}
                timings = {'host_prepare_ms': host_ms, 'prompt_n': st.get('prompt_tokens', 0), 'cache_n': st.get('reused_tokens', 0),
                           'prompt_ms': 1e3 * st.get('prefill_s', 0.0), 'predicted_n': st.get('completion_tokens', 0),
                           'predicted_ms': 1e3 * st.get('decode_s', 0.0)}
                if st.get('decode_s'):
                    timings['predicted_per_second'] = st['completion_tokens'] / st['decode_s']
                totals['requests'] += 1
                for k in ('prompt_tokens', 'reused_tokens', 'completion_tokens', 'prefill_s', 'decode_s'):
                    totals[k] += st.get(k, 0)
                if stream:
                    try:
                        if calls:
                            self._sse(dict(base, choices=[{'index': 0, 'finish_reason': None, 'delta': {
                                'tool_calls': [dict(c, index=i) for i, c in enumerate(calls)]}}]))
                        last = dict(base, choices=[dict({'index': 0, 'finish_reason': finish},
                                                        **({'delta': {}} if chat else {'text': ''}))])
                        last['usage'] = usage
                        last['timings'] = timings
                        self._sse(last)
                        self.wfile.write(b'data: [DONE]\n\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
            if chat:
                msg = {'role': 'assistant', 'content': content}
                if reasoning:
                    msg['reasoning_content'] = reasoning
                if calls:
                    msg['tool_calls'] = calls
                choice = {'index': 0, 'message': msg, 'finish_reason': finish}
            else:
                choice = {'index': 0, 'text': content, 'finish_reason': finish}
            self._json(200, {'id': rid, 'object': 'chat.completion' if chat else 'text_completion',
                             'created': created, 'model': model_name, 'choices': [choice], 'usage': usage,
                             'timings': timings})

    return Handler


def serve(engine, host='127.0.0.1', port=8000, model_name=MODEL_NAME, **kw):
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, model_name, **kw))
    httpd.daemon_threads = True
    return httpd
