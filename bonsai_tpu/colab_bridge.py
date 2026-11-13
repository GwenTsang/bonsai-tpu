"""Local OpenAI endpoint carried by authenticated Colab CLI execution (no SSH required)."""
import argparse
import base64
import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HEADER = 'BONSAI_HEADER:'
BODY = 'BONSAI_BODY:'
ERROR = 'BONSAI_ERROR:'


def request_code(method, path, body):
    """Encode data as literals; user prompts never become executable Python."""
    payload = base64.b64encode(body).decode()
    return f'''
import base64, json, sys, urllib.request, urllib.error
def _bonsai_forward():
    try:
        req = urllib.request.Request('http://127.0.0.1:8000' + {path!r},
            data=base64.b64decode({payload!r}) if {method!r} == 'POST' else None,
            headers={{'Content-Type': 'application/json'}}, method={method!r})
        try:
            response = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            meta = json.dumps({{'status': response.status, 'content_type': response.headers.get('Content-Type', 'application/json')}}).encode()
            sys.stdout.write({HEADER!r} + base64.b64encode(meta).decode() + '\\n'); sys.stdout.flush()
            if 'text/event-stream' in response.headers.get('Content-Type', ''):
                for chunk in response:
                    sys.stdout.write({BODY!r} + base64.b64encode(chunk).decode() + '\\n'); sys.stdout.flush()
            else:
                sys.stdout.write({BODY!r} + base64.b64encode(response.read()).decode() + '\\n'); sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write({ERROR!r} + type(exc).__name__ + '\\n'); sys.stdout.flush()
_bonsai_forward()
'''


def bridge_server(session, port, colab='colab'):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *args):
            pass                         # prompts and responses are not written to the bridge log

        def do_GET(self):
            if self.path == '/health':
                data = json.dumps(dict(transport='colab-exec', session=session)).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            if self.path not in ('/v1/models', '/v1/chat/completions', '/v1/completions', '/metrics'):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
            except ValueError:
                self.send_error(400)
                return
            if size < 0 or size > 32 << 20:
                self.send_error(413)
                return
            body = self.rfile.read(size)
            sent = False
            proc = None
            timer = None
            # One kernel cell/request at a time; the remote engine itself is single-user too.
            with lock, tempfile.TemporaryDirectory(prefix='bonsai-request-') as tmp:
                source = Path(tmp) / 'request.py'
                source.write_text(request_code(self.command, self.path, body))
                source.chmod(0o600)
                try:
                    proc = subprocess.Popen([colab, 'exec', '-s', session, '--timeout', '0', '-f', str(source)],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True)
                    timer = threading.Timer(1800, proc.terminate)
                    timer.daemon = True
                    timer.start()
                    for line in proc.stdout:
                        if line.startswith(HEADER):
                            meta = json.loads(base64.b64decode(line[len(HEADER):]))
                            self.send_response(meta['status'])
                            self.send_header('Content-Type', meta['content_type'])
                            self.send_header('Cache-Control', 'no-cache')
                            self.end_headers()
                            sent = True
                        elif line.startswith(BODY):
                            self.wfile.write(base64.b64decode(line[len(BODY):]))
                            self.wfile.flush()
                        elif line.startswith(ERROR):
                            raise ConnectionError('Remote inference request failed')
                    proc.wait(timeout=10)
                    if not sent:
                        raise ConnectionError('No response from Colab; check authentication and session availability')
                except (OSError, ValueError, subprocess.SubprocessError):
                    if not sent:
                        self.send_error(502, 'Colab request failed; check the session and Colab CLI authentication')
                finally:
                    if timer is not None:
                        timer.cancel()
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', default='bonsai-tpu')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--colab', default='colab')
    a = p.parse_args()
    bridge_server(a.session, a.port, a.colab).serve_forever()


if __name__ == '__main__':
    main()
