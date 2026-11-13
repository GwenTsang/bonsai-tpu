"""Setup/configuration and transport contracts without credentials or cloud resources."""
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from bonsai_tpu import setup_colab as setup
from bonsai_tpu.colab_bridge import bridge_server, request_code


def test_config_merge_preserves_other_models_credentials_and_is_idempotent(tmp_path):
    original = {'extra': True, 'providers': {
        'existing': {'apiKey': 'keep-private', 'models': [{'id': 'other'}]},
        'bonsai-tpu': {'custom': 42, 'models': [{'id': 'keep-this'}, {
            'id': 'ternary-bonsai-2-27b', 'custom': 3, 'samplingParams': {'temperature': 0.2}}]}}}
    path = tmp_path / 'models.json'
    path.write_text(json.dumps(original))
    _, changed = setup.configure_pi(tmp_path, 8123, 32768)
    assert changed
    result = json.loads(path.read_text())
    assert result['providers']['existing'] == original['providers']['existing']
    p = result['providers']['bonsai-tpu']
    assert p['custom'] == 42 and p['models'][0] == {'id': 'keep-this'}
    assert p['models'][1]['custom'] == 3 and p['models'][1]['contextWindow'] == 32768
    assert p['models'][1]['samplingParams']['temperature'] == 0.2
    assert p['baseUrl'] == 'http://127.0.0.1:8123/v1'
    backups = list(tmp_path.glob('models.json.bonsai-backup-*'))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == original
    assert backups[0].stat().st_mode & 0o777 == 0o600
    before = path.stat().st_mtime_ns
    assert setup.configure_pi(tmp_path, 8123, 32768)[1] is False
    assert path.stat().st_mtime_ns == before
    assert len(list(tmp_path.glob('models.json.bonsai-backup-*'))) == 1


@pytest.mark.parametrize('content', ['invalid json', '[]', '{"providers": []}'])
def test_invalid_config_is_never_overwritten(tmp_path, content):
    path = tmp_path / 'models.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        setup.configure_pi(tmp_path, 8000, 65536)
    assert path.read_text() == content


def test_existing_pi_is_reused_without_install(monkeypatch):
    monkeypatch.setattr(setup.shutil, 'which', lambda name: '/existing/pi' if name == 'pi' else None)
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **kw: pytest.fail('unexpected install'))
    assert setup.ensure_pi() == Path('/existing/pi')


def test_missing_pi_installs_to_private_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(setup.shutil, 'which', lambda name: None if name == 'pi' else '/usr/bin/' + name)
    monkeypatch.setattr(setup.subprocess, 'check_output', lambda *a, **kw: 'v22.23.2\n')
    calls = []
    monkeypatch.setattr(setup.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    path = setup.ensure_pi()
    assert str(path).startswith(str(tmp_path))
    assert calls[0][-1] == setup.PI_PACKAGE and '--ignore-scripts' in calls[0]


def test_busy_launch_does_not_mutate_print_or_state():
    marker = object()
    namespace = {'_THREAD': SimpleNamespace(is_alive=lambda: True), 'print': marker}
    with pytest.raises(RuntimeError, match='active'):
        exec(setup.launch_code(65536, 'test'), namespace)
    assert namespace['print'] is marker and '_BONSAI_SETUP' not in namespace


def test_token_staged_as_private_file_not_code_or_subprocess_argument(monkeypatch, tmp_path):
    token = 'hf_secret_test_only'
    seen = []

    def run(argv, **kwargs):
        assert all(token not in value for value in argv)
        if 'upload' in argv and argv[-1] == '/content/.bonsai-hf-token':
            path = Path(argv[-2])
            assert path.read_text() == token and path.stat().st_mode & 0o777 == 0o600
            seen.append(path)
        return SimpleNamespace(returncode=0, stdout='Hardware: V5E1', stderr='')

    monkeypatch.setattr(setup.subprocess, 'run', run)
    monkeypatch.setattr(setup, 'colab_code', lambda c, s, source: {'phase': 'ready'})
    setup.start_remote('colab', 'test', 65536, token)
    assert len(seen) == 1 and not seen[0].exists()


def test_runtime_upgrade_requests_restart_before_loading_model(monkeypatch):
    import importlib.metadata
    import importlib.util
    calls = []
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.7.0')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None if name == 'libtpu' else object())
    monkeypatch.setattr(setup.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace())
    state = {'phase': 'starting'}
    namespace = {'_BONSAI_SETUP': state, '_BONSAI_LOG': io.StringIO()}
    source = (setup.ROOT / 'runs/bootstrap_colab.py').read_text().split("_BONSAI_SETUP['phase'] = 'downloading/loading model'")[0]
    with pytest.raises(RuntimeError, match='Restart'):
        exec(source, namespace)
    assert state['phase'] == 'restart_required'
    assert 'jax[tpu]==0.11.2' in calls[0]


def test_bridge_can_restart_on_a_recently_closed_http_port():
    class Health(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    with urllib.request.urlopen('http://127.0.0.1:%d/health' % port) as response:
        response.read()
    server.shutdown()
    server.server_close()
    assert setup.check_bridge_port('test', port) is False


def test_bridge_round_trip_stream_errors_and_prompt_is_data(tmp_path):
    captured = []

    class Origin(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured.append(data)
            if data.get('fail'):
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error":"bad request"}')
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"text":"hello"}\n\ndata: [DONE]\n\n')

    origin = ThreadingHTTPServer(('127.0.0.1', 0), Origin)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    fake = tmp_path / 'colab'
    fake.write_text(f'#!{sys.executable}\nimport sys\nfrom pathlib import Path\n'
                    f"source=Path(sys.argv[-1]).read_text().replace('127.0.0.1:8000', '127.0.0.1:{origin.server_port}')\n"
                    'exec(compile(source,"remote-test","exec"))\n')
    fake.chmod(0o700)
    bridge = bridge_server('test', 0, str(fake))
    threading.Thread(target=bridge.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d/v1/chat/completions' % bridge.server_port
    prompt = "Unicode é 🌳; '); __import__('os').abort(); #"
    try:
        body = {'messages': [{'role': 'user', 'content': prompt}], 'stream': True}
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode()), timeout=10) as response:
            assert response.headers['Content-Type'] == 'text/event-stream'
            assert response.read() == b'data: {"text":"hello"}\n\ndata: [DONE]\n\n'
        assert captured[-1] == body
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, b'{"fail":true}'), timeout=10)
        assert error.value.code == 400 and b'bad request' in error.value.read()
    finally:
        bridge.shutdown()
        bridge.server_close()
        origin.shutdown()
        origin.server_close()
