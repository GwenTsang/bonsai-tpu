"""Prepare pi locally, then start Bonsai on an authenticated, existing Colab TPU session."""
import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PI_PACKAGE = '@earendil-works/pi-coding-agent@1.5.2'


def ensure_pi():
    found = shutil.which('pi')
    if found:
        return Path(found)
    prefix = Path.home() / '.local/share/bonsai_tpu/pi'
    for candidate in (prefix / 'bin/pi', Path.home() / '.local/bin/pi', Path.home() / '.bun/bin/pi'):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    if not shutil.which('npm') or not shutil.which('node'):
        raise RuntimeError('Install Node.js >=22.19 (https://nodejs.org/) or pi (https://pi.dev/) first, then rerun.')
    version = subprocess.check_output(['node', '--version'], text=True).strip().lstrip('v')
    if tuple(int(x) for x in version.split('.')[:3]) < (22, 19, 0):
        raise RuntimeError('pi requires Node.js >=22.19; update Node.js, then rerun.')
    print('Installing pi in %s (no sudo or global npm changes)' % prefix, flush=True)
    subprocess.run(['npm', 'install', '--prefix', str(prefix), '--global', '--ignore-scripts', PI_PACKAGE], check=True)
    return prefix / 'bin/pi'


def configure_pi(agent_dir, port, ctx):
    path = agent_dir / 'models.json'
    original = path.read_text() if path.exists() else None
    data = json.loads(original) if original is not None else {}
    if not isinstance(data, dict) or not isinstance(data.get('providers', {}), dict):
        raise ValueError('models.json must contain a JSON object with a providers object')
    before = copy.deepcopy(data)
    template = json.loads((ROOT / 'pi-agent/models.json').read_text())['providers']['bonsai-tpu']
    provider = data.setdefault('providers', {}).setdefault('bonsai-tpu', {})
    if not isinstance(provider, dict) or not isinstance(provider.get('models', []), list):
        raise ValueError('Existing bonsai-tpu provider/models have an invalid format')
    for key in ('name', 'api', 'baseUrl'):
        provider[key] = template[key]
    provider['baseUrl'] = 'http://127.0.0.1:%d/v1' % port
    provider.setdefault('apiKey', 'local')
    models = provider.setdefault('models', [])
    entry = next((m for m in models if isinstance(m, dict) and m.get('id') == 'ternary-bonsai-2-27b'), None)
    if entry is None:
        entry = {}
        models.append(entry)
    def defaults(target, values):
        for key, value in values.items():
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(value, dict) and isinstance(target[key], dict):
                defaults(target[key], value)
    defaults(entry, template['models'][0])
    entry['contextWindow'] = ctx
    entry['maxTokens'] = min(entry['maxTokens'], ctx // 2)
    if before == data:
        return path, False
    agent_dir.mkdir(parents=True, exist_ok=True)
    if original is not None:
        backup = path.with_name('models.json.bonsai-backup-' + str(time.time_ns()))
        backup.write_text(original)
        backup.chmod(0o600)
    fd, temporary = tempfile.mkstemp(prefix='.models-', dir=agent_dir)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(data, out, indent=2)
            out.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path, True


def colab_code(colab, session, source, timeout=60):
    with tempfile.TemporaryDirectory(prefix='bonsai-control-') as tmp:
        path = Path(tmp) / 'control.py'
        path.write_text(source)
        result = subprocess.run([colab, 'exec', '-s', session, '--timeout', str(timeout), '-f', str(path)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout + 30)
    if result.returncode:
        raise RuntimeError('Colab execution failed. Check `colab sessions` and authentication.\n' + result.stderr[-1000:])
    for line in result.stdout.splitlines():
        if line.startswith('BONSAI_STATUS:'):
            return json.loads(line[len('BONSAI_STATUS:'):])
    raise RuntimeError('No setup status received from Colab; rerun to inspect the existing job.')


def launch_code(ctx, source_hash):
    return f'''
import builtins, json, sys, threading, traceback, tarfile, time, urllib.request, os
def _bonsai_setup_launch():
    global _BONSAI_SETUP, _BONSAI_LOG, print
    if globals().get('_THREAD') is not None and _THREAD.is_alive():
        raise RuntimeError('A development run is active; wait for it to finish.')
    if os.path.exists('/content/.bonsai-hf-token'):
        os.chmod('/content/.bonsai-hf-token', 0o600)
    if globals().get('_BONSAI_SETUP', {{}}).get('thread') is not None and _BONSAI_SETUP['thread'].is_alive():
        if _BONSAI_SETUP['ctx'] != {ctx} or _BONSAI_SETUP.get('source_hash') != {source_hash!r}:
            raise RuntimeError('A different setup job is active; wait for it to finish.')
        return
    if globals().get('_BONSAI_SETUP', {{}}).get('phase') == 'ready' and _BONSAI_SETUP.get('ctx') == {ctx} and _BONSAI_SETUP.get('source_hash') == {source_hash!r}:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
    _BONSAI_SETUP = {{'phase': 'starting', 'ctx': {ctx}, 'started': time.time(), 'source_hash': {source_hash!r}}}
    _BONSAI_LOG = open('/content/bonsai-setup.log', 'w')
    def print(*a, **kw):
        kw.pop('file', None); kw.pop('flush', None)
        builtins.print(*a, file=_BONSAI_LOG, flush=True, **kw)
    with tarfile.open('/content/bonsai-setup.tar.gz') as archive:
        archive.extractall('/content', filter='data')
    for name in [name for name in list(sys.modules) if name.split('.')[0] == 'bonsai_tpu']:
        del sys.modules[name]
    if '/content' not in sys.path:
        sys.path.insert(0, '/content')
    def run():
        try:
            exec(compile(open('/content/runs/bootstrap_colab.py').read(), 'bootstrap_colab.py', 'exec'), globals())
        except BaseException as exc:
            if _BONSAI_SETUP['phase'] != 'restart_required':
                _BONSAI_SETUP['phase'] = 'failed'
            _BONSAI_SETUP['error'] = type(exc).__name__
            traceback.print_exc(file=_BONSAI_LOG)
    _BONSAI_SETUP['thread'] = threading.Thread(target=run, daemon=True)
    _BONSAI_SETUP['thread'].start()
_bonsai_setup_launch()
sys.stdout.write('BONSAI_STATUS:' + json.dumps({{k:v for k,v in _BONSAI_SETUP.copy().items() if k != 'thread'}}) + '\\n')
'''


STATUS_CODE = """
import json,sys
sys.stdout.write('BONSAI_STATUS:' + json.dumps({k:v for k,v in globals().get('_BONSAI_SETUP', {'phase':'missing'}).copy().items() if k != 'thread'}) + '\\n')
"""


def start_remote(colab, session, ctx, token):
    status = subprocess.run([colab, 'status', '-s', session], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    if status.returncode or 'V5E1' not in status.stdout.upper():
        raise RuntimeError('Authenticate with `colab sessions`, then attach/create a v5e1 session named ' + session)
    with tempfile.TemporaryDirectory(prefix='bonsai-setup-') as tmp:
        archive = Path(tmp) / 'source.tar.gz'
        digest = hashlib.sha256()
        with tarfile.open(archive, 'w:gz') as tar:
            for name in ('bonsai_tpu', 'runs'):
                for path in sorted((ROOT / name).rglob('*.py')):
                    relative = str(path.relative_to(ROOT))
                    if ((name == 'bonsai_tpu' and path.name not in ('setup_colab.py', 'colab_bridge.py'))
                            or relative in ('runs/bootstrap_colab.py', 'runs/load.py')):
                        digest.update(relative.encode())
                        digest.update(path.read_bytes())
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
        subprocess.run([colab, 'upload', '-s', session, str(archive), '/content/bonsai-setup.tar.gz'],
                       stdin=subprocess.DEVNULL, check=True)
        if token:
            secret = Path(tmp) / 'hf-token'
            secret.write_text(token)
            secret.chmod(0o600)
            subprocess.run([colab, 'upload', '-s', session, str(secret), '/content/.bonsai-hf-token'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
    state = colab_code(colab, session, launch_code(ctx, digest.hexdigest()))
    deadline = time.monotonic() + 1800
    last_phase = None
    while time.monotonic() < deadline:
        phase = state['phase']
        if phase != last_phase:
            print('Colab: ' + phase, flush=True)
            last_phase = phase
        if phase == 'ready':
            colab_code(colab, session, """
import os,json,sys
if os.path.exists('/content/.bonsai-hf-token'): os.unlink('/content/.bonsai-hf-token')
sys.stdout.write('BONSAI_STATUS:{"phase":"ready"}\\n')
""")
            return state
        if phase == 'restart_required':
            raise RuntimeError('Runtime packages updated; run `colab restart-kernel -s %s`, then rerun setup.' % session)
        if phase in ('failed', 'missing'):
            raise RuntimeError('Remote setup failed (%s). Inspect /content/bonsai-setup.log with Colab.' % state.get('error', phase))
        time.sleep(3)
        state = colab_code(colab, session, STATUS_CODE)
    raise RuntimeError('Setup observation timed out; the remote job may still be running. Rerun to inspect it.')


def check_bridge_port(session, port):
    url = 'http://127.0.0.1:%d/health' % port
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            health = json.load(response)
        if health == dict(transport='colab-exec', session=session):
            return True
        raise RuntimeError('Port %d is already serving another endpoint; choose --port.' % port)
    except OSError:
        pass
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
    except OSError as exc:
        raise RuntimeError('Port %d is occupied; choose another --port.' % port) from exc
    return False


def start_bridge(colab, session, port):
    if check_bridge_port(session, port):
        return
    url = 'http://127.0.0.1:%d/health' % port
    cache = Path.home() / '.cache/bonsai_tpu'
    cache.mkdir(parents=True, exist_ok=True)
    log_path = cache / ('bridge-%d.log' % port)
    with log_path.open('a') as log:
        env = dict(os.environ)
        env.pop('HF_TOKEN', None)
        child = subprocess.Popen([sys.executable, '-m', 'bonsai_tpu.colab_bridge', '--session', session,
                                  '--port', str(port), '--colab', colab], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(30):
        if child.poll() is not None:
            raise RuntimeError('Local bridge exited; check ' + str(log_path))
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if json.load(response) == dict(transport='colab-exec', session=session):
                    (cache / ('bridge-%d.pid' % port)).write_text(str(child.pid))
                    return
        except OSError:
            time.sleep(0.2)
    child.terminate()
    raise RuntimeError('Local bridge did not become ready; check ' + str(log_path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf-token', help='HF read token (alternatively HF_TOKEN or --hf-token-file)')
    p.add_argument('--hf-token-file', type=Path)
    p.add_argument('--session', default='bonsai-tpu')
    p.add_argument('--ctx', type=int, default=65536)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--pi-agent-dir', type=Path, default=Path(os.environ.get('PI_CODING_AGENT_DIR', '~/.pi/agent')).expanduser())
    p.add_argument('--prepare-only', action='store_true', help='install/configure pi before starting the TPU clock')
    a = p.parse_args(argv)
    a.pi_agent_dir = a.pi_agent_dir.expanduser().resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.session) or not 1024 <= a.ctx <= 98304 or not 1 <= a.port <= 65535:
        p.error('use a simple session name, --ctx 1024..98304 and --port 1..65535')
    try:
        pi = ensure_pi()
        path, changed = configure_pi(a.pi_agent_dir, a.port, a.ctx)
        print(('Updated ' if changed else 'Already configured: ') + str(path), flush=True)
        if not a.prepare_only:
            colab = shutil.which('colab')
            if not colab:
                raise RuntimeError('Install/authenticate the Colab CLI first: `uv tool install google-colab-cli`; `colab sessions`.')
            check_bridge_port(a.session, a.port)
            token = a.hf_token or (a.hf_token_file.read_text().strip() if a.hf_token_file else os.environ.get('HF_TOKEN'))
            start_remote(colab, a.session, a.ctx, token)
            start_bridge(colab, a.session, a.port)
        command = [str(pi), '--provider', 'bonsai-tpu', '--model', 'ternary-bonsai-2-27b']
        print('Run: PI_CODING_AGENT_DIR=%s %s' % (shlex.quote(str(a.pi_agent_dir.expanduser())), shlex.join(command)))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('Setup failed: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
