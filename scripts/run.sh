#!/bin/bash
# Usage: scripts/run.sh <script.py> [local-log]
# Syncs the package, then runs the script in a background thread of the persistent TPU kernel, logging to a
# remote file that is polled with short execs (long-lived exec streams drop: "Connection was lost").
set -e
cd "$(dirname "$0")/.."
scripts/sync.sh
name=$(basename "$1" .py)-$(date +%s)
log=${2:-/tmp/bonsai-run.log}
t=$(mktemp --suffix=.py); p=$(mktemp --suffix=.py)
cat > "$t" <<PY
import sys,tarfile,os,threading,traceback,builtins
with tarfile.open('/content/bonsai-pkg.tar.gz') as f:f.extractall('/content',filter='data')
if '/content' not in sys.path:sys.path.insert(0,'/content')
for _m in [m for m in sys.modules if m.split('.')[0] in ('bonsai_tpu','bench')]:del sys.modules[_m]
os.makedirs('/content/bonsai-results',exist_ok=True);os.makedirs('/content/logs',exist_ok=True)
for _kv in '''${RUN_ENV:-}'''.split():os.environ[_kv.split('=',1)[0]]=_kv.split('=',1)[1]
_LOG=open('/content/logs/$name.log','w')
def print(*a,**k):
    k.pop('file',None);k.pop('flush',None);builtins.print(*a,file=_LOG,flush=True,**k)
_SRC=open('/content/$1').read()
def _run():
    try:exec(compile(_SRC,'$1','exec'),globals())
    except BaseException:traceback.print_exc(file=_LOG)
    builtins.print('RUN_DONE',file=_LOG,flush=True)
if globals().get('_THREAD') is not None and _THREAD.is_alive():
    sys.stdout.write('BUSY: previous run still active\n')
else:
    _THREAD=threading.Thread(target=_run,daemon=True);_THREAD.start();sys.stdout.write('STARTED\n')
PY
cat > "$p" <<PY
import sys
sys.stdout.write(open('/content/logs/$name.log').read())
PY
: > "$log"
started=""
for attempt in 1 2 3; do
  started=$(timeout 120 colab exec -s bonsai-tpu --timeout 60 -f "$t" 2>&1 | grep -E 'STARTED|BUSY' | head -1 || true)
  [ -n "$started" ] && break
  sleep 5
done
echo "${started:-START FAILED}"
[ "$started" = "STARTED" ] || { rm -f "$t" "$p"; exit 1; }
end=$((SECONDS+${RUN_TIMEOUT:-1800}))
while [ $SECONDS -lt $end ]; do
  sleep ${RUN_POLL:-6}
  timeout 120 colab exec -s bonsai-tpu --timeout 60 -f "$p" > "$log.tmp" 2>/dev/null && mv "$log.tmp" "$log"
  grep -q '^RUN_DONE' "$log" && break
done
rm -f "$t" "$p"
sed -i "s/\x1b\[[0-9;]*m//g" "$log"; tail -n ${RUN_TAIL:-60} "$log"
