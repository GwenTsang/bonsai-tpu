#!/bin/bash
# Usage: scripts/run.sh <script.py> [logfile]  -- sync package, run script in the persistent TPU kernel.
set -e
cd "$(dirname "$0")/.."
scripts/sync.sh
t=$(mktemp --suffix=.py)
cat > "$t" <<'PY'
import sys,tarfile,os
with tarfile.open('/content/bonsai-pkg.tar.gz') as f:f.extractall('/content',filter='data')
if '/content' not in sys.path:sys.path.insert(0,'/content')
for _m in [m for m in sys.modules if m.split('.')[0] in ('bonsai_tpu','bench')]:del sys.modules[_m]
os.makedirs('/content/bonsai-results',exist_ok=True)
PY
cat "$1" >> "$t"
log=${2:-/tmp/bonsai-run.log}
timeout ${RUN_TIMEOUT:-1800} colab exec -s bonsai-tpu --timeout 0 -f "$t" > "$log" 2>&1 || echo "EXIT $?" >> "$log"
rm -f "$t"
tail -n ${RUN_TAIL:-60} "$log"
