#!/bin/bash
# Run a shell command on the attached Colab runtime (inside the existing kernel; no restart).
t=$(mktemp --suffix=.py)
python3 - "$@" > "$t" <<'PY'
import sys
print("import subprocess\nr=subprocess.run(%r,shell=True,capture_output=True,text=True)\nprint(r.stdout[-6000:],r.stderr[-3000:])"%sys.argv[1])
PY
timeout ${RSH_TIMEOUT:-300} colab exec -s bonsai-tpu -f "$t" 2>&1 | tail -n ${RSH_TAIL:-60}; rm -f "$t"
