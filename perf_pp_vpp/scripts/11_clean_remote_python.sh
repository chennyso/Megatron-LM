#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
source perf_pp_vpp/configs/env.sh

KUBECTL_BIN="${KUBECTL_BIN:-bash /usr/local/bin/kubectl}"
K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
NODE0_POD="${NODE0_POD:-chenny-dist-0}"
NODE1_POD="${NODE1_POD:-chenny-dist-1}"

clean_pod() {
  local pod="$1"
  bash /usr/local/bin/kubectl exec -n "${K8S_NAMESPACE}" "${pod}" -- bash -lc "python - <<'PY'
import os, signal, subprocess
out = subprocess.check_output(['ps', '-eo', 'pid=,stat=,args='], text=True)
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_s, stat, args = line.split(None, 2)
    if 'Z' in stat:
        continue
    if args.startswith('/usr/bin/python3'):
        try:
            os.kill(int(pid_s), signal.SIGKILL)
            print(f'killed {pid_s}')
        except ProcessLookupError:
            pass
PY" >/dev/null
}

clean_pod "${NODE0_POD}"
clean_pod "${NODE1_POD}"

