#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

MANIFEST="${PROJECT_ROOT}/perf_pp_vpp/k8s/chenny_dist.yaml"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest: ${MANIFEST}" >&2
  exit 2
fi

${KUBECTL_BIN} apply -f "${MANIFEST}"
${KUBECTL_BIN} rollout status -n "${K8S_NAMESPACE}" statefulset/chenny-dist --timeout=15m
${KUBECTL_BIN} get pods -n "${K8S_NAMESPACE}" -l app=chenny-dist -o wide
