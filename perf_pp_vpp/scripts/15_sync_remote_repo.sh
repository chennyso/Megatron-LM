#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

REF=""
PODS=("${NODE0_POD}" "${NODE1_POD}")
FORCE_CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      REF="$2"
      shift 2
      ;;
    --pod)
      PODS=("$2")
      shift 2
      ;;
    --pods)
      IFS=',' read -r -a PODS <<<"$2"
      shift 2
      ;;
    --force-clean)
      FORCE_CLEAN=1
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REF}" ]]; then
  echo "usage: $0 --ref <git-ref> [--pod <name> | --pods pod0,pod1] [--force-clean]" >&2
  exit 2
fi

for pod in "${PODS[@]}"; do
  echo "==> ${pod}: checking ${REMOTE_PROJECT_ROOT}"
  status="$(${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "cd \"${REMOTE_PROJECT_ROOT}\" && git status --porcelain" 2>/dev/null || true)"
  if [[ -n "${status}" && "${FORCE_CLEAN}" != "1" ]]; then
    echo "${status}" >&2
    echo "refusing to update dirty worktree on ${pod}; rerun with --force-clean if you want to discard local changes" >&2
    exit 1
  fi

  if [[ "${FORCE_CLEAN}" == "1" ]]; then
    ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc \
      "cd \"${REMOTE_PROJECT_ROOT}\" && git reset --hard HEAD && git clean -fd"
  fi

  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc \
    "cd \"${REMOTE_PROJECT_ROOT}\" && git fetch origin --tags --prune && (git checkout \"${REF}\" && git pull --ff-only origin \"${REF}\" || git checkout --detach \"${REF}\") && git rev-parse HEAD"
done
