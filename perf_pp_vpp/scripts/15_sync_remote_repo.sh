#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

REF=""
PODS=("${NODE0_POD}" "${NODE1_POD}")
FORCE_CLEAN=0
SSL_VERIFY=1

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
    --no-ssl-verify)
      SSL_VERIFY=0
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REF}" ]]; then
  echo "usage: $0 --ref <git-ref> [--pod <name> | --pods pod0,pod1] [--force-clean] [--no-ssl-verify]" >&2
  exit 2
fi

git_fetch_prefix="git"
if [[ "${SSL_VERIFY}" != "1" ]]; then
  git_fetch_prefix="git -c http.sslVerify=false"
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
    "cd \"${REMOTE_PROJECT_ROOT}\" && ${git_fetch_prefix} fetch origin --tags --prune && if git show-ref --verify --quiet \"refs/remotes/origin/${REF}\"; then git checkout -B \"${REF}\" \"origin/${REF}\" && ${git_fetch_prefix} pull --ff-only origin \"${REF}\"; else git checkout --detach \"${REF}\"; fi && git rev-parse HEAD"
done
