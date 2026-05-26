#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

OUT_DIR="${OUTPUT_ROOT}/system_info"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"
: > "${SUMMARY}"

run_local() {
  local name="$1"
  shift
  {
    echo "### ${name}"
    echo "$ $*"
    "$@" 2>&1 || true
    echo
  } | tee "${OUT_DIR}/${name}.txt" >> "${SUMMARY}"
}

run_remote() {
  local pod="$1"
  local name="$2"
  shift 2
  {
    echo "### ${pod}:${name}"
    echo "$ $*"
    ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- bash -lc "$*" 2>&1 || true
    echo
  } | tee "${OUT_DIR}/${pod}_${name}.txt" >> "${SUMMARY}"
}

run_local date date
for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  run_remote "${pod}" hostname hostname
  run_remote "${pod}" uname "uname -a"
  run_remote "${pod}" lscpu "lscpu"
  run_remote "${pod}" free "free -h"
  run_remote "${pod}" lsblk "lsblk"
  run_remote "${pod}" nvidia_smi "nvidia-smi"
  run_remote "${pod}" nvidia_smi_q "nvidia-smi -q"
  run_remote "${pod}" nvidia_topo "nvidia-smi topo -m"
  run_remote "${pod}" nvidia_nvlink "nvidia-smi nvlink --status"
  run_remote "${pod}" ibstat "command -v ibstat >/dev/null && ibstat || echo ibstat_not_found"
  run_remote "${pod}" ibv_devinfo "command -v ibv_devinfo >/dev/null && ibv_devinfo || echo ibv_devinfo_not_found"
  run_remote "${pod}" ip_addr "command -v ip >/dev/null && ip addr || echo ip_not_found"
  run_remote "${pod}" env "env | sort"
  run_remote "${pod}" torch_version "${PYTHON_BIN} -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())'"
  run_remote "${pod}" pip_freeze "${PYTHON_BIN} -m pip freeze"
  run_remote "${pod}" nvcc_version "command -v nvcc >/dev/null && nvcc --version || echo nvcc_not_found"
  run_remote "${pod}" nsys_version "command -v ${NSYS_BIN} >/dev/null && ${NSYS_BIN} --version || echo nsys_not_found"
  run_remote "${pod}" megatron_git "cd ${REMOTE_PROJECT_ROOT} && git rev-parse HEAD && git status --short"
done
