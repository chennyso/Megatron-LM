#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

OUT_DIR="${OUTPUT_ROOT}/nccl_tests"
mkdir -p "${OUT_DIR}"

if [[ -z "${NCCL_TESTS_ROOT}" ]]; then
  echo "NCCL_TESTS_ROOT is required" >&2
  exit 2
fi

SIZES=(8M 16M 32M 64M 128M 256M 512M)
FIRST="${SIZES[0]}"
LAST="${SIZES[${#SIZES[@]}-1]}"

run_test() {
  local bin_name="$2"
  local nnodes="$3"
  local outfile="$4"
  if ! ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${NODE0_POD}" -- bash -lc "test -x ${NCCL_TESTS_ROOT}/${bin_name}"; then
    echo "missing ${bin_name}" > "${outfile}"
    return 0
  fi
  local remote_cmd="
    cd ${REMOTE_PROJECT_ROOT} && \
    CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
    NCCL_DEBUG=${NCCL_DEBUG} \
    NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
    ${TORCHRUN_BIN} \
      --nnodes=${nnodes} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --node_rank=\${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      perf_pp_vpp/scripts/nccl_test_entry.py \
      --binary ${NCCL_TESTS_ROOT}/${bin_name} \
      -- -b ${FIRST} -e ${LAST} -f 2 -g 1
  "
  launch_node() {
    local pod="$1"; local rank="$2"
    ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- bash -lc "NODE_RANK=${rank}; ${remote_cmd}" > "${outfile}.node${rank}" 2>&1
  }
  if [[ "${nnodes}" == "1" ]]; then
    launch_node "${NODE0_POD}" 0
  else
    launch_node "${NODE1_POD}" 1 &
    local pid1=$!
    launch_node "${NODE0_POD}" 0
    wait "${pid1}"
  fi
  cat "${outfile}.node"* > "${outfile}" || true
}

run_test all_reduce_perf all_reduce_perf 1 "${OUT_DIR}/intra_node_all_reduce_perf.txt"
run_test all_reduce_perf all_reduce_perf 2 "${OUT_DIR}/inter_node_all_reduce_perf.txt"
run_test sendrecv_perf sendrecv_perf 2 "${OUT_DIR}/inter_node_sendrecv_perf.txt"
run_test alltoall_perf alltoall_perf 2 "${OUT_DIR}/inter_node_alltoall_perf.txt"
