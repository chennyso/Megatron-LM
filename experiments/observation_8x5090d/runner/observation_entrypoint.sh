#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:?PHASE is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
GIT_REMOTE_URL="${GIT_REMOTE_URL:?GIT_REMOTE_URL is required}"
GIT_BRANCH="${GIT_BRANCH:?GIT_BRANCH is required}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/runs/observation_8x5090d/${RUN_ID}}"
REPO_DIR="${REPO_DIR:-/workspace/code/Megatron-LM-observation}"
CASE_ID="${CASE_ID:-}"

mkdir -p "$(dirname "${REPO_DIR}")" "${RESULT_ROOT}"
export GIT_SSL_NO_VERIFY=1

if [ ! -d "${REPO_DIR}/.git" ]; then
  git -c http.sslVerify=false clone --branch "${GIT_BRANCH}" --single-branch "${GIT_REMOTE_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" -c http.sslVerify=false fetch origin "${GIT_BRANCH}"
  git -C "${REPO_DIR}" checkout "${GIT_BRANCH}"
  git -C "${REPO_DIR}" reset --hard "origin/${GIT_BRANCH}"
fi

cd "${REPO_DIR}"

python3 --version
nvidia-smi || true

MATRIX_PATH="experiments/observation_8x5090d/configs/observation_matrix.json"

if [ "${PHASE}" = "hardware" ]; then
  python3 experiments/observation_8x5090d/scripts/run_hardware_profile.py \
    --matrix-path "${MATRIX_PATH}" \
    --output-dir "${RESULT_ROOT}/hardware"
  exit 0
fi

python3 experiments/observation_8x5090d/scripts/run_megatron_observation.py \
  --matrix-path "${MATRIX_PATH}" \
  --phase "${PHASE}" \
  --output-dir "${RESULT_ROOT}/${PHASE}" \
  ${CASE_ID:+--case-id "${CASE_ID}"}
