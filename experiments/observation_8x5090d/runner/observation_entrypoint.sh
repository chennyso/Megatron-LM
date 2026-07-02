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

source "${REPO_DIR}/experiments/observation_8x5090d/runner/activate_observation_env.sh"

"${OBS_PYTHON}" --version
nvidia-smi || true

MATRIX_PATH="experiments/observation_8x5090d/configs/observation_matrix.json"

if [ "${PHASE}" != "hardware" ]; then
  DATASET_SPEC_ID="$(
    "${OBS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

matrix = json.loads(Path("experiments/observation_8x5090d/configs/observation_matrix.json").read_text(encoding="utf-8"))
phase = os.environ["PHASE"]
case_id = os.environ.get("CASE_ID") or ""
for case in matrix["cases"]:
    if case["phase"] != phase:
        continue
    if case_id and case["id"] != case_id:
        continue
    print(case["dataset_spec"])
    raise SystemExit(0)
raise SystemExit(f"Could not resolve dataset spec for phase={phase!r} case_id={case_id!r}")
PY
  )"
  export DATASET_SPEC_ID
  "${OBS_PYTHON}" experiments/observation_8x5090d/scripts/prepare_observation_dataset.py \
    --matrix-path "${MATRIX_PATH}" \
    --dataset-spec-id "${DATASET_SPEC_ID}" \
    --output-root "/workspace/datasets/$( \
      "${OBS_PYTHON}" - <<'PY'
import json
from pathlib import Path

matrix = json.loads(Path("experiments/observation_8x5090d/configs/observation_matrix.json").read_text(encoding="utf-8"))
dataset = matrix["datasets"][__import__("os").environ["DATASET_SPEC_ID"]]
print(Path(dataset["data_path"]).parent.relative_to("/workspace/datasets"))
PY
    )"
fi

if [ "${PHASE}" = "hardware" ]; then
  "${OBS_PYTHON}" experiments/observation_8x5090d/scripts/run_hardware_profile.py \
    --matrix-path "${MATRIX_PATH}" \
    --output-dir "${RESULT_ROOT}/hardware"
  exit 0
fi

"${OBS_PYTHON}" experiments/observation_8x5090d/scripts/run_megatron_observation.py \
  --matrix-path "${MATRIX_PATH}" \
  --phase "${PHASE}" \
  --output-dir "${RESULT_ROOT}/${PHASE}" \
  ${CASE_ID:+--case-id "${CASE_ID}"}
