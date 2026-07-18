#!/usr/bin/env bash
set -euo pipefail

PHASE="${PHASE:?PHASE is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
GIT_REMOTE_URL="${GIT_REMOTE_URL:?GIT_REMOTE_URL is required}"
GIT_REF="${GIT_REF:?GIT_REF is required}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/runs/observation_8x5090d/${RUN_ID}}"
REPO_DIR="${REPO_DIR:-/workspace/code/Megatron-LM-observation}"
CASE_ID="${CASE_ID:-}"

mkdir -p "$(dirname "${REPO_DIR}")" "${RESULT_ROOT}"
export GIT_SSL_NO_VERIFY=1

if [ -f "${REPO_DIR}/.git/index.lock" ]; then
  rm -f "${REPO_DIR}/.git/index.lock"
fi

if [ ! -d "${REPO_DIR}/.git" ]; then
  git -c http.sslVerify=false clone --filter=blob:none --no-checkout "${GIT_REMOTE_URL}" "${REPO_DIR}"
fi

if EXPECTED_COMMIT="$(git -C "${REPO_DIR}" rev-parse --verify "${GIT_REF}^{commit}" 2>/dev/null)" && \
   [ "$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || true)" = "${EXPECTED_COMMIT}" ]; then
  echo "[observation-code] reusing ${REPO_DIR} at ${EXPECTED_COMMIT}"
else
  git -C "${REPO_DIR}" -c http.sslVerify=false fetch --depth 1 origin "${GIT_REF}"
  git -C "${REPO_DIR}" checkout --detach FETCH_HEAD
fi

EXPECTED_COMMIT="$(git -C "${REPO_DIR}" rev-parse --verify "${GIT_REF}^{commit}")"
ACTUAL_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
  echo "[observation-code] commit mismatch: expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2
  exit 1
fi
echo "[observation-code] verified commit ${ACTUAL_COMMIT}"

cd "${REPO_DIR}"

source "${REPO_DIR}/experiments/observation_8x5090d/runner/activate_observation_env.sh"

"${OBS_PYTHON}" --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[observation-env] nvidia-smi not found; skipping GPU query"
fi

MATRIX_PATH="${MATRIX_PATH:-experiments/observation_8x5090d/configs/observation_matrix.json}"
export MATRIX_PATH

if [ "${PHASE}" != "hardware" ] && [ "${PHASE}" != "motif" ]; then
  DATASET_SPEC_ID="$(
    "${OBS_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

matrix = json.loads(Path(os.environ["MATRIX_PATH"]).read_text(encoding="utf-8"))
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
import os
from pathlib import Path

matrix = json.loads(Path(os.environ["MATRIX_PATH"]).read_text(encoding="utf-8"))
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

if [ "${PHASE}" = "motif" ]; then
  MOTIF_DIR="${RESULT_ROOT}/motif"
  mkdir -p "${MOTIF_DIR}"
  nvidia-smi dmon -s pucvmet -d 1 -o DT > "${MOTIF_DIR}/nvidia-smi-dmon.log" 2>&1 &
  DMON_PID=$!
  trap 'kill "${DMON_PID}" 2>/dev/null || true; wait "${DMON_PID}" 2>/dev/null || true' EXIT
  MOTIF_TARGET="${OBS_MOTIF_TARGET:-all}"
  if [ "${MOTIF_TARGET}" = "all" ]; then
    torchrun --standalone --nproc_per_node=8 \
      experiments/observation_8x5090d/scripts/benchmark_concurrent_motifs.py \
      --matrix-path "${MATRIX_PATH}" \
      --output-dir "${MOTIF_DIR}"
  fi
  if [ "${MOTIF_TARGET}" = "all" ] || [ "${MOTIF_TARGET}" = "compute-comm" ]; then
    COMPUTE_CMD=(
      torchrun --standalone --nproc_per_node=8
      experiments/observation_8x5090d/scripts/benchmark_compute_comm_overlap.py
      --matrix-path "${MATRIX_PATH}"
      --output-dir "${MOTIF_DIR}/compute_comm"
    )
    if [ -n "${OBS_COMPUTE_COMM_CASE_ID:-}" ]; then
      COMPUTE_CMD+=(--case-id "${OBS_COMPUTE_COMM_CASE_ID}")
    fi
    if [ -n "${OBS_COMPUTE_COMM_LOCATIONS:-}" ]; then
      IFS=',' read -r -a COMPUTE_LOCATIONS <<< "${OBS_COMPUTE_COMM_LOCATIONS}"
      for location in "${COMPUTE_LOCATIONS[@]}"; do
        COMPUTE_CMD+=(--compute-location "${location}")
      done
    fi
    if [ "${OBS_MOTIF_NSYS:-0}" = "1" ]; then
      mkdir -p "${MOTIF_DIR}/compute_comm"
      NSYS_PREFIX="${MOTIF_DIR}/compute_comm/compute_comm_nsys"
      nsys profile \
        --trace=cuda,nvtx,nccl,cublas \
        --sample=none \
        --cpuctxsw=none \
        --trace-fork-before-exec=true \
        --stats=false \
        --force-overwrite=true \
        --output="${NSYS_PREFIX}" \
        "${COMPUTE_CMD[@]}"
      nsys export \
        --type=sqlite \
        --force-overwrite=true \
        --output="${NSYS_PREFIX}.sqlite" \
        "${NSYS_PREFIX}.nsys-rep"
    else
      "${COMPUTE_CMD[@]}"
    fi
  fi
  if [ "${MOTIF_TARGET}" != "all" ] && [ "${MOTIF_TARGET}" != "compute-comm" ]; then
    echo "Unsupported OBS_MOTIF_TARGET=${MOTIF_TARGET}" >&2
    exit 2
  fi
  exit 0
fi

"${OBS_PYTHON}" experiments/observation_8x5090d/scripts/run_megatron_observation.py \
  --matrix-path "${MATRIX_PATH}" \
  --phase "${PHASE}" \
  --output-dir "${RESULT_ROOT}/${PHASE}" \
  ${CASE_ID:+--case-id "${CASE_ID}"}
