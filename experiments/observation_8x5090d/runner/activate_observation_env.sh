#!/usr/bin/env bash
set -euo pipefail

OBS_VENV_DIR="${OBS_VENV_DIR:-/workspace/venvs/observation-8x5090d}"
export OBS_VENV_DIR
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [ -x "${OBS_VENV_DIR}/bin/python" ]; then
  export OBS_PYTHON="${OBS_VENV_DIR}/bin/python"
  # The observation jobs run in a stock NGC image; prefer the shared PVC venv when present.
  # It is created with --system-site-packages so torch/TE from the image remain visible.
  # shellcheck disable=SC1091
  source "${OBS_VENV_DIR}/bin/activate"
  echo "[observation-env] activated ${OBS_VENV_DIR}"
else
  export OBS_PYTHON="$(command -v python3)"
  echo "[observation-env] shared venv not found at ${OBS_VENV_DIR}; using system Python"
fi

"${OBS_PYTHON}" --version
