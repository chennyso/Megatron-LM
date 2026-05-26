#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

CANDIDATE_DIR="${CANDIDATE_DIR:-${PROJECT_ROOT}/perf_pp_vpp/outputs/covpipe_candidates/q32_pp8_vpp4}"
EXPERIMENT_YAML="${EXPERIMENT_YAML:-perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-q32_pp8_dp1_tp2_vpp4}"
SEQ_LEN="${SEQ_LEN:-1024}"
REPEAT_BASE="${REPEAT_BASE:-8000}"
OVERLAP_MODE="${OVERLAP_MODE:-baseline}"
MAX_RUNS="${MAX_RUNS:-0}"

mkdir -p "${CANDIDATE_DIR}"
if [[ ! -f "${CANDIDATE_DIR}/manifest.json" ]]; then
  "${PYTHON_BIN}" perf_pp_vpp/scripts/14d_generate_covpipe_search_space.py --output-dir "${CANDIDATE_DIR}"
fi

ran=0
idx=0
while IFS= read -r candidate_path; do
  repeat_id="$((REPEAT_BASE + idx))"
  idx="$((idx + 1))"
  perf_pp_vpp/scripts/14c_run_covpipe_candidate.sh \
    "${candidate_path}" \
    "${EXPERIMENT_YAML}" \
    "${EXPERIMENT_NAME}" \
    "${SEQ_LEN}" \
    "${repeat_id}" \
    "${OVERLAP_MODE}" || true
  ran="$((ran + 1))"
  if [[ "${MAX_RUNS}" -gt 0 && "${ran}" -ge "${MAX_RUNS}" ]]; then
    break
  fi
done < <(find "${CANDIDATE_DIR}" -maxdepth 1 -type f -name '*.json' ! -name 'manifest.json' | sort)
