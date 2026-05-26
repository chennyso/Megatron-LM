#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q14_topology_layout_$(date +%Y%m%d)}"
CAMPAIGN_ROOT="${OUTPUT_ROOT}/campaigns/${CAMPAIGN_NAME}"
REMOTE_CAMPAIGN_ROOT="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
mkdir -p "${CAMPAIGN_ROOT}"

export PERF_RUNS_ROOT_LOCAL="${CAMPAIGN_ROOT}/runs"
export PERF_RUNS_ROOT_REMOTE="${REMOTE_CAMPAIGN_ROOT}/runs"
export PERF_TRANSFORMER_IMPL="${PERF_TRANSFORMER_IMPL:-transformer_engine}"

run_case() {
  local exp_name="$1"
  local repeat_id="$2"
  local run_tag="$3"
  local layout_mode="${4:-}"
  local vpp="$5"

  local layout=""
  if [[ -n "${layout_mode}" ]]; then
    layout="$(${PYTHON_BIN} perf_pp_vpp/scripts/13_generate_topology_aware_layout.py \
      --pp 16 --vpp "${vpp}" --num-layers 40 --mode "${layout_mode}")"
  fi

  PERF_PIPELINE_MODEL_PARALLEL_LAYOUT="${layout}" \
  PERF_RUN_TAG="${run_tag}" \
  PERF_TRAIN_ITERS="${PERF_TRAIN_ITERS:-6}" \
  PERF_WARMUP_ITERS="${PERF_WARMUP_ITERS:-2}" \
  PERF_PROFILE_START_ITER="${PERF_PROFILE_START_ITER:-3}" \
  PERF_PROFILE_END_ITER="${PERF_PROFILE_END_ITER:-4}" \
  perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
    --experiment-yaml perf_pp_vpp/configs/experiments/phase1_qwen14b_topology_aware.yaml \
    --experiment-name "${exp_name}" \
    --seq-len 4096 \
    --repeat-id "${repeat_id}" \
    --profile-mode none \
    --profile-ranks none \
    --overlap-mode baseline
}

run_case q14_pp16_dp1_tp1_vpp1 3101 topology_guarded_vpp1 topology_guarded 1 || true
run_case q14_pp16_dp1_tp1_vpp2 3102 uniform_vpp2 uniform 2 || true
run_case q14_pp16_dp1_tp1_vpp2 3103 topology_guarded_vpp2 topology_guarded 2 || true
run_case q14_pp16_dp1_tp1_vpp2 3104 boundary_dilated_vpp2 boundary_dilated 2 || true

CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" ${PYTHON_BIN} perf_pp_vpp/scripts/10c_analyze_topology_layout_campaign.py
printf '%s\n' "${CAMPAIGN_ROOT}"
