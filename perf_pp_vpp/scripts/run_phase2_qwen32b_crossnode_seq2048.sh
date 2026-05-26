#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q32_crossnode_seq2048_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_ROOT="${OUTPUT_ROOT}/campaigns/${CAMPAIGN_NAME}"
REMOTE_CAMPAIGN_ROOT="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
mkdir -p "${CAMPAIGN_ROOT}"

export PERF_RUNS_ROOT_LOCAL="${CAMPAIGN_ROOT}/runs"
export PERF_RUNS_ROOT_REMOTE="${REMOTE_CAMPAIGN_ROOT}/runs"
export PERF_TRANSFORMER_IMPL="${PERF_TRANSFORMER_IMPL:-transformer_engine}"

TARGETS=(
  "q32_pp8_dp1_tp2_vpp2 4201"
  "q32_pp8_dp1_tp2_vpp4 4202"
  "q32_pp16_dp1_tp1_vpp2 4203"
  "q32_pp16_dp1_tp1_vpp4 4204"
)

for item in "${TARGETS[@]}"; do
  read -r name repeat_id <<<"${item}"
  PERF_TRAIN_ITERS="${PERF_TRAIN_ITERS:-4}" \
  PERF_WARMUP_ITERS="${PERF_WARMUP_ITERS:-2}" \
  PERF_PROFILE_START_ITER="${PERF_PROFILE_START_ITER:-3}" \
  PERF_PROFILE_END_ITER="${PERF_PROFILE_END_ITER:-4}" \
    perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
      --experiment-name "${name}" \
      --seq-len 2048 \
      --repeat-id "${repeat_id}" \
      --profile-mode none \
      --profile-ranks none \
      --overlap-mode baseline || true
done

printf '%s\n' "${CAMPAIGN_ROOT}"
