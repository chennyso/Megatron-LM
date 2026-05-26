#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

export KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"
export NODE0_POD="${NODE0_POD:-chenny-dist-0}"
export NODE1_POD="${NODE1_POD:-chenny-dist-1}"
export MASTER_ADDR="${MASTER_ADDR:-chenny-dist-0.chenny-dist.default.svc.sspu.local}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q32_overlap_innov_$(date +%Y%m%d_%H%M%S)}"
export CAMPAIGN_ROOT="${PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
mkdir -p "${CAMPAIGN_ROOT}"
ln -sfn "${CAMPAIGN_ROOT}" "${PROJECT_ROOT}/perf_pp_vpp/outputs/current_innovation_campaign"

export OUTPUT_ROOT="${CAMPAIGN_ROOT}"
export DATA_ROOT="${OUTPUT_ROOT}/data"
export CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
export PPVPP_SEQ_LEN="${PPVPP_SEQ_LEN:-1024}"
export PERF_TRAIN_ITERS="${PERF_TRAIN_ITERS:-6}"
export PERF_WARMUP_ITERS="${PERF_WARMUP_ITERS:-2}"
export PERF_PROFILE_START_ITER="${PERF_PROFILE_START_ITER:-3}"
export PERF_PROFILE_END_ITER="${PERF_PROFILE_END_ITER:-5}"
export PERF_USE_PRECISION_AWARE_OPTIMIZER="${PERF_USE_PRECISION_AWARE_OPTIMIZER:-0}"
export PERF_GLOBAL_BATCH_SIZE_PP16="${PERF_GLOBAL_BATCH_SIZE_PP16:-16}"
export PERF_GLOBAL_BATCH_SIZE_DP2="${PERF_GLOBAL_BATCH_SIZE_DP2:-16}"

perf_pp_vpp/scripts/12_deploy_chenny_dist.sh
perf_pp_vpp/scripts/02_prepare_mock_data.sh

for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- bash -lc "pkill -f torchrun || true; pkill -f pretrain_gpt.py || true" || true
done

run_case() {
  local experiment="$1"
  local overlap_mode="$2"
  local run_tag="$3"
  local transformer_impl="$4"
  local warmup_flush="$5"
  local delay_wgrad="$6"
  local group_size="$7"
  local overlap_optstep="$8"
  local global_batch="$9"
  local repeat="${10}"

  PERF_RUN_TAG="${run_tag}" \
  PERF_TRANSFORMER_IMPL="${transformer_impl}" \
  PERF_OVERLAP_P2P_WARMUP_FLUSH="${warmup_flush}" \
  PERF_DELAY_WGRAD_COMPUTE="${delay_wgrad}" \
  PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE="${group_size}" \
  PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${overlap_optstep}" \
  PERF_GLOBAL_BATCH_SIZE="${global_batch}" \
    perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
      --experiment-name "${experiment}" \
      --seq-len "${PPVPP_SEQ_LEN}" \
      --repeat-id "${repeat}" \
      --profile-mode none \
      --profile-ranks none \
      --overlap-mode "${overlap_mode}"
}

run_case_safe() {
  if ! run_case "$@"; then
    echo "case failed: $*" >&2
  fi
}

REPEATS="${REPEATS:-2}"
for repeat in $(seq 1 "${REPEATS}"); do
  run_case_safe q32_pp8_dp2_tp1_vpp2 baseline vpp2_baseline local 0 0 "" 0 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 baseline vpp4_baseline local 0 0 "" 0 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 all_overlap dp2_all_overlap local 0 0 "" 0 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 all_overlap_p2p_flush dp2_all_flush local 1 0 "" 0 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 all_overlap_p2p_flush dp2_all_flush_g8 local 1 0 "8" 0 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 all_overlap dp2_all_optstep local 0 0 "" 1 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
  run_case_safe q32_pp8_dp2_tp1_vpp4 all_overlap_p2p_flush dp2_all_flush_optstep local 1 0 "" 1 "${PERF_GLOBAL_BATCH_SIZE_DP2}" "${repeat}"
done

python perf_pp_vpp/scripts/10b_analyze_innovation_campaign.py

mapfile -t TOP2 < <(
  python - <<'PY'
import json
import os
from pathlib import Path
rows = json.loads((Path(os.environ["CAMPAIGN_ROOT"]) / "innovation_summary.json").read_text())
best = [r for r in rows if r["mean_step_time_ms"] is not None and r["valid_repeats"] > 0]
best.sort(key=lambda x: x["mean_step_time_ms"])
for row in best[:2]:
    print("\t".join([row["experiment"], row["overlap_mode"], row["run_tag"]]))
PY
)

for item in "${TOP2[@]}"; do
  IFS=$'\t' read -r experiment overlap_mode run_tag <<<"${item}"
  transformer_impl="local"
  warmup_flush="0"
  delay_wgrad="0"
  group_size=""
  overlap_optstep="0"
  if [[ "${run_tag}" == "vpp2_flush" || "${run_tag}" == "vpp4_flush" ]]; then
    warmup_flush="1"
  elif [[ "${run_tag}" == "vpp2_zbw" || "${run_tag}" == "vpp4_zbw" ]]; then
    transformer_impl="transformer_engine"
    warmup_flush="1"
    delay_wgrad="1"
  elif [[ "${run_tag}" == "dp2_all_flush" ]]; then
    warmup_flush="1"
  elif [[ "${run_tag}" == "dp2_all_flush_g8" ]]; then
    warmup_flush="1"
    group_size="8"
  elif [[ "${run_tag}" == "dp2_all_optstep" ]]; then
    overlap_optstep="1"
  elif [[ "${run_tag}" == "dp2_all_flush_optstep" ]]; then
    warmup_flush="1"
    overlap_optstep="1"
  fi
  PERF_RUN_TAG="${run_tag}" \
  PERF_TRANSFORMER_IMPL="${transformer_impl}" \
  PERF_OVERLAP_P2P_WARMUP_FLUSH="${warmup_flush}" \
  PERF_DELAY_WGRAD_COMPUTE="${delay_wgrad}" \
  PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE="${group_size}" \
  PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${overlap_optstep}" \
  PERF_GLOBAL_BATCH_SIZE="$([[ "${experiment}" == q32_pp16_* ]] && echo "${PERF_GLOBAL_BATCH_SIZE_PP16}" || echo "${PERF_GLOBAL_BATCH_SIZE_DP2}")" \
    perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
      --experiment-name "${experiment}" \
      --seq-len "${PPVPP_SEQ_LEN}" \
      --repeat-id 1 \
      --profile-ranks 0,7,8,15 \
      --overlap-mode "${overlap_mode}"
done

perf_pp_vpp/scripts/05_export_nsys_sqlite.sh "${OUTPUT_ROOT}/runs"
while IFS= read -r -d '' run; do
  ${PYTHON_BIN} perf_pp_vpp/scripts/06_parse_megatron_logs.py "${run}" || true
  ${PYTHON_BIN} perf_pp_vpp/scripts/07_parse_nsys_sqlite.py "${run}" || true
done < <(find "${OUTPUT_ROOT}/runs" -mindepth 4 -maxdepth 4 -type d -print0)
python perf_pp_vpp/scripts/10b_analyze_innovation_campaign.py
