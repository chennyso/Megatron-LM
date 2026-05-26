#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

export KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"
export NODE0_POD="${NODE0_POD:-chenny-dist-0}"
export NODE1_POD="${NODE1_POD:-chenny-dist-1}"
export MASTER_ADDR="${MASTER_ADDR:-chenny-dist-0.chenny-dist.default.svc.sspu.local}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q32_tp2_ppvpp_innov_$(date +%Y%m%d_%H%M%S)}"
export CAMPAIGN_ROOT="${PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
mkdir -p "${CAMPAIGN_ROOT}"
ln -sfn "${CAMPAIGN_ROOT}" "${PROJECT_ROOT}/perf_pp_vpp/outputs/current_tp2_ppvpp_campaign"
export PERF_RUNS_ROOT_LOCAL="${CAMPAIGN_ROOT}/runs"
export PERF_RUNS_ROOT_REMOTE="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}/runs"

export OUTPUT_ROOT="${CAMPAIGN_ROOT}"
export DATA_ROOT="${OUTPUT_ROOT}/data"
export CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
export PERF_TRAIN_ITERS="${PERF_TRAIN_ITERS:-6}"
export PERF_WARMUP_ITERS="${PERF_WARMUP_ITERS:-2}"
export PERF_PROFILE_START_ITER="${PERF_PROFILE_START_ITER:-3}"
export PERF_PROFILE_END_ITER="${PERF_PROFILE_END_ITER:-4}"
export PERF_TRANSFORMER_IMPL="${PERF_TRANSFORMER_IMPL:-transformer_engine}"
export PERF_USE_PRECISION_AWARE_OPTIMIZER="${PERF_USE_PRECISION_AWARE_OPTIMIZER:-1}"
export PERF_MAIN_GRADS_DTYPE="${PERF_MAIN_GRADS_DTYPE:-bf16}"
export PERF_EXP_AVG_DTYPE="${PERF_EXP_AVG_DTYPE:-bf16}"
export PERF_EXP_AVG_SQ_DTYPE="${PERF_EXP_AVG_SQ_DTYPE:-bf16}"

perf_pp_vpp/scripts/12_deploy_chenny_dist.sh
perf_pp_vpp/scripts/02_prepare_mock_data.sh

for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- bash -lc "pkill -f torchrun || true; pkill -f pretrain_gpt.py || true" || true
done

run_case() {
  local experiment="$1"
  local overlap_mode="$2"
  local run_tag="$3"
  local warmup_flush="$4"
  local overlap_optstep="$5"
  local group_size="$6"
  local repeat="$7"

  PERF_RUN_TAG="${run_tag}" \
  PERF_OVERLAP_P2P_WARMUP_FLUSH="${warmup_flush}" \
  PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${overlap_optstep}" \
  PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE="${group_size}" \
    perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b_tp2_ppvpp.yaml \
      --experiment-name "${experiment}" \
      --seq-len 4096 \
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

REPEATS="${REPEATS:-1}"
for repeat in $(seq 1 "${REPEATS}"); do
  run_case_safe q32_pp4_dp2_tp2_vpp2 baseline vpp2_baseline 0 0 "" "${repeat}"
  run_case_safe q32_pp4_dp2_tp2_vpp4 baseline vpp4_baseline 0 0 "" "${repeat}"
  run_case_safe q32_pp4_dp2_tp2_vpp4 all_overlap vpp4_all_overlap 0 0 "" "${repeat}"
  run_case_safe q32_pp4_dp2_tp2_vpp4 all_overlap_p2p_flush vpp4_flush 1 0 "" "${repeat}"
  run_case_safe q32_pp4_dp2_tp2_vpp4 all_overlap_p2p_flush vpp4_flush_optstep 1 1 "" "${repeat}"
  run_case_safe q32_pp4_dp2_tp2_vpp4 all_overlap_p2p_flush vpp4_flush_g8 1 0 "8" "${repeat}"
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
  warmup_flush="0"
  overlap_optstep="0"
  group_size=""
  if [[ "${run_tag}" == "vpp4_flush" ]]; then
    warmup_flush="1"
  elif [[ "${run_tag}" == "vpp4_flush_optstep" ]]; then
    warmup_flush="1"
    overlap_optstep="1"
  elif [[ "${run_tag}" == "vpp4_flush_g8" ]]; then
    warmup_flush="1"
    group_size="8"
  fi
  PERF_RUN_TAG="${run_tag}" \
  PERF_OVERLAP_P2P_WARMUP_FLUSH="${warmup_flush}" \
  PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${overlap_optstep}" \
  PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE="${group_size}" \
    perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b_tp2_ppvpp.yaml \
      --experiment-name "${experiment}" \
      --seq-len 4096 \
      --repeat-id 1 \
      --profile-ranks 0,3,4,7,8,11,12,15 \
      --overlap-mode "${overlap_mode}"
done

perf_pp_vpp/scripts/05_export_nsys_sqlite.sh "${OUTPUT_ROOT}/runs"
while IFS= read -r -d '' run; do
  ${PYTHON_BIN} perf_pp_vpp/scripts/06_parse_megatron_logs.py "${run}" || true
  ${PYTHON_BIN} perf_pp_vpp/scripts/07_parse_nsys_sqlite.py "${run}" || true
done < <(find "${OUTPUT_ROOT}/runs" -mindepth 4 -maxdepth 4 -type d -print0)
python perf_pp_vpp/scripts/10b_analyze_innovation_campaign.py
