#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

export KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"
export MASTER_ADDR="${MASTER_ADDR:-chenny-dist-0.chenny-dist.default.svc.sspu.local}"
export PERF_TRANSFORMER_IMPL="${PERF_TRANSFORMER_IMPL:-transformer_engine}"
export PERF_USE_PRECISION_AWARE_OPTIMIZER=0
export PERF_MAIN_GRADS_DTYPE="${PERF_MAIN_GRADS_DTYPE:-bf16}"
export PERF_EXP_AVG_DTYPE="${PERF_EXP_AVG_DTYPE:-bf16}"
export PERF_EXP_AVG_SQ_DTYPE="${PERF_EXP_AVG_SQ_DTYPE:-bf16}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q32_tp2_nonuniform_nsys_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_ROOT="${PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
REMOTE_CAMPAIGN_ROOT="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
mkdir -p "${CAMPAIGN_ROOT}"

export OUTPUT_ROOT="${CAMPAIGN_ROOT}"
export DATA_ROOT="${OUTPUT_ROOT}/data"
export CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
export PERF_RUNS_ROOT_LOCAL="${CAMPAIGN_ROOT}/runs"
export PERF_RUNS_ROOT_REMOTE="${REMOTE_CAMPAIGN_ROOT}/runs"
export PERF_TRAIN_ITERS="${PERF_TRAIN_ITERS:-4}"
export PERF_WARMUP_ITERS="${PERF_WARMUP_ITERS:-2}"
export PERF_PROFILE_START_ITER="${PERF_PROFILE_START_ITER:-3}"
export PERF_PROFILE_END_ITER="${PERF_PROFILE_END_ITER:-4}"
export PERF_GLOBAL_BATCH_SIZE="${PERF_GLOBAL_BATCH_SIZE:-32}"

NONUNIFORM_LAYOUT="${NONUNIFORM_LAYOUT:-Ett|tt|tt|t|tt|tt|tt|ttt|tt|tt|tt|ttt|tt|tt|tt|t|tt|tt|tt|t|tt|tt|tt|ttt|tt|tt|tt|ttt|t|tt|tt|ttL}"
PROFILE_RANKS="${PROFILE_RANKS:-6,7,8,9}"

perf_pp_vpp/scripts/02_prepare_mock_data.sh

for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "pkill -9 python3 || true" >/dev/null 2>&1 || true
done
sleep 2

unset PERF_PIPELINE_MODEL_PARALLEL_LAYOUT PERF_NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK PERF_DECODER_FIRST_PIPELINE_NUM_LAYERS PERF_DECODER_LAST_PIPELINE_NUM_LAYERS PERF_ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT PERF_ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT
perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
  --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
  --experiment-name q32_pp8_dp1_tp2_vpp4 \
  --seq-len "${SEQ_LEN:-2048}" \
  --repeat-id "${UNIFORM_REPEAT_ID:-6401}" \
  --profile-ranks "${PROFILE_RANKS}" \
  --overlap-mode baseline || true

for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "pkill -9 python3 || true" >/dev/null 2>&1 || true
done
sleep 2

export PERF_PIPELINE_MODEL_PARALLEL_LAYOUT="${NONUNIFORM_LAYOUT}"
perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
  --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
  --experiment-name q32_pp8_dp1_tp2_vpp4 \
  --seq-len "${SEQ_LEN:-2048}" \
  --repeat-id "${NONUNIFORM_REPEAT_ID:-6402}" \
  --profile-ranks "${PROFILE_RANKS}" \
  --overlap-mode baseline || true

printf '%s\n' "${CAMPAIGN_ROOT}"
