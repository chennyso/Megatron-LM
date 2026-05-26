#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

export KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"
export NODE0_POD="${NODE0_POD:-chenny-dist-0}"
export NODE1_POD="${NODE1_POD:-chenny-dist-1}"
export MASTER_ADDR="${MASTER_ADDR:-chenny-dist-0.chenny-dist.default.svc.sspu.local}"

perf_pp_vpp/scripts/12_deploy_chenny_dist.sh
perf_pp_vpp/scripts/02_prepare_mock_data.sh

TARGETS=(
  "q32_pp8_dp2_tp1_vpp1 4096 local"
  "q32_pp8_dp2_tp1_vpp2 4096 local"
  "q32_pp8_dp1_tp2_vpp1 4096 transformer_engine"
  "q32_pp8_dp1_tp2_vpp2 4096 transformer_engine"
)

for item in "${TARGETS[@]}"; do
  read -r name seq transformer_impl <<<"${item}"
  for repeat in 1 2 3; do
    PERF_TRANSFORMER_IMPL="${transformer_impl}" \
      perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
        --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
        --experiment-name "${name}" \
        --seq-len "${seq}" \
        --repeat-id "${repeat}" \
        --profile-mode none \
        --profile-ranks none \
        --overlap-mode baseline
  done
done
