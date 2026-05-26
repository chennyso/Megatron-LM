#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

perf_pp_vpp/scripts/02_prepare_mock_data.sh

TARGETS=(
  "q32_pp16_dp1_tp1_vpp1 4096"
  "q32_pp16_dp1_tp1_vpp2 4096"
  "q32_pp16_dp1_tp1_vpp4 4096"
  "q32_pp8_dp2_tp1_vpp1 4096"
  "q32_pp8_dp2_tp1_vpp2 4096"
)

for item in "${TARGETS[@]}"; do
  read -r name seq <<<"${item}"
  for repeat in 1 2 3; do
    perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
      --experiment-name "${name}" \
      --seq-len "${seq}" \
      --repeat-id "${repeat}" \
      --profile-mode none \
      --profile-ranks none \
      --overlap-mode baseline || true
  done
  perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
    --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
    --experiment-name "${name}" \
    --seq-len "${seq}" \
    --repeat-id 1 \
    --profile-ranks 0,7,8,15 \
    --overlap-mode baseline || true
done

perf_pp_vpp/scripts/05_export_nsys_sqlite.sh || true
while IFS= read -r -d '' run; do
  ${PYTHON_BIN} perf_pp_vpp/scripts/06_parse_megatron_logs.py "${run}" || true
  ${PYTHON_BIN} perf_pp_vpp/scripts/07_parse_nsys_sqlite.py "${run}" || true
done < <(find perf_pp_vpp/outputs/runs -mindepth 4 -maxdepth 4 -type d -print0)
${PYTHON_BIN} perf_pp_vpp/scripts/08_merge_metrics.py perf_pp_vpp/outputs
${PYTHON_BIN} perf_pp_vpp/scripts/09_generate_report.py
