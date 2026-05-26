#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
cd "${PROJECT_ROOT}"

export KUBECTL_BIN="${KUBECTL_BIN:-kubectl --insecure-skip-tls-verify=true}"
export NODE0_POD="${NODE0_POD:-chenny-dist-0}"
export NODE1_POD="${NODE1_POD:-chenny-dist-1}"
export MASTER_ADDR="${MASTER_ADDR:-chenny-dist-0.chenny-dist.default.svc.sspu.local}"

CAMPAIGN_NAME="${CAMPAIGN_NAME:-q32_ppvpp_deep_$(date +%Y%m%d_%H%M%S)}"
PPVPP_SEQ_LEN="${PPVPP_SEQ_LEN:-2048}"
PPVPP_REPEATS="${PPVPP_REPEATS:-3}"
PPVPP_TOPK="${PPVPP_TOPK:-3}"
export OUTPUT_ROOT="${PROJECT_ROOT}/perf_pp_vpp/outputs/campaigns/${CAMPAIGN_NAME}"
export DATA_ROOT="${OUTPUT_ROOT}/data"
export CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
mkdir -p "${OUTPUT_ROOT}" "${DATA_ROOT}" "${CHECKPOINT_ROOT}"

perf_pp_vpp/scripts/12_deploy_chenny_dist.sh
perf_pp_vpp/scripts/02_prepare_mock_data.sh

BASELINE_TARGETS=(
  "q32_pp16_dp1_tp1_vpp1 ${PPVPP_SEQ_LEN}"
  "q32_pp16_dp1_tp1_vpp2 ${PPVPP_SEQ_LEN}"
  "q32_pp16_dp1_tp1_vpp4 ${PPVPP_SEQ_LEN}"
  "q32_pp8_dp2_tp1_vpp2 ${PPVPP_SEQ_LEN}"
  "q32_pp8_dp2_tp1_vpp4 ${PPVPP_SEQ_LEN}"
)

for item in "${BASELINE_TARGETS[@]}"; do
  read -r name seq <<<"${item}"
  for repeat in $(seq 1 "${PPVPP_REPEATS}"); do
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

mapfile -t TOP_TARGETS < <(
  python - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"]) / "runs"
seq_dir = f"seq{os.environ['PPVPP_SEQ_LEN']}"
scores = defaultdict(list)
for path in root.glob(f"*/{seq_dir}/repeat*/baseline/step_summary.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("oom"):
        continue
    value = payload.get("tokens_per_second_mean")
    if value is None:
        value = payload.get("tflops_mean")
    if value is None:
        continue
    scores[path.parts[-5]].append(float(value))

ranked = sorted(
    ((sum(values) / len(values), name) for name, values in scores.items() if values),
    reverse=True,
)
limit = int(os.environ.get("PPVPP_TOPK", "3"))
for _, name in ranked[:limit]:
    print(name)
PY
)

if [[ ${#TOP_TARGETS[@]} -eq 0 ]]; then
  echo "no successful baseline runs found under ${OUTPUT_ROOT}" >&2
  exit 1
fi

printf '%s\n' "${TOP_TARGETS[@]}" > "${OUTPUT_ROOT}/top_ppvpp_targets.txt"

for name in "${TOP_TARGETS[@]}"; do
  overlap_modes=(baseline p2p_overlap)
  if [[ "${name}" == *"_dp2_"* ]]; then
    overlap_modes+=(dp_overlap all_overlap)
  fi
  for overlap_mode in "${overlap_modes[@]}"; do
    perf_pp_vpp/scripts/04_run_nsys_experiment.sh \
      --experiment-yaml perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml \
      --experiment-name "${name}" \
      --seq-len "${PPVPP_SEQ_LEN}" \
      --repeat-id 1 \
      --profile-ranks 0,7,8,15 \
      --overlap-mode "${overlap_mode}"
  done
done

perf_pp_vpp/scripts/05_export_nsys_sqlite.sh
while IFS= read -r -d '' run; do
  ${PYTHON_BIN} perf_pp_vpp/scripts/06_parse_megatron_logs.py "${run}" || true
  ${PYTHON_BIN} perf_pp_vpp/scripts/07_parse_nsys_sqlite.py "${run}" || true
done < <(find "${OUTPUT_ROOT}/runs" -mindepth 4 -maxdepth 4 -type d -print0)
${PYTHON_BIN} perf_pp_vpp/scripts/08_merge_metrics.py "${OUTPUT_ROOT}"
${PYTHON_BIN} perf_pp_vpp/scripts/09_generate_report.py
