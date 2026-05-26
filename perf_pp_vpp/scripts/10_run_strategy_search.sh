#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
source perf_pp_vpp/configs/env.sh

MODEL="${MODEL:-qwen3_32b}"
SEQ_LEN="${SEQ_LEN:-4096}"
REPEAT_BASE="${REPEAT_BASE:-100}"
TRAIN_ITERS="${TRAIN_ITERS:-6}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
PROFILE_START="${PROFILE_START:-3}"
PROFILE_END="${PROFILE_END:-4}"
OVERLAP_MODE="${OVERLAP_MODE:-baseline}"
STRATEGY_FILTER="${STRATEGY_FILTER:-}"
MAX_RUNS="${MAX_RUNS:-0}"
case "${MODEL}" in
  qwen3_32b) MODEL_TAG="q32" ;;
  qwen3_14b) MODEL_TAG="q14" ;;
  *) MODEL_TAG="${MODEL}" ;;
esac

SUMMARY_DIR="${OUTPUT_ROOT}/strategy_search/${MODEL}/seq${SEQ_LEN}"
TMP_DIR="${SUMMARY_DIR}/tmp"
mkdir -p "${SUMMARY_DIR}" "${TMP_DIR}"
SUMMARY_CSV="${SUMMARY_DIR}/summary.csv"

printf 'experiment,model,seq_len,pp,tp,dp,vpp,repeat_id,rc,status,reason\n' > "${SUMMARY_CSV}"

python "${ROOT_DIR}/perf_pp_vpp/scripts/_common.py" >/dev/null 2>&1 || true

mapfile -t CANDIDATES < <(
python - <<'PY'
import os
from pathlib import Path
from perf_pp_vpp.scripts._common import resolve_remote_model_config, world_size

model_name = os.environ["MODEL"]
if model_name == "qwen3_32b":
    pp_order = [4, 8, 16, 2]
    vpp_order = [1, 2, 4]
elif model_name == "qwen3_14b":
    pp_order = [4, 8, 2]
    vpp_order = [1, 2, 4]
else:
    raise SystemExit(f"unsupported model for strategy search: {model_name}")

world = world_size()
model = resolve_remote_model_config(model_name)
layers = int(model["num_layers"])

for pp in pp_order:
    if world % pp != 0:
        continue
    for tp in [1, 2, 4, 8, 16]:
        denom = pp * tp
        if denom <= 0 or world % denom != 0:
            continue
        dp = world // denom
        if dp <= 0:
            continue
        for vpp in vpp_order:
            if layers % (pp * vpp) != 0:
                continue
            print(f"{pp},{tp},{dp},{vpp}")
PY
)

ran=0
for idx in "${!CANDIDATES[@]}"; do
  IFS=',' read -r PP TP DP VPP <<<"${CANDIDATES[$idx]}"
  EXP_NAME="${MODEL_TAG}_pp${PP}_dp${DP}_tp${TP}_vpp${VPP}"
  if [[ -n "${STRATEGY_FILTER}" && "${EXP_NAME}" != *"${STRATEGY_FILTER}"* ]]; then
    continue
  fi
  YAML_PATH="${TMP_DIR}/${EXP_NAME}.yaml"
  REPEAT_ID="$((REPEAT_BASE + idx))"

  cat > "${YAML_PATH}" <<EOF
model: ${MODEL}
seq_lengths: [${SEQ_LEN}]
micro_batch_size: 1
precision: bf16
activation_checkpointing: true
distributed_optimizer: true
train_iters: ${TRAIN_ITERS}
warmup_iters: ${WARMUP_ITERS}
profile_start_iter: ${PROFILE_START}
profile_end_iter: ${PROFILE_END}
repeat: 1
overlap_modes: [${OVERLAP_MODE}]
experiments:
  - name: ${EXP_NAME}
    pp: ${PP}
    dp: ${DP}
    tp: ${TP}
    vpp: ${VPP}
EOF

  ./perf_pp_vpp/scripts/11_clean_remote_python.sh >/dev/null 2>&1 || true

  set +e
  ./perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
    --experiment-yaml "${YAML_PATH}" \
    --experiment-name "${EXP_NAME}" \
    --seq-len "${SEQ_LEN}" \
    --repeat-id "${REPEAT_ID}" \
    --overlap-mode "${OVERLAP_MODE}" \
    --profile-mode none \
    --profile-ranks none >/dev/null 2>&1
  RC=$?
  set -e

  RUN_DIR="${OUTPUT_ROOT}/runs/${EXP_NAME}/seq${SEQ_LEN}/repeat${REPEAT_ID}/${OVERLAP_MODE}"
  REASON="$(python - <<'PY' "${RUN_DIR}"
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
patterns = [
    "CUDA out of memory",
    "OutOfMemory",
    "ncclUnhandledCudaError",
    "DistBackendError",
    "world_size mismatch",
    "not divisible",
    "Traceback",
]
lines = []
for path in sorted(run_dir.rglob("*.log")):
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if any(p in line for p in patterns):
                lines.append(line.strip())
    except FileNotFoundError:
        pass
if lines:
    print(lines[-1].replace(",", ";"))
else:
    print("")
PY
)"

  if [[ ${RC} -eq 0 ]]; then
    STATUS="success"
  else
    STATUS="failed"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${EXP_NAME}" "${MODEL}" "${SEQ_LEN}" "${PP}" "${TP}" "${DP}" "${VPP}" "${REPEAT_ID}" "${RC}" "${STATUS}" "${REASON}" \
    >> "${SUMMARY_CSV}"
  ran=$((ran + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${ran}" -ge "${MAX_RUNS}" ]]; then
    break
  fi
done

echo "${SUMMARY_CSV}"
