#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <candidate-json> <experiment-yaml> <experiment-name> <seq-len> <repeat-id> [overlap-mode]" >&2
  exit 2
fi

CANDIDATE_JSON="$1"
EXPERIMENT_YAML="$2"
EXPERIMENT_NAME="$3"
SEQ_LEN="$4"
REPEAT_ID="$5"
OVERLAP_MODE="${6:-baseline}"

mapfile -t CANDIDATE_ENV < <("${PYTHON_BIN}" perf_pp_vpp/scripts/14_generate_covpipe_candidate.py --candidate-json "${CANDIDATE_JSON}" --print-env)

for line in "${CANDIDATE_ENV[@]}"; do
  export "${line}"
done

exec perf_pp_vpp/scripts/03_run_megatron_experiment.sh \
  --experiment-yaml "${EXPERIMENT_YAML}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --seq-len "${SEQ_LEN}" \
  --repeat-id "${REPEAT_ID}" \
  --overlap-mode "${OVERLAP_MODE}" \
  --profile-mode none \
  --profile-ranks none
