#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

MANIFEST="${OUTPUT_ROOT}/data_manifest.json"
mkdir -p "${DATA_ROOT}"

if ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${NODE0_POD}" -- bash -lc "cd ${REMOTE_PROJECT_ROOT} && grep -q -- --mock-data megatron/training/arguments.py"; then
  cat > "${MANIFEST}" <<EOF
{
  "mode": "mock_data",
  "MOCK_DATA_MODE": "mock_data",
  "seq_len": 8192,
  "steps": 200,
  "seed": ${PERF_SEED}
}
EOF
  echo "MOCK_DATA_MODE=mock_data"
  exit 0
fi

JSONL="${DATA_ROOT}/synthetic_tokens.jsonl"
export JSONL
${PYTHON_BIN} - <<'PY'
import json
import os
import random
from pathlib import Path

out = Path(os.environ["JSONL"])
seq_len = 8192
steps = 200
batch = 16
vocab = 151936
rng = random.Random(int(os.environ.get("PERF_SEED", "1234")))
with out.open("w", encoding="utf-8") as fh:
    for _ in range(steps * batch):
        sample = {"text": " ".join(str(rng.randrange(vocab)) for _ in range(seq_len))}
        fh.write(json.dumps(sample) + "\n")
PY

PREPROCESS="${REMOTE_PROJECT_ROOT}/tools/preprocess_data.py"
REMOTE_JSONL="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/data/synthetic_tokens.jsonl"
REMOTE_PREFIX="${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/data/synthetic_tokens"
${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${NODE0_POD}" -- bash -lc "mkdir -p ${REMOTE_PROJECT_ROOT}/perf_pp_vpp/outputs/data"
${KUBECTL_BIN} cp "${JSONL}" "${K8S_NAMESPACE}/${NODE0_POD}:${REMOTE_JSONL}"
${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${NODE0_POD}" -- bash -lc "cd ${REMOTE_PROJECT_ROOT} && ${PYTHON_BIN} ${PREPROCESS} --input ${REMOTE_JSONL} --output-prefix ${REMOTE_PREFIX} --dataset-impl mmap --tokenizer-type NullTokenizer --workers 1"
cat > "${MANIFEST}" <<EOF
{
  "mode": "synthetic_indexed_dataset",
  "MOCK_DATA_MODE": "synthetic_indexed_dataset",
  "dataset_prefix": "${REMOTE_PREFIX}",
  "seq_len": 8192,
  "steps": 200,
  "seed": ${PERF_SEED}
}
EOF
echo "MOCK_DATA_MODE=synthetic_indexed_dataset"
