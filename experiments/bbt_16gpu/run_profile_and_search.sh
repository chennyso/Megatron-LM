#!/usr/bin/env bash
set -euo pipefail

MODE=""
MODEL=""
BRANCH=""
TRACE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" || -z "$MODEL" || -z "$BRANCH" ]]; then
  echo "usage: $0 --mode <profile|search> --model <qwen3-0.6b|qwen3-8b> --branch <git-branch> [--trace-dir DIR]" >&2
  exit 1
fi

case "$MODEL" in
  qwen3-0.6b)
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    FFN_HIDDEN_SIZE=3072
    NUM_HEADS=16
    NUM_QUERY_GROUPS=8
    PP=4
    VP=4
    MICRO_BATCH=2
    GLOBAL_BATCH=32
    SEQ_LEN=4096
    HF_CKPT="/workspace/models/qwen3-0.6B"
    ;;
  qwen3-8b)
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=12288
    NUM_HEADS=32
    NUM_QUERY_GROUPS=8
    PP=8
    VP=2
    MICRO_BATCH=1
    GLOBAL_BATCH=16
    SEQ_LEN=4096
    HF_CKPT="/workspace/models/qwen3-8B"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 1
    ;;
esac

RUN_ROOT="/workspace/runs/seampipe/${MODEL}"
PROFILE_ID="$(date +%Y%m%d-%H%M%S)"
PROFILE_DIR="${RUN_ROOT}/profile-${PROFILE_ID}"
TRACE_DIR_DEFAULT="${PROFILE_DIR}/traces"
TRACE_DIR="${TRACE_DIR:-$TRACE_DIR_DEFAULT}"
PLAN_PATH="${RUN_ROOT}/plans/${PROFILE_ID}-best-plan.json"
REPORT_PATH="${RUN_ROOT}/plans/${PROFILE_ID}-candidate-report.json"

COMMON_ARGS="\
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size ${PP} \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_QUERY_GROUPS} \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings 40960 \
  --micro-batch-size ${MICRO_BATCH} \
  --global-batch-size ${GLOBAL_BATCH} \
  --bf16 \
  --use-mcore-models \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model ${HF_CKPT} \
  --no-masked-softmax-fusion \
  --normalization RMSNorm \
  --swiglu \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --mock-data \
  --train-iters 20"

if [[ "$MODE" == "profile" ]]; then
  cat <<EOF
PROFILE_MODE=1
GIT_BRANCH=${BRANCH}
RUN_DIR=${PROFILE_DIR}
TRACE_DIR=${TRACE_DIR}
PLAN_PATH=${PLAN_PATH}
MEGATRON_ARGS=${COMMON_ARGS} \\
  --pipeline-model-parallel-layout auto \\
  --num-layers-per-virtual-pipeline-stage $(( NUM_LAYERS / PP / VP )) \\
  --pipeline-strategy-policy default \\
  --pipeline-strategy-runtime fixed \\
  --pipeline-strategy-profile-steps 8 \\
  --pipeline-strategy-trace-path ${TRACE_DIR}/rank{pp_rank}.json
EOF
  exit 0
fi

if [[ "$MODE" == "search" ]]; then
  if [[ -z "$TRACE_DIR" ]]; then
    echo "--trace-dir is required for search mode" >&2
    exit 1
  fi
  cat <<EOF
python3 tools/run_bcp_vpp_loop.py \\
  --trace ${TRACE_DIR}/rank*.json \\
  --output-dir ${RUN_ROOT}/search-${PROFILE_ID} \\
  --num-microbatches ${GLOBAL_BATCH} \\
  --num-model-chunks ${VP} \\
  --microbatch-group-size 4 \\
  --pipeline-parallel-size ${PP} \\
  --num-layers ${NUM_LAYERS} \\
  --runtime bcp-ready \\
  --objective bcp \\
  --candidate-budget 24 \\
  --bcp-activation-budget-mb 28000 \\
  --bcp-p2p-credit-budget 2 \\
  --bcp-fb-delay-budget 24
EOF
  exit 0
fi

echo "unsupported mode: $MODE" >&2
exit 1
