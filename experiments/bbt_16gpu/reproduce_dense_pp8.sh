#!/usr/bin/env bash
set -euo pipefail

# Single-node reproduction harness for the native Megatron PP baselines.
# It intentionally keeps the model, optimizer, data and world fixed while
# changing only VPP, so the result is a usable baseline for paper replication.
MODE="${1:?usage: $0 <vpp1|vpp2> <run_id> [iters]}"
RUN_ID="${2:?usage: $0 <vpp1|vpp2> <run_id> [iters]}"
ITERS="${3:-12}"
[[ "$MODE" == vpp1 || "$MODE" == vpp2 ]] || { echo "mode must be vpp1 or vpp2" >&2; exit 2; }

CODE="${CODE_OVERRIDE:-/workspace/runs/phaseweaver-20260818/code}"
OUT="${OUT_OVERRIDE:-/workspace/runs/phaseweaver-20260818/results/repro-dense-pp8-${MODE}-${RUN_ID}}"
PY="${PY_OVERRIDE:-python}"
VPP_ARGS=""
if [[ "$MODE" == vpp2 ]]; then
  VPP_ARGS="--num-layers-per-virtual-pipeline-stage 2"
fi

mkdir -p "$OUT"
cd "$CODE"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LD_PRELOAD NCCL_HOOK_ENABLE NCCL_HOOK_K8S_MODE NCCL_HOOK_COORDINATOR_ADDR \
  NCCL_HOOK_POD_UID NCCL_HOOK_STATE_DIR NCCL_HOOK_LOG_DIR NCCL_HOOK_LOG_LEVEL \
  G10_METRICS_PORT MIRAGE_DAEMON_SOCKET

"$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --master_port="${MASTER_PORT:-29831}" \
  pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 4 \
  --num-layers 16 --hidden-size 1024 --ffn-hidden-size 4096 \
  --num-attention-heads 16 --group-query-attention --num-query-groups 8 \
  --seq-length 1024 --max-position-embeddings 1024 \
  --micro-batch-size 1 --global-batch-size 8 --bf16 \
  --use-mcore-models --position-embedding-type rope --rotary-percent 1.0 \
  --rotary-base 1000000 --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /models/qwen3-8B --no-masked-softmax-fusion \
  --normalization RMSNorm --swiglu --disable-bias-linear \
  --untie-embeddings-and-output-weights --sequence-parallel \
  --use-distributed-optimizer --mock-data --train-iters "$ITERS" \
  --eval-iters 1 --eval-interval 1000 --log-interval 1 \
  --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant \
  $VPP_ARGS --pipeline-strategy-policy default \
  --pipeline-strategy-runtime fixed \
  > "$OUT/train.log" 2>&1

echo "$OUT"
