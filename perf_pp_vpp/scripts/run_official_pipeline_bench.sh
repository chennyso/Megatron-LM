#!/usr/bin/env bash
set -euo pipefail

SCHEDULE="${1:?schedule required}"
NUM_LAYERS="${NUM_LAYERS:-16}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
SEQ_LEN="${SEQ_LEN:-4096}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MICROBATCHES="${MICROBATCHES:-16}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-net1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_IB_TC="${NCCL_IB_TC:-136}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_HCA="$(ls /sys/class/net/${NCCL_SOCKET_IFNAME}/device/infiniband/ | head -1)"

exec torchrun \
  --nnodes="${NNODES:-2}" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --node_rank="${NODE_RANK:?}" \
  --master_addr="${MASTER_ADDR:?}" \
  --master_port="${MASTER_PORT:?}" \
  /workspace/code/Megatron-LM/perf_pp_vpp/scripts/official_pipeline_bench.py \
  --schedule "${SCHEDULE}" \
  --num-layers "${NUM_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --microbatches "${MICROBATCHES}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}"
