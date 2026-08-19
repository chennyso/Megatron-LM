#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
ITERS="${2:-10}"
NUM_EXPERTS="${NUM_EXPERTS_OVERRIDE:-128}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE_OVERRIDE:-768}"
SEQ_LENGTH="${SEQ_LENGTH_OVERRIDE:-1024}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE_OVERRIDE:-16}"
TRACE_OVERRIDE="${TRACE_OVERRIDE:-true}"
PROFILE_STEPS="${STRATEGY_PROFILE_STEPS:-4}"
MOE_OVERLAP_OVERRIDE="${MOE_OVERLAP_OVERRIDE:-false}"
MOE_DISPATCHER="${MOE_DISPATCHER_OVERRIDE:-alltoall}"
P2P_OVERLAP_OVERRIDE="${P2P_OVERLAP_OVERRIDE:-true}"
VPP_GROUP_SIZE_OVERRIDE="${VPP_GROUP_SIZE_OVERRIDE:-}"
NSYS_TAG="${NSYS_TAG:-}"
NSYS_NODES="${NSYS_NODES:-g5}"
NSYS_PROFILE_START_STEP="${NSYS_PROFILE_START_STEP:-4}"
NSYS_PROFILE_END_STEP="${NSYS_PROFILE_END_STEP:-8}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-6}"
NS=default
G5="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g5 -o jsonpath='{.items[0].metadata.name}')"
G6="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g6 -o jsonpath='{.items[0].metadata.name}')"
MASTER="$(kubectl get pod -n "$NS" "$G5" -o jsonpath='{.status.podIP}')"
RUN_DIR="/workspace/runs/phaseweaver-20260818/results/qwen3-30b-a3b-${RUN_ID}"
CODE=/workspace/runs/phaseweaver-20260818/code
PY=/workspace/runs/transitpipe-20260817-232632/venv/bin/python
PORT="${PORT_OVERRIDE:-29810}"

kubectl exec -n "$NS" "$G5" -- mkdir -p "$RUN_DIR/traces"
kubectl exec -n "$NS" "$G6" -- mkdir -p "$RUN_DIR/traces"
if [[ "$TRACE_OVERRIDE" == "true" ]]; then
  STRATEGY_TRACE_ARGS="--pipeline-strategy-profile-steps '$PROFILE_STEPS' --pipeline-strategy-trace-path '$RUN_DIR/traces/rank{rank}.json'"
elif [[ "$TRACE_OVERRIDE" == "false" ]]; then
  STRATEGY_TRACE_ARGS=""
else
  echo "TRACE_OVERRIDE must be true or false, got $TRACE_OVERRIDE" >&2
  exit 2
fi
if [[ "$MOE_OVERLAP_OVERRIDE" == "true" ]]; then
  MOE_OVERLAP_ARGS="--overlap-moe-expert-parallel-comm"
elif [[ "$MOE_OVERLAP_OVERRIDE" == "false" ]]; then
  MOE_OVERLAP_ARGS=""
else
  echo "MOE_OVERLAP_OVERRIDE must be true or false, got $MOE_OVERLAP_OVERRIDE" >&2
  exit 2
fi
if [[ "$P2P_OVERLAP_OVERRIDE" == "true" ]]; then
  P2P_ARGS=""
elif [[ "$P2P_OVERLAP_OVERRIDE" == "false" ]]; then
  P2P_ARGS="--no-overlap-p2p-communication"
else
  echo "P2P_OVERLAP_OVERRIDE must be true or false, got $P2P_OVERLAP_OVERRIDE" >&2
  exit 2
fi
if [[ -n "$VPP_GROUP_SIZE_OVERRIDE" ]]; then
  VPP_GROUP_ARGS="--microbatch-group-size-per-virtual-pipeline-stage '$VPP_GROUP_SIZE_OVERRIDE'"
else
  VPP_GROUP_ARGS=""
fi
if [[ -n "$NSYS_TAG" ]]; then
  PROFILE_ARGS="--profile --profile-step-start '$NSYS_PROFILE_START_STEP' --profile-step-end '$NSYS_PROFILE_END_STEP' --profile-ranks '$NSYS_PROFILE_RANKS' --nvtx-ranges"
else
  PROFILE_ARGS=""
fi
launch() {
  local pod="$1" node_rank="$2"
  local node_name="g5"
  if [[ "$node_rank" == "1" ]]; then
    node_name="g6"
  fi
  local nsys_prefix=""
  if [[ -n "$NSYS_TAG" && ( "$NSYS_NODES" == "both" || "$NSYS_NODES" == "$node_name" ) ]]; then
    nsys_prefix="nsys profile --trace=cuda,nvtx,osrt --sample=none --trace-fork-before-exec=true --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --output '$RUN_DIR/nsys-node$node_rank-$NSYS_TAG'"
  fi
  kubectl exec -n "$NS" "$pod" -- bash -lc "
    set -euo pipefail; cd '$CODE'
    export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5 NCCL_IB_GID_INDEX=0 NCCL_IB_TC=136 NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_TIMEOUT=22 NCCL_MIN_NCHANNELS=4 NCCL_SOCKET_IFNAME=net1 UCX_NET_DEVICES=net1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_HCA=\$(ls /sys/class/net/net1/device/infiniband/ | head -1)
    unset LD_PRELOAD NCCL_HOOK_ENABLE NCCL_HOOK_K8S_MODE NCCL_HOOK_COORDINATOR_ADDR NCCL_HOOK_POD_UID NCCL_HOOK_STATE_DIR NCCL_HOOK_LOG_DIR NCCL_HOOK_LOG_LEVEL G10_METRICS_PORT MIRAGE_DAEMON_SOCKET
    $nsys_prefix '$PY' -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank='$node_rank' --master_addr='$MASTER' --master_port='$PORT' --rdzv_backend=static pretrain_gpt.py \\
      --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --expert-model-parallel-size 2 \\
      --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 \\
      --num-experts '$NUM_EXPERTS' --moe-ffn-hidden-size '$MOE_FFN_HIDDEN_SIZE' --moe-router-topk 8 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.001 --moe-token-dispatcher-type '$MOE_DISPATCHER' \\
      --num-layers-per-virtual-pipeline-stage 4 --seq-length '$SEQ_LENGTH' --max-position-embeddings 40960 --micro-batch-size 1 --global-batch-size '$GLOBAL_BATCH_SIZE' \\
      --bf16 --use-mcore-models --position-embedding-type rope --rotary-percent 1.0 --rotary-base 1000000 --tokenizer-type HuggingFaceTokenizer --tokenizer-model /models/qwen3-30B-A3B \\
      --normalization RMSNorm --swiglu --disable-bias-linear --untie-embeddings-and-output-weights --sequence-parallel --use-distributed-optimizer --mock-data $MOE_OVERLAP_ARGS \\
      --train-iters '$ITERS' --eval-iters 1 --eval-interval 1000 --log-interval 1 --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant \\
      --pipeline-strategy-policy default --pipeline-strategy-runtime fixed $STRATEGY_TRACE_ARGS $PROFILE_ARGS $P2P_ARGS $VPP_GROUP_ARGS \\
      > '$RUN_DIR/node.g$node_rank.log' 2>&1
  " &
}
launch "$G5" 0
sleep 3
launch "$G6" 1
wait
echo "completed $RUN_DIR"
