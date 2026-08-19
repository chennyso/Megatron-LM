#!/usr/bin/env bash
set -euo pipefail

TP="${1:?usage: $0 <tp> <pp> <vpp> <run_id> [iters] [plan_path] [group_size]}"
PP="${2:?usage: $0 <tp> <pp> <vpp> <run_id> [iters] [plan_path] [group_size]}"
VPP="${3:?usage: $0 <tp> <pp> <vpp> <run_id> [iters] [plan_path] [group_size]}"
RUN_ID="${4:?usage: $0 <tp> <pp> <vpp> <run_id> [iters] [plan_path] [group_size]}"
ITERS="${5:-10}"
PLAN_PATH="${6:-}"
GROUP_SIZE="${7:-}"
DDP_MODE="${8:-default}"
LAYOUT="${9:-}"
NSYS_TAG="${10:-}"
CP="${CP_OVERRIDE:-1}"
CP_COMM_TYPE="${CP_COMM_TYPE_OVERRIDE:-}"
HCP_SIZES="${HIERARCHICAL_CP_SIZES_OVERRIDE:-}"
WORLD=16
if (( WORLD % TP != 0 || WORLD % PP != 0 || WORLD % CP != 0 || WORLD % (TP * PP * CP) != 0 )); then
  echo "invalid factorization TP=$TP PP=$PP CP=$CP for world=$WORLD" >&2
  exit 2
fi
DP=$((WORLD / TP / PP / CP))
LAYERS="${LAYERS_OVERRIDE:-36}"
if (( VPP <= 0 || LAYERS % (PP * VPP) != 0 )); then
  echo "illegal VPP=$VPP for layers=$LAYERS and PP=$PP" >&2
  exit 2
fi
if [[ -n "$LAYOUT" ]]; then
  VPP_ARG="--pipeline-model-parallel-layout '$LAYOUT'"
elif (( VPP > 1 )); then
  VPP_ARG="--num-layers-per-virtual-pipeline-stage $((LAYERS / PP / VPP))"
else
  VPP_ARG=""
fi
if [[ -n "$PLAN_PATH" ]]; then
  STRATEGY_PLAN_ARG="--pipeline-strategy-plan '$PLAN_PATH'"
else
  STRATEGY_PLAN_ARG=""
fi
if [[ -n "$GROUP_SIZE" ]]; then
  GROUP_SIZE_ARG="--microbatch-group-size-per-virtual-pipeline-stage '$GROUP_SIZE'"
else
  GROUP_SIZE_ARG=""
fi
case "$DDP_MODE" in
  default) DDP_ARGS="" ;;
  overlap4) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 4" ;;
  overlap8) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 8" ;;
  overlap4_p2p) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 4 --overlap-p2p-communication" ;;
  overlap8_p2p) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 8 --overlap-p2p-communication" ;;
  overlap4_p2p_adaptive) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 4 --overlap-p2p-communication --pipeline-strategy-adaptive-vpp-group" ;;
  allchunks4) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy all-chunks" ;;
  allchunks8) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 8 --vpp-bucket-policy all-chunks" ;;
  rank0chunks4) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy rank0-all-chunks" ;;
  rank0chunks8) DDP_ARGS="--overlap-grad-reduce --ddp-num-buckets 8 --vpp-bucket-policy rank0-all-chunks" ;;
  *) echo "unsupported ddp_mode=$DDP_MODE (expected default|overlap4|overlap8|overlap4_p2p|overlap8_p2p|overlap4_p2p_adaptive|rank0chunks4|rank0chunks8|allchunks4|allchunks8)" >&2; exit 2 ;;
esac

NS=default
G5="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g5 -o jsonpath='{.items[0].metadata.name}')"
G6="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g6 -o jsonpath='{.items[0].metadata.name}')"
MASTER="$(kubectl get pod -n "$NS" "$G5" -o jsonpath='{.status.podIP}')"
RUN_DIR="/workspace/runs/phaseweaver-20260818/results/strategy-tp${TP}-pp${PP}-vpp${VPP}-dp${DP}-${RUN_ID}"
# Use a pinned Git checkout for reproducible traces while retaining historical snapshots.
CODE="${CODE_OVERRIDE:-/workspace/runs/phaseweaver-20260818/code}"
PY=/workspace/runs/transitpipe-20260817-232632/venv/bin/python
PORT="${PORT_OVERRIDE:-$((29600 + TP * 100 + PP * 10 + VPP))}"

kubectl exec -n "$NS" "$G5" -- mkdir -p "$RUN_DIR/traces"
kubectl exec -n "$NS" "$G6" -- mkdir -p "$RUN_DIR/traces"

if (( TP > 1 )); then
  SP_ARG="--sequence-parallel"
else
  SP_ARG=""
fi
if [[ -n "$CP_COMM_TYPE" ]]; then
  CP_COMM_ARG="--cp-comm-type $CP_COMM_TYPE"
else
  CP_COMM_ARG=""
fi
if [[ -n "$HCP_SIZES" ]]; then
  HCP_ARG="--hierarchical-context-parallel-sizes $HCP_SIZES"
else
  HCP_ARG=""
fi

launch() {
  local pod="$1" rank="$2" log="$3"
  local nsys_prefix=""
  if [[ -n "$NSYS_TAG" ]]; then
    nsys_prefix="nsys profile --trace=cuda,nvtx,osrt --sample=none --trace-fork-before-exec=true --force-overwrite=true --duration=18 --output '$RUN_DIR/nsys-node${rank}-${NSYS_TAG}'"
  fi
  kubectl exec -n "$NS" "$pod" -- bash -lc "
    set -euo pipefail
    cd '$CODE'
    export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5
    export NCCL_IB_GID_INDEX=0 NCCL_IB_TC=136 NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TIMEOUT=22 NCCL_MIN_NCHANNELS=4 NCCL_SOCKET_IFNAME=net1
    export UCX_NET_DEVICES=net1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_HCA=\$(ls /sys/class/net/net1/device/infiniband/ | head -1)
    unset LD_PRELOAD NCCL_HOOK_ENABLE NCCL_HOOK_K8S_MODE NCCL_HOOK_COORDINATOR_ADDR NCCL_HOOK_POD_UID NCCL_HOOK_STATE_DIR NCCL_HOOK_LOG_DIR NCCL_HOOK_LOG_LEVEL G10_METRICS_PORT MIRAGE_DAEMON_SOCKET
    $nsys_prefix '$PY' -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=$rank \
      --master_addr='$MASTER' --master_port='$PORT' --rdzv_backend=static \
      pretrain_gpt.py \
      --tensor-model-parallel-size '$TP' \
      --pipeline-model-parallel-size '$PP' \
      --context-parallel-size '$CP' \
      $CP_COMM_ARG \
      $HCP_ARG \
      --num-layers '$LAYERS' --hidden-size 4096 --ffn-hidden-size 12288 \
      --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
      --seq-length 4096 --max-position-embeddings 40960 \
      --micro-batch-size 1 --global-batch-size 16 --bf16 \
      --use-mcore-models --position-embedding-type rope --rotary-percent 1.0 \
      --rotary-base 1000000 --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model /models/qwen3-8B --no-masked-softmax-fusion \
      --normalization RMSNorm --swiglu --disable-bias-linear \
      --untie-embeddings-and-output-weights $SP_ARG \
      --use-distributed-optimizer --mock-data --train-iters '$ITERS' \
      --eval-iters 1 --eval-interval 1000 --log-interval 1 \
      --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant \
      $VPP_ARG \
      --pipeline-strategy-policy default --pipeline-strategy-runtime fixed \
      $STRATEGY_PLAN_ARG \
      $GROUP_SIZE_ARG \
      $DDP_ARGS \
      --pipeline-strategy-profile-steps 4 \
      --pipeline-strategy-trace-path '$RUN_DIR/traces/rank{rank}.json' \
      > '$log' 2>&1
  " &
}

launch "$G5" 0 "$RUN_DIR/node.g5.log"
sleep 3
launch "$G6" 1 "$RUN_DIR/node.g6.log"
wait
echo "completed TP=$TP PP=$PP VPP=$VPP DP=$DP ddp_mode=$DDP_MODE at $RUN_DIR"
