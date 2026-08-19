#!/usr/bin/env bash
set -euo pipefail

# Execute a verifier-native flexible-VPP layout for an alternating GDN/attention
# model.  The pattern's `|` boundaries, rather than a synthetic cost model,
# define both the PP and virtual-PP ownership.
PATTERN="${1:?usage: $0 <hybrid-layer-pattern> <run-id> [iters]}"
RUN_ID="${2:?usage: $0 <hybrid-layer-pattern> <run-id> [iters]}"
ITERS="${3:-12}"
NS=default
G5="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g5 -o jsonpath='{.items[0].metadata.name}')"
G6="$(kubectl get pod -n "$NS" -l app.kubernetes.io/name=chenny-g5-g6-16gpu-reservation,bbt.sspu.edu.cn/reservation-node=g6 -o jsonpath='{.items[0].metadata.name}')"
MASTER="$(kubectl get pod -n "$NS" "$G5" -o jsonpath='{.status.podIP}')"
RUN_DIR="/workspace/runs/phaseweaver-20260818/results/hybrid-fvpp-${RUN_ID}"
CODE="${CODE_OVERRIDE:-/workspace/runs/phaseweaver-20260818/code}"
PY=/workspace/runs/transitpipe-20260817-232632/venv/bin/python
PORT="${PORT_OVERRIDE:-29780}"

kubectl exec -n "$NS" "$G5" -- mkdir -p "$RUN_DIR/traces"
kubectl exec -n "$NS" "$G6" -- mkdir -p "$RUN_DIR/traces"

launch() {
  local pod="$1" node_rank="$2"
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
    '$PY' -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank='$node_rank' \\
      --master_addr='$MASTER' --master_port='$PORT' --rdzv_backend=static pretrain_hybrid.py \\
      --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 \\
      --hybrid-layer-pattern '$PATTERN' \\
      --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \\
      --hidden-size 1024 --ffn-hidden-size 4096 --num-attention-heads 16 \\
      --group-query-attention --num-query-groups 8 \\
      --linear-attention-freq 4 --linear-conv-kernel-dim 4 \\
      --linear-key-head-dim 64 --linear-value-head-dim 64 \\
      --linear-num-key-heads 8 --linear-num-value-heads 16 \\
      --attention-output-gate --seq-length 2048 --max-position-embeddings 2048 \\
      --micro-batch-size 1 --global-batch-size 16 --bf16 --use-mcore-models \\
      --position-embedding-type rope --rotary-percent 1.0 --rotary-base 1000000 \\
      --tokenizer-type HuggingFaceTokenizer --tokenizer-model /models/qwen3-8B \\
      --normalization RMSNorm --swiglu --disable-bias-linear --untie-embeddings-and-output-weights \\
      --sequence-parallel --use-distributed-optimizer --mock-data --train-iters '$ITERS' \\
      --eval-iters 1 --eval-interval 1000 --log-interval 1 --lr 1e-6 --min-lr 1e-7 \\
      --lr-decay-style constant --transformer-impl transformer_engine --attention-backend unfused \\
      --pipeline-strategy-policy default --pipeline-strategy-runtime fixed \\
      --pipeline-strategy-profile-steps 4 \\
      --pipeline-strategy-trace-path '$RUN_DIR/traces/rank{rank}.json' \\
      > '$RUN_DIR/node.g$node_rank.log' 2>&1
  " &
}

launch "$G5" 0
sleep 3
launch "$G6" 1
wait
echo "completed $RUN_DIR"
