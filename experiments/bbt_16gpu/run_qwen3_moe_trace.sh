#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
ITERS="${2:-10}"
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
launch() {
  local pod="$1" node_rank="$2"
  kubectl exec -n "$NS" "$pod" -- bash -lc "
    set -euo pipefail; cd '$CODE'
    export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=5 NCCL_IB_GID_INDEX=0 NCCL_IB_TC=136 NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_TIMEOUT=22 NCCL_MIN_NCHANNELS=4 NCCL_SOCKET_IFNAME=net1 UCX_NET_DEVICES=net1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_HCA=\$(ls /sys/class/net/net1/device/infiniband/ | head -1)
    unset LD_PRELOAD NCCL_HOOK_ENABLE NCCL_HOOK_K8S_MODE NCCL_HOOK_COORDINATOR_ADDR NCCL_HOOK_POD_UID NCCL_HOOK_STATE_DIR NCCL_HOOK_LOG_DIR NCCL_HOOK_LOG_LEVEL G10_METRICS_PORT MIRAGE_DAEMON_SOCKET
    '$PY' -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank='$node_rank' --master_addr='$MASTER' --master_port='$PORT' --rdzv_backend=static pretrain_gpt.py \\
      --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --expert-model-parallel-size 2 \\
      --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 \\
      --num-experts 128 --moe-ffn-hidden-size 768 --moe-router-topk 8 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.001 \\
      --num-layers-per-virtual-pipeline-stage 4 --seq-length 1024 --max-position-embeddings 40960 --micro-batch-size 1 --global-batch-size 16 \\
      --bf16 --use-mcore-models --position-embedding-type rope --rotary-percent 1.0 --rotary-base 1000000 --tokenizer-type HuggingFaceTokenizer --tokenizer-model /models/qwen3-30B-A3B \\
      --normalization RMSNorm --swiglu --disable-bias-linear --untie-embeddings-and-output-weights --sequence-parallel --use-distributed-optimizer --mock-data \\
      --train-iters '$ITERS' --eval-iters 1 --eval-interval 1000 --log-interval 1 --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant \\
      --pipeline-strategy-policy default --pipeline-strategy-runtime fixed --pipeline-strategy-profile-steps 4 --pipeline-strategy-trace-path '$RUN_DIR/traces/rank{rank}.json' \\
      > '$RUN_DIR/node.g$node_rank.log' 2>&1
  " &
}
launch "$G5" 0
sleep 3
launch "$G6" 1
wait
echo "completed $RUN_DIR"
