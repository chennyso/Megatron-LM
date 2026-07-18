#!/usr/bin/env bash
set -euo pipefail

required_env=(
  CASE_ID RUN_ID REPEAT_ID NODE_RANK MASTER_ADDR MASTER_PORT
  GIT_REMOTE GIT_REF WORKSPACE_ROOT MODEL_PATH DATA_PATH
  VPP_SIZE OVERLAP_P2P WARMUP_FLUSH_OVERLAP PROFILE_MODE
  WARMUP_STEPS MEASURE_STEPS
)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done

RUN_DIR="${WORKSPACE_ROOT}/${RUN_ID}/${CASE_ID}/repeat_$(printf '%02d' "${REPEAT_ID}")"
NODE_DIR="${RUN_DIR}/node_${NODE_RANK}"
CODE_DIR="/opt/observation-code"
mkdir -p "${NODE_DIR}"
exec > >(tee -a "${NODE_DIR}/launcher.log") 2>&1

on_exit() {
  status=$?
  if [[ -n "${DMON_PID:-}" ]]; then
    kill "${DMON_PID}" 2>/dev/null || true
    wait "${DMON_PID}" 2>/dev/null || true
  fi
  printf '{"exit_code":%d,"finished_at":"%s"}\n' \
    "${status}" "$(date -Iseconds)" > "${NODE_DIR}/exit_status.json"
  exit "${status}"
}
trap on_exit EXIT

echo "OBSERVATION_START case=${CASE_ID} repeat=${REPEAT_ID} node_rank=${NODE_RANK}"
echo "host=$(hostname) date=$(date -Iseconds)"

export GIT_SSL_NO_VERIFY=1
rm -rf "${CODE_DIR}"
git clone --filter=blob:none --no-checkout "${GIT_REMOTE}" "${CODE_DIR}"
git -C "${CODE_DIR}" fetch --depth=1 origin "${GIT_REF}"
git -C "${CODE_DIR}" checkout --detach FETCH_HEAD
GIT_COMMIT="$(git -C "${CODE_DIR}" rev-parse HEAD)"

if ! python3 -c 'import sentencepiece, transformers' >/dev/null 2>&1; then
  python3 -m pip install --quiet --no-cache-dir \
    'transformers==4.51.0' \
    'sentencepiece==0.2.0'
fi

test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/tokenizer.json"
test -f "${DATA_PATH}.bin"
test -f "${DATA_PATH}.idx"

IB_IF=""
for dev in net1 net2 net3; do
  if [[ -d "/sys/class/net/${dev}" ]]; then
    IB_IF="${dev}"
    break
  fi
done
if [[ -z "${IB_IF}" ]]; then
  echo "no SR-IOV IB interface found" >&2
  exit 3
fi
RDMA_DEV="$(ls "/sys/class/net/${IB_IF}/device/infiniband" | head -n 1)"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA="${RDMA_DEV}"
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TC=136
export NCCL_SOCKET_IFNAME="${IB_IF}"
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=22
export NCCL_MIN_NCHANNELS=4
export NCCL_DEBUG="${NCCL_DEBUG_LEVEL:-WARN}"
export UCX_NET_DEVICES="${IB_IF}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,power.limit \
  --format=csv > "${NODE_DIR}/gpu_inventory.csv"
nvidia-smi topo -m > "${NODE_DIR}/gpu_topology.txt"
nsys --version > "${NODE_DIR}/nsys_version.txt" 2>&1 || true
python3 - <<PY
import json
import os
import platform
import subprocess

payload = {
    "case_id": os.environ["CASE_ID"],
    "run_id": os.environ["RUN_ID"],
    "repeat_id": int(os.environ["REPEAT_ID"]),
    "node_rank": int(os.environ["NODE_RANK"]),
    "hostname": platform.node(),
    "git_commit": "${GIT_COMMIT}",
    "git_ref": os.environ["GIT_REF"],
    "profile_mode": os.environ["PROFILE_MODE"],
    "vpp_size": int(os.environ["VPP_SIZE"]),
    "overlap_p2p": os.environ["OVERLAP_P2P"] == "1",
    "warmup_flush_overlap": os.environ["WARMUP_FLUSH_OVERLAP"] == "1",
    "microbatch_group_size": int(os.environ.get("MICROBATCH_GROUP_SIZE", "0")) or None,
    "warmup_steps": int(os.environ["WARMUP_STEPS"]),
    "measure_steps": int(os.environ["MEASURE_STEPS"]),
    "model_path": os.environ["MODEL_PATH"],
    "data_path": os.environ["DATA_PATH"],
    "ib_interface": "${IB_IF}",
    "rdma_device": "${RDMA_DEV}",
    "python": platform.python_version(),
    "torch": __import__("torch").__version__,
    "transformers": __import__("transformers").__version__,
    "sentencepiece": __import__("sentencepiece").__version__,
    "cuda_runtime": __import__("torch").version.cuda,
    "command_line": " ".join(subprocess.list2cmdline([x]) for x in os.sys.argv),
}
with open("${NODE_DIR}/run_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY

nvidia-smi dmon -s pucvmet -d 1 -o DT > "${NODE_DIR}/nvidia_smi_dmon.log" 2>&1 &
DMON_PID=$!

TOTAL_STEPS=$((WARMUP_STEPS + MEASURE_STEPS))
COMMON_ARGS=(
  --use-mcore-models
  --attention-backend flash
  --num-layers 64
  --hidden-size 5120
  --ffn-hidden-size 25600
  --num-attention-heads 64
  --kv-channels 80
  --group-query-attention
  --num-query-groups 8
  --seq-length 4096
  --max-position-embeddings 40960
  --micro-batch-size 1
  --global-batch-size 16
  --train-iters "${TOTAL_STEPS}"
  --lr 1.0e-4
  --min-lr 1.0e-4
  --lr-decay-style constant
  --lr-warmup-iters 0
  --seed 1234
  --bf16
  --transformer-impl transformer_engine
  --empty-unused-memory-level 1
  --use-distributed-optimizer
  --use-precision-aware-optimizer
  --main-grads-dtype bf16
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --main-params-dtype fp32
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --no-gradient-accumulation-fusion
  --no-rope-fusion
  --no-persist-layer-norm
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --position-embedding-type rope
  --rotary-base 1000000
  --rotary-percent 1.0
  --qk-layernorm
  --swiglu
  --disable-bias-linear
  --untie-embeddings-and-output-weights
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --make-vocab-size-divisible-by 128
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --split 99,1,0
  --data-cache-path "${WORKSPACE_ROOT}/dataset_cache/qwen3_32b_s4096"
  --no-create-attention-mask-in-dataloader
  --num-workers 2
  --log-interval 1
  --eval-iters 0
  --eval-interval 1000
  --timing-log-level 2
  --log-throughput
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 8
)

SCHEDULE_ARGS=()
if [[ "${VPP_SIZE}" -eq 1 ]]; then
  SCHEDULE_ARGS+=(--no-overlap-p2p-communication)
else
  SCHEDULE_ARGS+=(--num-virtual-stages-per-pipeline-rank "${VPP_SIZE}")
  SCHEDULE_ARGS+=(--microbatch-group-size-per-virtual-pipeline-stage "${MICROBATCH_GROUP_SIZE}")
  if [[ "${OVERLAP_P2P}" != "1" ]]; then
    SCHEDULE_ARGS+=(--no-overlap-p2p-communication)
  fi
  if [[ "${WARMUP_FLUSH_OVERLAP}" == "1" ]]; then
    SCHEDULE_ARGS+=(--overlap-p2p-communication-warmup-flush)
  fi
fi

PROFILE_ARGS=()
if [[ "${PROFILE_MODE}" == "trace" ]]; then
  PROFILE_ARGS+=(--pipeline-strategy-trace-path "${RUN_DIR}/strategy_traces/rank{rank}.json")
elif [[ "${PROFILE_MODE}" == "nsys" ]]; then
  PROFILE_ARGS+=(
    --profile
    --profile-step-start "${WARMUP_STEPS}"
    --profile-step-end "$((WARMUP_STEPS + 2))"
    --profile-ranks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  )
fi

TORCHRUN=(
  torchrun
  --nnodes=2
  --nproc_per_node=8
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  --rdzv_backend=static
  pretrain_gpt.py
  "${COMMON_ARGS[@]}"
  "${SCHEDULE_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
)

printf '%q ' "${TORCHRUN[@]}" > "${NODE_DIR}/command.sh"
printf '\n' >> "${NODE_DIR}/command.sh"
cd "${CODE_DIR}"

if [[ "${PROFILE_MODE}" == "nsys" ]]; then
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${NODE_DIR}/nsys_profile" \
    "${TORCHRUN[@]}" 2>&1 | tee "${NODE_DIR}/training.log"
else
  "${TORCHRUN[@]}" 2>&1 | tee "${NODE_DIR}/training.log"
fi

nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv > "${NODE_DIR}/gpu_final.csv"
touch "${NODE_DIR}/DONE"
echo "OBSERVATION_DONE case=${CASE_ID} repeat=${REPEAT_ID} node_rank=${NODE_RANK}"
