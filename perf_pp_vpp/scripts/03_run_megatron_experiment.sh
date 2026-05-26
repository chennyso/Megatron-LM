#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

EXPERIMENT_YAML=""
EXPERIMENT_NAME=""
SEQ_LEN=""
REPEAT_ID=""
PROFILE_MODE="none"
PROFILE_RANKS="none"
OVERLAP_MODE="baseline"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-yaml) EXPERIMENT_YAML="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --repeat-id) REPEAT_ID="$2"; shift 2 ;;
    --profile-mode) PROFILE_MODE="$2"; shift 2 ;;
    --profile-ranks) PROFILE_RANKS="$2"; shift 2 ;;
    --overlap-mode) OVERLAP_MODE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${EXPERIMENT_YAML}" || -z "${EXPERIMENT_NAME}" || -z "${SEQ_LEN}" || -z "${REPEAT_ID}" ]]; then
  echo "missing required args" >&2
  exit 2
fi

export PERF_EXPERIMENT_YAML="${EXPERIMENT_YAML}"
export PERF_EXPERIMENT_NAME="${EXPERIMENT_NAME}"
export PERF_SEQ_LEN="${SEQ_LEN}"
export PERF_REPEAT_ID="${REPEAT_ID}"
export PERF_OVERLAP_MODE="${OVERLAP_MODE}"
RUN_MASTER_PORT="$((MASTER_PORT + REPEAT_ID))"
export RUN_MASTER_PORT
if [[ "${PROFILE_RANKS}" == "boundary" ]]; then
  PROFILE_RANKS="0,7,8,15"
fi

RESOLVED_JSON="$(${PYTHON_BIN} - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "perf_pp_vpp" / "scripts"))
from _common import resolved_experiment
data = resolved_experiment(Path(os.environ["PERF_EXPERIMENT_YAML"]), os.environ["PERF_EXPERIMENT_NAME"], int(os.environ["PERF_SEQ_LEN"]), int(os.environ["PERF_REPEAT_ID"]), os.environ["PERF_OVERLAP_MODE"])
print(json.dumps(data))
PY
)"
export RESOLVED_JSON

RUN_DIR="$(${PYTHON_BIN} - <<'PY'
import json, os
print(json.loads(os.environ["RESOLVED_JSON"])["run_dir"])
PY
)"
REMOTE_RUN_DIR="$(${PYTHON_BIN} - <<'PY'
import json, os
print(json.loads(os.environ["RESOLVED_JSON"])["remote_run_dir"])
PY
)"
export RUN_DIR REMOTE_RUN_DIR
mkdir -p "${RUN_DIR}"
printf '%s\n' "$0 $*" > "${RUN_DIR}/args.txt"
env | sort > "${RUN_DIR}/env.txt"
${PYTHON_BIN} - <<'PY'
import json, os, sys
from pathlib import Path
path = Path(os.environ["RUN_DIR"]) / "config_resolved.yaml"
import yaml
path.write_text(yaml.safe_dump(json.loads(os.environ["RESOLVED_JSON"]), sort_keys=False), encoding="utf-8")
PY

DATA_MANIFEST="${OUTPUT_ROOT}/data_manifest.json"
if [[ ! -f "${DATA_MANIFEST}" ]]; then
  echo "missing ${DATA_MANIFEST}; run 02_prepare_mock_data.sh first" >&2
  exit 2
fi
export DATA_MANIFEST
MOCK_MODE="$(${PYTHON_BIN} - <<'PY'
import json, os
print(json.load(open(os.environ["DATA_MANIFEST"], "r", encoding="utf-8"))["MOCK_DATA_MODE"])
PY
)"

METRICS_JSONL="${RUN_DIR}/metrics_raw.jsonl"
: > "${METRICS_JSONL}"

WORKER_ENV="${RUN_DIR}/worker_env.sh"
WORKER_CMD="${RUN_DIR}/worker_command.sh"
REMOTE_WORKER_ENV="${REMOTE_RUN_DIR}/worker_env.sh"
REMOTE_WORKER_CMD="${REMOTE_RUN_DIR}/worker_command.sh"

export WORKER_ENV WORKER_CMD RUN_DIR REMOTE_RUN_DIR MOCK_MODE PROFILE_MODE PROFILE_RANKS TOKENIZER_PATH
${PYTHON_BIN} - <<'PY'
import json
import os
from pathlib import Path

resolved = json.loads(os.environ["RESOLVED_JSON"])
base = resolved["base"]
model = resolved["model"]
exp = resolved["experiment"]
overlap = resolved["overlap"]
work_env = Path(os.environ["WORKER_ENV"])
work_cmd = Path(os.environ["WORKER_CMD"])
run_dir = Path(os.environ["RUN_DIR"])
remote_run = Path(os.environ["REMOTE_RUN_DIR"])
mock_mode = os.environ["MOCK_MODE"]
profile_mode = os.environ["PROFILE_MODE"]
profile_ranks = os.environ["PROFILE_RANKS"]
tokenizer_path = os.environ.get("TOKENIZER_PATH", "")
layers_per_virtual = exp["layers_per_virtual_stage"]
num_virtual_stages = os.environ.get("PERF_NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK", "")

global_batch = int(resolved["global_batch_size"])
micro_batch = int(resolved["micro_batch_size"])
seq_len = int(resolved["seq_len"])
train_iters = int(os.environ.get("PERF_TRAIN_ITERS", base["train_iters"]))
warmup_iters = int(os.environ.get("PERF_WARMUP_ITERS", base["warmup_iters"]))
profile_start = int(os.environ.get("PERF_PROFILE_START_ITER", base["profile_start_iter"]))
profile_end = int(os.environ.get("PERF_PROFILE_END_ITER", base["profile_end_iter"]))
use_mock_data = mock_mode == "mock_data"
data_path = ""
if not use_mock_data:
    manifest = json.load(open(Path(os.environ["OUTPUT_ROOT"]) / "data_manifest.json", "r", encoding="utf-8"))
    data_path = manifest["dataset_prefix"]
recompute_granularity = os.environ.get("PERF_RECOMPUTE_GRANULARITY", "full")
recompute_method = os.environ.get("PERF_RECOMPUTE_METHOD", "uniform")
recompute_num_layers = os.environ.get("PERF_RECOMPUTE_NUM_LAYERS", "1")
recompute_modules = os.environ.get("PERF_RECOMPUTE_MODULES", "")

env_lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    f"export REMOTE_PROJECT_ROOT={os.environ['REMOTE_PROJECT_ROOT']!r}",
    f"export PYTHONPATH={os.environ['REMOTE_PROJECT_ROOT']}:{os.environ['REMOTE_PROJECT_ROOT']}/perf_pp_vpp/scripts:{os.environ['REMOTE_PROJECT_ROOT']}/perf_pp_vpp/megatron_patches:${{PYTHONPATH:-}}",
    f"export PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')!r}",
    f"export CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '1')!r}",
    f"export NCCL_DEBUG={os.environ.get('NCCL_DEBUG', 'WARN')!r}",
    f"export NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'net1')!r}",
    f"export GLOO_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'net1')!r}",
    f"export NSYS_BIN={os.environ.get('NSYS_BIN', 'nsys')!r}",
    f"export PERF_PROFILE_MODE={profile_mode!r}",
    f"export PERF_PROFILE_RANKS={profile_ranks!r}",
    f"export PERF_SEED={os.environ.get('PERF_SEED', '1234')!r}",
    f"export REMOTE_RUN_DIR={str(remote_run)!r}",
    f"export TORCHRUN_BIN={os.environ.get('TORCHRUN_BIN', 'torchrun')!r}",
    f"export PYTHON_BIN={os.environ.get('PYTHON_BIN', 'python')!r}",
]
work_env.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

cmd = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "source \"$(dirname \"$0\")/worker_env.sh\"",
    "cd \"${REMOTE_PROJECT_ROOT}\"",
    "mkdir -p \"${REMOTE_RUN_DIR}\" \"${REMOTE_RUN_DIR}/nsys\"",
    "RDMA_DEV=\"$(ls /sys/class/net/${NCCL_SOCKET_IFNAME}/device/infiniband 2>/dev/null | head -n 1 || true)\"",
    "if [[ -n \"${RDMA_DEV}\" ]]; then export NCCL_IB_HCA=\"${RDMA_DEV}\"; fi",
    "export NCCL_P2P_DISABLE=1",
    "export NCCL_IB_DISABLE=0",
    "export NCCL_IB_GID_INDEX=0",
    "export NCCL_NET_GDR_LEVEL=5",
    "export NCCL_IB_TC=136",
    "python - <<'PY'\n"
    "import json, os, pathlib\n"
    "rank = os.environ.get('NODE_RANK', '0')\n"
    "path = pathlib.Path(os.environ['REMOTE_RUN_DIR']) / f'rank_mapping_node{rank}.json'\n"
    "payload = {'node_rank': int(rank), 'master_addr': os.environ.get('MASTER_ADDR', ''), 'master_port': os.environ.get('MASTER_PORT', '')}\n"
    "path.write_text(json.dumps(payload, indent=2), encoding='utf-8')\n"
    "PY",
]

megatron_args = [
    "--use-mcore-models",
    "--transformer-impl", os.environ.get("PERF_TRANSFORMER_IMPL", "local"),
    "--attention-backend", os.environ.get("PERF_ATTENTION_BACKEND", "flash"),
    "--use-flash-attn",
    "--num-layers", str(model["num_layers"]),
    "--hidden-size", str(model["hidden_size"]),
    "--ffn-hidden-size", str(model["intermediate_size"]),
    "--num-attention-heads", str(model["num_attention_heads"]),
    "--group-query-attention",
    "--num-query-groups", str(model["num_query_groups"]),
    "--kv-channels", str(model["hidden_size"] // model["num_attention_heads"]),
    "--seq-length", str(seq_len),
    "--max-position-embeddings", str(model["max_position_embeddings"]),
    "--position-embedding-type", "rope",
    "--rotary-percent", "1.0",
    "--rotary-base", str(model["rotary_base"]),
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-6",
    "--qk-layernorm",
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--disable-bias-linear",
    "--attention-dropout", "0.0",
    "--hidden-dropout", "0.0",
    "--make-vocab-size-divisible-by", "128",
    "--vocab-size", str(model["vocab_size"]),
    "--tensor-model-parallel-size", str(exp["tp"]),
    "--pipeline-model-parallel-size", str(exp["pp"]),
    "--micro-batch-size", str(micro_batch),
    "--global-batch-size", str(global_batch),
    "--train-iters", str(train_iters),
    "--eval-iters", "0",
    "--eval-interval", "1",
    "--log-interval", "1",
    "--timing-log-level", "2",
    "--log-throughput",
    "--distributed-timeout-minutes", "60",
    "--lr", "1e-4",
    "--min-lr", "1e-4",
    "--lr-decay-style", "constant",
    "--lr-warmup-iters", "0",
    "--seed", str(os.environ.get("PERF_SEED", "1234")),
    "--bf16",
    "--empty-unused-memory-level", os.environ.get("PERF_EMPTY_UNUSED_MEMORY_LEVEL", "1"),
    "--recompute-granularity", recompute_granularity,
    "--no-gradient-accumulation-fusion",
    "--no-rope-fusion",
    "--no-persist-layer-norm",
]
if recompute_granularity == "selective":
    if recompute_modules:
        megatron_args.extend(["--recompute-modules", recompute_modules])
else:
    megatron_args.extend(["--recompute-method", recompute_method])
    megatron_args.extend(["--recompute-num-layers", recompute_num_layers])
distopt_mode = os.environ.get("PERF_DISTOPT_MODE", "auto")
enable_distopt = (
    distopt_mode == "on" or
    (distopt_mode == "auto" and int(exp["dp"]) > 1)
)
if enable_distopt:
    megatron_args.append("--use-distributed-optimizer")
if int(exp["tp"]) > 1 and os.environ.get("PERF_SEQUENCE_PARALLEL", "1") == "1":
    megatron_args.append("--sequence-parallel")
if enable_distopt and os.environ.get("PERF_USE_PRECISION_AWARE_OPTIMIZER", "0") == "1":
    megatron_args.append("--use-precision-aware-optimizer")
    megatron_args.extend(["--main-grads-dtype", os.environ.get("PERF_MAIN_GRADS_DTYPE", "bf16")])
    megatron_args.extend(["--exp-avg-dtype", os.environ.get("PERF_EXP_AVG_DTYPE", "bf16")])
    megatron_args.extend(["--exp-avg-sq-dtype", os.environ.get("PERF_EXP_AVG_SQ_DTYPE", "bf16")])
    if os.environ.get("PERF_MAIN_PARAMS_DTYPE", ""):
        megatron_args.extend(["--main-params-dtype", os.environ["PERF_MAIN_PARAMS_DTYPE"]])
first_stage_layers = os.environ.get("PERF_DECODER_FIRST_PIPELINE_NUM_LAYERS")
last_stage_layers = os.environ.get("PERF_DECODER_LAST_PIPELINE_NUM_LAYERS")
pipeline_layout = os.environ.get("PERF_PIPELINE_MODEL_PARALLEL_LAYOUT")
account_embedding = os.environ.get("PERF_ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT", "0") == "1"
account_loss = os.environ.get("PERF_ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT", "0") == "1"
if first_stage_layers:
    megatron_args.extend(["--decoder-first-pipeline-num-layers", first_stage_layers])
if last_stage_layers:
    megatron_args.extend(["--decoder-last-pipeline-num-layers", last_stage_layers])
if pipeline_layout:
    megatron_args.extend(["--pipeline-model-parallel-layout", pipeline_layout])
if account_embedding:
    megatron_args.append("--account-for-embedding-in-pipeline-split")
if account_loss:
    megatron_args.append("--account-for-loss-in-pipeline-split")
if exp["dp"] > 1:
    megatron_args.extend(["--overlap-grad-reduce"] if overlap["overlap_grad_reduce"] else [])
    megatron_args.extend(["--overlap-param-gather"] if overlap["overlap_param_gather"] else [])
if os.environ.get("PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP", "0") == "1":
    megatron_args.append("--overlap-param-gather-with-optimizer-step")
if not overlap["overlap_p2p_comm"]:
    megatron_args.append("--no-overlap-p2p-communication")
if os.environ.get("PERF_OVERLAP_P2P_WARMUP_FLUSH", "0") == "1":
    megatron_args.append("--overlap-p2p-communication-warmup-flush")
if int(exp.get("vpp", 1)) > 1 and not pipeline_layout and num_virtual_stages:
    megatron_args.extend([
        "--num-virtual-stages-per-pipeline-rank",
        num_virtual_stages,
    ])
elif int(exp.get("vpp", 1)) > 1 and not pipeline_layout:
    megatron_args.extend(["--num-layers-per-virtual-pipeline-stage", str(layers_per_virtual)])
if os.environ.get("PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE", ""):
    megatron_args.extend([
        "--microbatch-group-size-per-virtual-pipeline-stage",
        os.environ["PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE"],
    ])
if os.environ.get("PERF_DELAY_WGRAD_COMPUTE", "0") == "1":
    megatron_args.append("--delay-wgrad-compute")
if use_mock_data:
    megatron_args.extend([
        "--mock-data",
        "--tokenizer-type", "NullTokenizer",
        "--split", "99,1,0",
        "--no-create-attention-mask-in-dataloader",
        "--no-mmap-bin-files",
        "--num-workers", "1",
    ])
else:
    megatron_args.extend([
        "--data-path", data_path,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_path or model["model_root"],
        "--tokenizer-hf-include-special-tokens",
        "--split", "99,1,0",
        "--no-create-attention-mask-in-dataloader",
        "--no-mmap-bin-files",
        "--num-workers", "1",
    ])
if profile_mode == "none":
    pass
else:
    megatron_args.extend(["--profile", "--profile-step-start", str(profile_start), "--profile-step-end", str(profile_end)])

quoted = " ".join(__import__("shlex").quote(x) for x in megatron_args)
torchrun_cmd = (
    f"{os.environ.get('TORCHRUN_BIN', 'torchrun')} "
    f"--nnodes={os.environ.get('NNODES', '2')} "
    f"--nproc_per_node={os.environ.get('NPROC_PER_NODE', '8')} "
    f"--node_rank=${{NODE_RANK}} "
    f"--master_addr=${{MASTER_ADDR}} "
    f"--master_port=${{RUN_MASTER_PORT}} "
    f"{os.environ['REMOTE_PROJECT_ROOT']}/perf_pp_vpp/scripts/torchrun_rank_entry.py "
    f"--entry {os.environ['REMOTE_PROJECT_ROOT']}/pretrain_gpt.py "
    f"--profile-mode {profile_mode} "
    f"--profile-ranks {profile_ranks} "
    f"--nsys-bin {os.environ.get('NSYS_BIN', 'nsys')} "
    f"--nsys-output-prefix {str(remote_run / 'nsys' / exp['name'])} "
    f"--patch-bootstrap perf_pp_vpp.megatron_patches.nvtx_instrumentation "
    f"-- {quoted}"
)
cmd.extend([
    f"echo {__import__('shlex').quote(torchrun_cmd)} > \"${{REMOTE_RUN_DIR}}/torchrun_command.txt\"",
    f"stdbuf -oL -eL bash -lc {__import__('shlex').quote(torchrun_cmd)} > \"${{REMOTE_RUN_DIR}}/megatron_stdout_node${{NODE_RANK}}.log\" 2> \"${{REMOTE_RUN_DIR}}/megatron_stderr_node${{NODE_RANK}}.log\"",
])
work_cmd.write_text("\n".join(cmd) + "\n", encoding="utf-8")
PY

upload_file() {
  local pod="$1"
  local local_path="$2"
  local remote_path="$3"

  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "mkdir -p \"$(dirname "${remote_path}")\"" < /dev/null
  cat "${local_path}" | ${KUBECTL_BIN} exec -i -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "cat > \"${remote_path}\""
}

for pod in "${NODE0_POD}" "${NODE1_POD}"; do
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "mkdir -p \"${REMOTE_RUN_DIR}\"" < /dev/null
  upload_file "${pod}" "${WORKER_ENV}" "${REMOTE_WORKER_ENV}"
  upload_file "${pod}" "${WORKER_CMD}" "${REMOTE_WORKER_CMD}"
  ${KUBECTL_BIN} exec -n "${K8S_NAMESPACE}" "${pod}" -- sh -lc "chmod +x \"${REMOTE_WORKER_ENV}\" \"${REMOTE_WORKER_CMD}\"" < /dev/null
done

NODE1_CMD="${KUBECTL_BIN} exec -n ${K8S_NAMESPACE} ${NODE1_POD} -- bash -lc 'export NODE_RANK=1 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${RUN_MASTER_PORT} RUN_MASTER_PORT=${RUN_MASTER_PORT}; cd ${REMOTE_PROJECT_ROOT}; bash ${REMOTE_WORKER_CMD}' < /dev/null"
NODE0_CMD="${KUBECTL_BIN} exec -n ${K8S_NAMESPACE} ${NODE0_POD} -- bash -lc 'export NODE_RANK=0 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${RUN_MASTER_PORT} RUN_MASTER_PORT=${RUN_MASTER_PORT}; cd ${REMOTE_PROJECT_ROOT}; bash ${REMOTE_WORKER_CMD}' < /dev/null"

bash -lc "${NODE1_CMD}" &
PID1=$!
sleep 5
set +e
bash -lc "${NODE0_CMD}"
RC0=$?
wait "${PID1}"
RC1=$?
set -e

${KUBECTL_BIN} cp "${K8S_NAMESPACE}/${NODE0_POD}:${REMOTE_RUN_DIR}" "${RUN_DIR}/remote_artifacts" >/dev/null 2>&1 || true
${PYTHON_BIN} perf_pp_vpp/scripts/06_parse_megatron_logs.py "${RUN_DIR}"
printf '%s\n' "{\"node0_returncode\": ${RC0}, \"node1_returncode\": ${RC1}}" > "${RUN_DIR}/rank_mapping.json"
if [[ ${RC0} -ne 0 || ${RC1} -ne 0 ]]; then
  exit 1
fi
