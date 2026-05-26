#!/usr/bin/env bash
set -euo pipefail

# Test Matrix Runner - runs inside pod
# Usage: bash run_test_matrix_pod.sh [test_id]
# test_id: 1-8, or "all"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/perf_pp_vpp/outputs/test_matrix_$(date +%Y%m%d_%H%M%S)"

# Common model args for 32B
COMMON_ARGS="--use-mcore-models --transformer-impl transformer_engine --attention-backend flash --use-flash-attn \
--num-layers 64 --hidden-size 5120 --ffn-hidden-size 25600 --num-attention-heads 64 \
--group-query-attention --num-query-groups 8 --kv-channels 80 \
--max-position-embeddings 40960 --position-embedding-type rope --rotary-percent 1.0 --rotary-base 1000000 \
--normalization RMSNorm --norm-epsilon 1e-6 --qk-layernorm --swiglu \
--untie-embeddings-and-output-weights --disable-bias-linear \
--attention-dropout 0.0 --hidden-dropout 0.0 --make-vocab-size-divisible-by 128 --vocab-size 151936 \
--global-batch-size 32 --train-iters 6 --eval-iters 0 --eval-interval 1 --log-interval 1 \
--timing-log-level 2 --log-throughput --distributed-timeout-minutes 60 \
--lr 1e-4 --min-lr 1e-4 --lr-decay-style constant --lr-warmup-iters 0 --seed 1234 --bf16 \
--empty-unused-memory-level 1 --recompute-granularity full --no-gradient-accumulation-fusion \
--no-rope-fusion --no-persist-layer-norm --recompute-method uniform --recompute-num-layers 1 \
--use-distributed-optimizer --sequence-parallel --use-precision-aware-optimizer \
--main-grads-dtype bf16 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 \
--mock-data --tokenizer-type NullTokenizer --split 99,1,0 \
--no-create-attention-mask-in-dataloader --no-mmap-bin-files --num-workers 1"

run_test() {
    local test_name="$1"
    local tp="$2"
    local pp="$3"
    local vpp="$4"
    local mbs="$5"
    local seq_len="$6"
    local overlap="$7"
    
    local run_dir="${OUTPUT_DIR}/${test_name}"
    mkdir -p "${run_dir}/nsys"
    
    echo "=========================================="
    echo "Running: ${test_name}"
    echo "  TP=${tp} PP=${pp} VPP=${vpp} MBS=${mbs} seq=${seq_len}"
    echo "  overlap=${overlap}"
    echo "  output=${run_dir}"
    echo "=========================================="
    
    local VPP_ARG=""
    if [[ "${vpp}" -gt 0 ]]; then
        VPP_ARG="--num-layers-per-virtual-pipeline-stage ${vpp}"
    fi
    
    local OVERLAP_ARG="--no-overlap-p2p-communication"
    if [[ "${overlap}" == "yes" ]]; then
        OVERLAP_ARG="--overlap-p2p-communication"
    fi
    
    local CMD="torchrun --nnodes=2 --nproc_per_node=8 --node_rank=\${NODE_RANK} --master_addr=\${MASTER_ADDR} --master_port=29500 \
${PROJECT_ROOT}/pretrain_gpt.py \
--tensor-model-parallel-size ${tp} \
--pipeline-model-parallel-size ${pp} \
${VPP_ARG} \
--micro-batch-size ${mbs} \
--seq-length ${seq_len} \
${OVERLAP_ARG} \
${COMMON_ARGS}"
    
    echo "Command: ${CMD}" > "${run_dir}/command.txt"
    
    # Run on both nodes
    if [[ "${NODE_RANK}" == "0" ]]; then
        eval "${CMD}" > "${run_dir}/stdout_node0.log" 2> "${run_dir}/stderr_node0.log" &
    else
        eval "${CMD}" > "${run_dir}/stdout_node1.log" 2> "${run_dir}/stderr_node1.log" &
    fi
    
    wait
    
    echo "Completed: ${test_name}"
}

# Main
TEST_ID="${1:-all}"

# Define tests: name tp pp vpp mbs seq overlap
declare -a TEST_NAMES=(
    "tp2_pp4_vpp1_mbs1"
    "tp2_pp4_vpp2_mbs1"
    "tp2_pp4_vpp4_mbs1_baseline"
    "tp2_pp4_vpp2_mbs2"
    "tp1_pp8_vpp1"
    "tp1_pp8_vpp2"
    "tp4_pp2_vpp1"
    "tp4_pp2_vpp2"
)
declare -a TEST_TP=(2 2 2 2 1 1 4 4)
declare -a TEST_PP=(4 4 4 4 8 8 2 2)
declare -a TEST_VPP=(1 2 4 2 1 2 1 2)
declare -a TEST_MBS=(1 1 1 2 1 1 1 1)
declare -a TEST_SEQ=(4096 4096 4096 4096 4096 4096 4096 4096)
declare -a TEST_OVERLAP=(no no no no no no no no)

mkdir -p "${OUTPUT_DIR}"

if [[ "${TEST_ID}" == "all" ]]; then
    for i in "${!TEST_NAMES[@]}"; do
        run_test "${TEST_NAMES[$i]}" "${TEST_TP[$i]}" "${TEST_PP[$i]}" "${TEST_VPP[$i]}" "${TEST_MBS[$i]}" "${TEST_SEQ[$i]}" "${TEST_OVERLAP[$i]}"
    done
else
    idx=$((TEST_ID - 1))
    if [[ ${idx} -ge 0 && ${idx} -lt ${#TEST_NAMES[@]} ]]; then
        run_test "${TEST_NAMES[$idx]}" "${TEST_TP[$idx]}" "${TEST_PP[$idx]}" "${TEST_VPP[$idx]}" "${TEST_MBS[$idx]}" "${TEST_SEQ[$idx]}" "${TEST_OVERLAP[$idx]}"
    else
        echo "Invalid test ID: ${TEST_ID}" >&2
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
