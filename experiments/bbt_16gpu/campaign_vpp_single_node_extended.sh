#!/usr/bin/env bash
set -uo pipefail

# Single-node 8x5090 campaign.  Every case uses the same model, seed, batch,
# and process topology; only the PP/VPP/layout/communication policy changes.
ROOT="${RUN_ROOT:-/workspace/runs/g5-8gpu-strategy-research-20260829/vpp_single_extended_$(date -u +%Y%m%dT%H%M%SZ)}"
CODE="${CODE_DIR:-/workspace/code/Megatron-LM}"
mkdir -p "$ROOT"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

COMMON=(
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 4
  --num-layers 32 --hidden-size 1024 --ffn-hidden-size 3072
  --num-attention-heads 16 --group-query-attention --num-query-groups 4
  --seq-length 1024 --max-position-embeddings 4096
  --micro-batch-size 1 --global-batch-size 8 --bf16
  --use-mcore-models --use-distributed-optimizer
  --position-embedding-type rope --rotary-percent 1.0 --rotary-base 10000
  --tokenizer-type NullTokenizer --vocab-size 32768
  --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear
  --untie-embeddings-and-output-weights --mock-data
  --train-iters 50 --eval-iters 0 --eval-interval 100000
  --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant --log-interval 1
  --seed 20260829 --pipeline-schedule interleaved
)

run_case() {
  local name="$1"; shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  printf 'name=%s\ncode=%s\nstart=%s\nargs=%q ' "$name" "$CODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" python3 > "$out/meta.txt"
  printf '%q ' -m torch.distributed.run --standalone --nproc-per-node=8 "$CODE/pretrain_gpt.py" "${COMMON[@]}" "$@" >> "$out/meta.txt"
  printf '\n' >> "$out/meta.txt"
  timeout --signal=TERM --kill-after=45s 1200 \
    python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$CODE/pretrain_gpt.py" "${COMMON[@]}" "$@" > "$out/train.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/status.tsv"
  grep -E 'elapsed time per iteration|throughput per GPU|iteration|memory \(MB\)|Traceback|RuntimeError|AssertionError|ValueError|unsupported' \
    "$out/train.log" | tail -100 > "$out/extract.txt" || true
}

run_nsys() {
  local name="$1"; shift
  local out="$ROOT/nsys_$name"
  mkdir -p "$out"
  printf 'name=%s\nstart=%s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/meta.txt"
  timeout --signal=TERM --kill-after=60s 2400 \
    nsys profile --force-overwrite=true --sample=none \
      --trace=cuda,nvtx,osrt --trace-fork-before-exec=true \
      --cuda-memory-usage=true --output "$out/$name" \
    python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
      "$CODE/pretrain_gpt.py" "${COMMON[@]}" "$@" > "$out/train.log" 2>&1
  local rc=$?
  if [ -f "$out/$name.nsys-rep" ]; then
    nsys stats --force-export=true \
      --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum \
      --format csv --output "$out/summary" "$out/$name.nsys-rep" > "$out/nsys.log" 2>&1 || true
  fi
  printf '%s\t%s\t%s\n' "nsys_$name" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/status.tsv"
  grep -E 'elapsed time per iteration|throughput per GPU|iteration|Traceback|RuntimeError|AssertionError|ValueError' \
    "$out/train.log" | tail -100 > "$out/extract.txt" || true
}

echo "root=$ROOT" > "$ROOT/meta.txt"
echo "code=$CODE" >> "$ROOT/meta.txt"
echo "host=$(hostname)" >> "$ROOT/meta.txt"
echo "env=NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_IB_DISABLE=$NCCL_IB_DISABLE" >> "$ROOT/meta.txt"

# Equal-chunk VPP schedule and group-size sweep.
run_case pp4_vpp1 --pipeline-schedule 1f1b
run_case pp4_vpp1_auto --pipeline-schedule auto
run_case pp4_vpp2_default --num-layers-per-virtual-pipeline-stage 4
run_case pp4_vpp2_group4 --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case pp4_vpp2_group8 --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 8
run_case pp4_vpp2_group12 --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 12
run_case pp4_vpp2_warmup_flush --num-layers-per-virtual-pipeline-stage 4 --overlap-p2p-communication-warmup-flush
run_case pp4_vpp2_no_p2p --num-layers-per-virtual-pipeline-stage 4 --no-overlap-p2p-communication
run_case pp4_vpp4_default --num-layers-per-virtual-pipeline-stage 2
run_case pp4_vpp4_group4 --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case pp4_vpp4_group8 --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 8
run_case pp4_vpp4_warmup_flush --num-layers-per-virtual-pipeline-stage 2 --overlap-p2p-communication-warmup-flush
run_case pp4_vpp4_no_p2p --num-layers-per-virtual-pipeline-stage 2 --no-overlap-p2p-communication
run_case pp4_vpp8_default --num-layers-per-virtual-pipeline-stage 1
run_case pp4_vpp8_group4 --num-layers-per-virtual-pipeline-stage 1 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case pp4_vpp8_group8 --num-layers-per-virtual-pipeline-stage 1 --microbatch-group-size-per-virtual-pipeline-stage 8

# Megatron's existing distributed-optimizer/VPP bucket policies.
run_case pp4_vpp2_grad4 --num-layers-per-virtual-pipeline-stage 4 --overlap-grad-reduce --ddp-num-buckets 4
run_case pp4_vpp2_grad8 --num-layers-per-virtual-pipeline-stage 4 --overlap-grad-reduce --ddp-num-buckets 8
run_case pp4_vpp2_grad8_pg --num-layers-per-virtual-pipeline-stage 4 --overlap-grad-reduce --ddp-num-buckets 8 --overlap-param-gather
run_case pp4_vpp2_allchunks4 --num-layers-per-virtual-pipeline-stage 4 --overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy all-chunks
run_case pp4_vpp2_rank0chunks4 --num-layers-per-virtual-pipeline-stage 4 --overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy rank0-all-chunks
run_case pp4_vpp4_grad8 --num-layers-per-virtual-pipeline-stage 2 --overlap-grad-reduce --ddp-num-buckets 8
run_case pp4_vpp4_allchunks4 --num-layers-per-virtual-pipeline-stage 2 --overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy all-chunks

# Communication granularity and TP/SP controls, all with the same PP/VPP.
run_case pp4_vpp2_nosg --num-layers-per-virtual-pipeline-stage 4 --no-scatter-gather-tensors-in-pipeline
run_case pp4_vpp2_tpcomm --num-layers-per-virtual-pipeline-stage 4 --tp-comm-overlap
run_case pp4_vpp2_sp --num-layers-per-virtual-pipeline-stage 4 --sequence-parallel
run_case pp2_vpp2_tp4 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --num-layers-per-virtual-pipeline-stage 8
run_case pp8_vpp1_tp1 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 8

# Flexible layout: same ownership count but explicitly placed VPP chunks.
run_case layout_uniform_vpp2 --pipeline-model-parallel-layout 'Etttt|tttt|tttt|tttt|tttt|tttt|tttt|ttttL'
run_case layout_first_heavy_vpp2 --pipeline-model-parallel-layout 'Ettttt|ttt|tttt|tttt|tttt|tttt|tttt|ttttL'
run_case layout_boundary_light_vpp2 --pipeline-model-parallel-layout 'Ettt|ttttt|tttt|tttt|tttt|tttt|tttt|ttttL'

# Full traces for the best regular candidates and the custom-layout candidate.
run_nsys vpp2_group4 --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 4
run_nsys vpp4_group4 --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 4
run_nsys vpp4_allchunks4 --num-layers-per-virtual-pipeline-stage 2 --overlap-grad-reduce --ddp-num-buckets 4 --vpp-bucket-policy all-chunks
run_nsys layout_uniform_vpp2 --pipeline-model-parallel-layout 'Etttt|tttt|tttt|tttt|tttt|tttt|tttt|ttttL'

# Two independent repeats to estimate variance for the regular winner and its
# strongest negative control.  They are kept separate from the main matrix.
run_case repeat_vpp2_group4_a --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case repeat_vpp2_group4_b --num-layers-per-virtual-pipeline-stage 4 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case repeat_vpp4_group4_a --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 4
run_case repeat_vpp4_group4_b --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 4

{
  echo "root=$ROOT"
  echo "status:"
  cat "$ROOT/status.tsv" 2>/dev/null || true
  echo "throughput:"
  find "$ROOT" -name train.log -print0 | xargs -0 grep -H -E 'elapsed time per iteration|throughput per GPU' 2>/dev/null || true
} > "$ROOT/summary.txt"
echo "experiments_done=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/done.txt"

# Keep the 8-GPU reservation alive through 02:00 Asia/Shanghai tomorrow.
sleep "${HOLD_SECONDS:-110000}"
