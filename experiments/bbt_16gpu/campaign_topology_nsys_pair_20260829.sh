#!/usr/bin/env bash
set -uo pipefail
ROOT="${RUN_ROOT:-/workspace/runs/g5-8gpu-strategy-research-20260829/topology_nsys_pair_$(date -u +%Y%m%dT%H%M%SZ)}"
CODE="${CODE_DIR:-/workspace/code/Megatron-LM}"
mkdir -p "$ROOT"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8
BASE=(--tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --num-layers 32 --hidden-size 1024 --ffn-hidden-size 3072 --num-attention-heads 16 --group-query-attention --num-query-groups 4 --seq-length 1024 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 8 --bf16 --use-mcore-models --use-distributed-optimizer --position-embedding-type rope --rotary-percent 1.0 --rotary-base 10000 --tokenizer-type NullTokenizer --vocab-size 32768 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear --untie-embeddings-and-output-weights --mock-data --train-iters 30 --eval-iters 0 --eval-interval 100000 --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant --log-interval 1 --seed 20260829 --pipeline-schedule interleaved --num-layers-per-virtual-pipeline-stage 2 --microbatch-group-size-per-virtual-pipeline-stage 4)
run_case(){
  local name="$1" order="$2"; local out="$ROOT/$name"; mkdir -p "$out"
  printf 'name=%s\norder=%s\n' "$name" "$order" > "$out/meta.txt"
  CUDA_VISIBLE_DEVICES="$order" timeout --signal=TERM --kill-after=60s 1800 nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt --trace-fork-before-exec=true --cuda-memory-usage=true --output "$out/$name" python3 -m torch.distributed.run --standalone --nproc-per-node=8 "$CODE/pretrain_gpt.py" "${BASE[@]}" > "$out/train.log" 2>&1
  local rc=$?
  if [ -f "$out/$name.nsys-rep" ]; then nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum,cuda_gpu_mem_size_sum --format csv --output "$out/summary" "$out/$name.nsys-rep" > "$out/nsys.log" 2>&1 || true; fi
  printf '%s\t%s\n' "$name" "$rc" >> "$ROOT/status.tsv"
  grep 'elapsed time per iteration' "$out/train.log" | tail -20 > "$out/steady.tsv" || true
}
run_case default_01234567 0,1,2,3,4,5,6,7
run_case cross_pair_control 0,2,1,3,4,7,5,6
{ echo root="$ROOT"; cat "$ROOT/status.tsv"; find "$ROOT" -name steady.tsv -print0 | xargs -0 grep -H 'elapsed time per iteration' 2>/dev/null || true; } > "$ROOT/summary.txt"
printf 'experiments_done=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/done.txt"
sleep "${HOLD_SECONDS:-50000}"
