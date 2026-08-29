#!/usr/bin/env bash
set -uo pipefail
ROOT="${RUN_ROOT:-/workspace/runs/g5-8gpu-strategy-research-20260829/layout_boundary_repeat_$(date -u +%Y%m%dT%H%M%SZ)}"
CODE="${CODE_DIR:-/workspace/code/Megatron-LM}"
mkdir -p "$ROOT"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
BASE=(--tensor-model-parallel-size 2 --pipeline-model-parallel-size 4 --num-layers 32 --hidden-size 1024 --ffn-hidden-size 3072 --num-attention-heads 16 --group-query-attention --num-query-groups 4 --seq-length 1024 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 8 --bf16 --use-mcore-models --use-distributed-optimizer --position-embedding-type rope --rotary-percent 1.0 --rotary-base 10000 --tokenizer-type NullTokenizer --vocab-size 32768 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear --untie-embeddings-and-output-weights --mock-data --train-iters 50 --eval-iters 0 --eval-interval 100000 --lr 1e-6 --min-lr 1e-7 --lr-decay-style constant --log-interval 1 --seed 20260829 --pipeline-schedule interleaved)
run_case(){ local name="$1" layout="$2"; local out="$ROOT/$name"; mkdir -p "$out"; printf 'name=%s\nlayout=%s\n' "$name" "$layout" > "$out/meta.txt"; timeout --signal=TERM --kill-after=45s 1200 python3 -m torch.distributed.run --standalone --nproc-per-node=8 "$CODE/pretrain_gpt.py" "${BASE[@]}" --pipeline-model-parallel-layout "$layout" > "$out/train.log" 2>&1; local rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$ROOT/status.tsv"; grep 'elapsed time per iteration' "$out/train.log" | tail -30 > "$out/steady.tsv" || true; }
U='Etttt|tttt|tttt|tttt|tttt|tttt|tttt|ttttL'
B='Ettt|ttttt|tttt|tttt|tttt|tttt|tttt|ttttL'
H='Ettttt|ttt|tttt|tttt|tttt|tttt|tttt|ttttL'
for r in 1 2 3; do
  run_case "uniform_r${r}" "$U"
  run_case "boundary_light_r${r}" "$B"
  run_case "first_heavy_r${r}" "$H"
done
{ echo root="$ROOT"; cat "$ROOT/status.tsv"; for f in "$ROOT"/*/steady.tsv; do printf '%s ' "$f"; grep 'elapsed time' "$f" | sed -n 's/.*elapsed time per iteration (ms): \([0-9.]*\).*/\1/p' | tail -15 | sort -n | awk '{a[NR]=$1;s+=$1} END{if(NR){printf "n=%d med=%.3f mean=%.3f trim=%.3f\n",NR,a[int((NR+1)/2)],s/NR,s/NR}}'; done; } > "$ROOT/summary.txt"
printf 'experiments_done=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/done.txt"
sleep "${HOLD_SECONDS:-50000}"
