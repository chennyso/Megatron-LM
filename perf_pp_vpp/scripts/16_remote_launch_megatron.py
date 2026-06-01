#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from _common import resolved_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-yaml", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--profile-mode", choices=["none", "nsys"], default="none")
    parser.add_argument("--profile-ranks", default="none")
    parser.add_argument("--overlap-mode", default="baseline")
    return parser.parse_args()


def load_mock_mode() -> str:
    manifest = Path(os.environ["OUTPUT_ROOT"]) / "data_manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["MOCK_DATA_MODE"]


def build_megatron_args(
    resolved: dict,
    mock_mode: str,
    profile_mode: str,
    tokenizer_path: str,
) -> list[str]:
    base = resolved["base"]
    model = resolved["model"]
    exp = resolved["experiment"]
    overlap = resolved["overlap"]
    seq_len = int(resolved["seq_len"])
    global_batch = int(resolved["global_batch_size"])
    micro_batch = int(resolved["micro_batch_size"])
    layers_per_virtual = exp["layers_per_virtual_stage"]
    num_virtual_stages = os.environ.get("PERF_NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK", "").strip()
    recompute_granularity = os.environ.get("PERF_RECOMPUTE_GRANULARITY", "full")
    recompute_method = os.environ.get("PERF_RECOMPUTE_METHOD", "uniform")
    recompute_num_layers = os.environ.get("PERF_RECOMPUTE_NUM_LAYERS", "1")
    recompute_modules = os.environ.get("PERF_RECOMPUTE_MODULES", "")

    train_iters = int(os.environ.get("PERF_TRAIN_ITERS", base["train_iters"]))
    profile_start = int(os.environ.get("PERF_PROFILE_START_ITER", base["profile_start_iter"]))
    profile_end = int(os.environ.get("PERF_PROFILE_END_ITER", base["profile_end_iter"]))

    args = [
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
        "--seed", os.environ.get("PERF_SEED", "1234"),
        "--bf16",
        "--empty-unused-memory-level", os.environ.get("PERF_EMPTY_UNUSED_MEMORY_LEVEL", "1"),
        "--recompute-granularity", recompute_granularity,
        "--no-gradient-accumulation-fusion",
        "--no-rope-fusion",
        "--no-persist-layer-norm",
    ]
    if recompute_granularity == "selective":
        if recompute_modules:
            args.extend(["--recompute-modules", recompute_modules])
    else:
        args.extend(["--recompute-method", recompute_method])
        args.extend(["--recompute-num-layers", recompute_num_layers])

    distopt_mode = os.environ.get("PERF_DISTOPT_MODE", "auto")
    enable_distopt = distopt_mode == "on" or (distopt_mode == "auto" and int(exp["dp"]) > 1)
    if enable_distopt:
        args.append("--use-distributed-optimizer")
    if int(exp["tp"]) > 1 and os.environ.get("PERF_SEQUENCE_PARALLEL", "1") == "1":
        args.append("--sequence-parallel")
    if enable_distopt and os.environ.get("PERF_USE_PRECISION_AWARE_OPTIMIZER", "0") == "1":
        args.append("--use-precision-aware-optimizer")
        args.extend(["--main-grads-dtype", os.environ.get("PERF_MAIN_GRADS_DTYPE", "bf16")])
        args.extend(["--exp-avg-dtype", os.environ.get("PERF_EXP_AVG_DTYPE", "bf16")])
        args.extend(["--exp-avg-sq-dtype", os.environ.get("PERF_EXP_AVG_SQ_DTYPE", "bf16")])
        if os.environ.get("PERF_MAIN_PARAMS_DTYPE", ""):
            args.extend(["--main-params-dtype", os.environ["PERF_MAIN_PARAMS_DTYPE"]])

    first_stage_layers = os.environ.get("PERF_DECODER_FIRST_PIPELINE_NUM_LAYERS")
    last_stage_layers = os.environ.get("PERF_DECODER_LAST_PIPELINE_NUM_LAYERS")
    pipeline_layout = os.environ.get("PERF_PIPELINE_MODEL_PARALLEL_LAYOUT")
    account_embedding = os.environ.get("PERF_ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT", "0") == "1"
    account_loss = os.environ.get("PERF_ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT", "0") == "1"
    if first_stage_layers:
        args.extend(["--decoder-first-pipeline-num-layers", first_stage_layers])
    if last_stage_layers:
        args.extend(["--decoder-last-pipeline-num-layers", last_stage_layers])
    if pipeline_layout:
        args.extend(["--pipeline-model-parallel-layout", pipeline_layout])
    if account_embedding:
        args.append("--account-for-embedding-in-pipeline-split")
    if account_loss:
        args.append("--account-for-loss-in-pipeline-split")

    if exp["dp"] > 1:
        if overlap["overlap_grad_reduce"]:
            args.append("--overlap-grad-reduce")
        if overlap["overlap_param_gather"]:
            args.append("--overlap-param-gather")
    if os.environ.get("PERF_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP", "0") == "1":
        args.append("--overlap-param-gather-with-optimizer-step")
    if not overlap["overlap_p2p_comm"]:
        args.append("--no-overlap-p2p-communication")
    if os.environ.get("PERF_OVERLAP_P2P_WARMUP_FLUSH", "0") == "1":
        args.append("--overlap-p2p-communication-warmup-flush")
    if int(exp.get("vpp", 1)) > 1 and not pipeline_layout and num_virtual_stages:
        args.extend(["--num-virtual-stages-per-pipeline-rank", num_virtual_stages])
    elif int(exp.get("vpp", 1)) > 1 and not pipeline_layout:
        args.extend(["--num-layers-per-virtual-pipeline-stage", str(layers_per_virtual)])
    if os.environ.get("PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE", ""):
        args.extend([
            "--microbatch-group-size-per-virtual-pipeline-stage",
            os.environ["PERF_MICROBATCH_GROUP_SIZE_PER_VP_STAGE"],
        ])
    if os.environ.get("PERF_DELAY_WGRAD_COMPUTE", "0") == "1":
        args.append("--delay-wgrad-compute")

    if mock_mode == "mock_data":
        args.extend([
            "--mock-data",
            "--tokenizer-type", "NullTokenizer",
            "--split", "99,1,0",
            "--no-create-attention-mask-in-dataloader",
            "--no-mmap-bin-files",
            "--num-workers", "1",
        ])
    else:
        manifest = json.loads((Path(os.environ["OUTPUT_ROOT"]) / "data_manifest.json").read_text(encoding="utf-8"))
        args.extend([
            "--data-path", manifest["dataset_prefix"],
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--tokenizer-model", tokenizer_path or model["model_root"],
            "--tokenizer-hf-include-special-tokens",
            "--split", "99,1,0",
            "--no-create-attention-mask-in-dataloader",
            "--no-mmap-bin-files",
            "--num-workers", "1",
        ])

    if profile_mode != "none":
        args.extend(["--profile", "--profile-step-start", str(profile_start), "--profile-step-end", str(profile_end)])
    return args


def main() -> None:
    args = parse_args()
    resolved = resolved_experiment(
        Path(args.experiment_yaml),
        args.experiment_name,
        args.seq_len,
        args.repeat_id,
        args.overlap_mode,
    )
    remote_run_dir = Path(resolved["remote_run_dir"])
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    (remote_run_dir / "nsys").mkdir(parents=True, exist_ok=True)

    rank = os.environ.get("NODE_RANK", "0")
    rank_mapping_path = remote_run_dir / f"rank_mapping_node{rank}.json"
    rank_mapping_path.write_text(
        json.dumps(
            {
                "node_rank": int(rank),
                "master_addr": os.environ.get("MASTER_ADDR", ""),
                "master_port": os.environ.get("MASTER_PORT", ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rdma_root = Path(f"/sys/class/net/{os.environ.get('NCCL_SOCKET_IFNAME', 'net1')}/device/infiniband")
    if rdma_root.exists():
        devices = sorted(x.name for x in rdma_root.iterdir())
        if devices:
            os.environ["NCCL_IB_HCA"] = devices[0]
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_IB_GID_INDEX"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"
    os.environ["NCCL_IB_TC"] = "136"

    megatron_args = build_megatron_args(
        resolved=resolved,
        mock_mode=load_mock_mode(),
        profile_mode=args.profile_mode,
        tokenizer_path=os.environ.get("TOKENIZER_PATH", ""),
    )
    quoted = " ".join(shlex.quote(x) for x in megatron_args)
    torchrun_cmd = (
        f"{os.environ.get('TORCHRUN_BIN', 'torchrun')} "
        f"--nnodes={os.environ.get('NNODES', '2')} "
        f"--nproc_per_node={os.environ.get('NPROC_PER_NODE', '8')} "
        f"--node_rank={os.environ['NODE_RANK']} "
        f"--master_addr={os.environ['MASTER_ADDR']} "
        f"--master_port={os.environ['RUN_MASTER_PORT']} "
        f"{os.environ['REMOTE_PROJECT_ROOT']}/perf_pp_vpp/scripts/torchrun_rank_entry.py "
        f"--entry {os.environ['REMOTE_PROJECT_ROOT']}/pretrain_gpt.py "
        f"--profile-mode {args.profile_mode} "
        f"--profile-ranks {args.profile_ranks} "
        f"--nsys-bin {os.environ.get('NSYS_BIN', 'nsys')} "
        f"--nsys-output-prefix {shlex.quote(str(remote_run_dir / 'nsys' / resolved['experiment']['name']))} "
        f"--patch-bootstrap perf_pp_vpp.megatron_patches.nvtx_instrumentation "
        f"-- {quoted}"
    )
    (remote_run_dir / "torchrun_command.txt").write_text(torchrun_cmd + "\n", encoding="utf-8")

    stdout_path = remote_run_dir / f"megatron_stdout_node{rank}.log"
    stderr_path = remote_run_dir / f"megatron_stderr_node{rank}.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        proc = subprocess.run(
            ["bash", "-lc", f"stdbuf -oL -eL {torchrun_cmd}"],
            cwd=os.environ["REMOTE_PROJECT_ROOT"],
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=os.environ.copy(),
            text=True,
            check=False,
        )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
