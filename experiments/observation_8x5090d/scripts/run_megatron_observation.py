#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_SCRIPT = REPO_ROOT / "experiments" / "observation_8x5090d" / "scripts" / "summarize_megatron_run.py"


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_args(model_cfg: dict) -> list[str]:
    args = [
        "--num-layers", str(model_cfg["num_layers"]),
        "--hidden-size", str(model_cfg["hidden_size"]),
        "--ffn-hidden-size", str(model_cfg["ffn_hidden_size"]),
        "--num-attention-heads", str(model_cfg["num_attention_heads"]),
        "--num-query-groups", str(model_cfg["num_query_groups"]),
        "--kv-channels", str(model_cfg["kv_channels"]),
        "--seq-length", str(model_cfg["seq_length"]),
        "--max-position-embeddings", str(model_cfg["max_position_embeddings"]),
        "--vocab-size", str(model_cfg["vocab_size"]),
        "--make-vocab-size-divisible-by", str(model_cfg["make_vocab_size_divisible_by"]),
        "--data-cache-path", str(REPO_ROOT / "benchmark_cache_observation"),
    ]
    args.extend(model_cfg["common_flags"])
    if "moe" in model_cfg:
        moe = model_cfg["moe"]
        args.extend(
            [
                "--num-experts", str(moe["num_experts"]),
                "--moe-ffn-hidden-size", str(moe["moe_ffn_hidden_size"]),
                "--moe-router-topk", str(moe["moe_router_topk"]),
                "--moe-router-load-balancing-type", str(moe["moe_router_load_balancing_type"]),
                "--moe-token-dispatcher-type", str(moe["moe_token_dispatcher_type"]),
                "--moe-layer-freq", str(moe["moe_layer_freq"]),
            ]
        )
    return args


def case_matches(case: dict, phase: str, case_id: str | None) -> bool:
    if case["phase"] != phase:
        return False
    if case_id and case["id"] != case_id:
        return False
    return True


def maybe_start_dmon(case_dir: Path) -> subprocess.Popen | None:
    output_path = case_dir / "nvidia-smi-dmon.log"
    handle = output_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_case(case: dict, model_cfg: dict, output_dir: Path) -> None:
    case_dir = output_dir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_config.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

    parallelism = case["parallelism"]
    runtime = case["runtime"]

    args = [
        "torchrun",
        "--nproc_per_node", "8",
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "127.0.0.1",
        "--master_port", str(29500 + (abs(hash(case["id"])) % 3000)),
        "pretrain_gpt.py",
    ]
    args.extend(build_model_args(model_cfg))
    args.extend(
        [
            "--tensor-model-parallel-size", str(parallelism["tp"]),
            "--pipeline-model-parallel-size", str(parallelism["pp"]),
            "--expert-model-parallel-size", str(parallelism["ep"]),
            "--micro-batch-size", str(runtime["micro_batch_size"]),
            "--global-batch-size", str(runtime["global_batch_size"]),
            "--train-iters", str(runtime["train_iters"]),
            "--lr", "1e-4",
            "--min-lr", "1e-5",
            "--lr-decay-style", "cosine",
            "--weight-decay", "0.01",
            "--clip-grad", "1.0",
            "--adam-beta1", "0.9",
            "--adam-beta2", "0.95",
        ]
    )

    if parallelism["tp"] > 1:
        args.append("--sequence-parallel")

    if case.get("pipeline_model_parallel_layout"):
        args.extend(["--pipeline-model-parallel-layout", case["pipeline_model_parallel_layout"]])
    elif case.get("layers_per_virtual_stage"):
        args.extend(["--num-layers-per-virtual-pipeline-stage", str(case["layers_per_virtual_stage"])])

    if not case.get("overlap_p2p_comm", False):
        args.append("--no-overlap-p2p-communication")

    fsdp = case.get("fsdp", {})
    if fsdp.get("enabled", False):
        args.extend(
            [
                "--use-megatron-fsdp",
                "--data-parallel-sharding-strategy", fsdp.get("sharding_strategy", "optim_grads_params"),
                "--no-gradient-accumulation-fusion",
                "--calculate-per-token-loss",
                "--init-model-with-meta-device",
                "--ckpt-format", "fsdp_dtensor",
                "--grad-reduce-in-bf16",
            ]
        )

    profile_cfg = case.get("profile")
    env = os.environ.copy()
    log_path = case_dir / "train.log"

    if profile_cfg and profile_cfg.get("nsys", False):
        args.extend(
            [
                "--profile",
                "--profile-step-start", str(profile_cfg["step_start"]),
                "--profile-step-end", str(profile_cfg["step_end"]),
                "--profile-ranks", "0",
            ]
        )
        prefix = [
            "nsys",
            "profile",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx,cublas,cudnn",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--cuda-memory-usage=true",
            "-f",
            "true",
            "-x",
            "true",
            "-o",
            str(case_dir / "nsys_profile"),
        ]
        cmd = prefix + args
    else:
        cmd = args

    cmd.extend(case.get("extra_flags", []))
    (case_dir / "command.sh").write_text(" ".join(shlex.quote(arg) for arg in cmd) + "\n", encoding="utf-8")

    dmon_proc = maybe_start_dmon(case_dir)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    stop_process(dmon_proc)

    subprocess.run(
        [
            "python3",
            str(SUMMARY_SCRIPT),
            "--case-id",
            case["id"],
            "--log-path",
            str(log_path),
            "--dmon-path",
            str(case_dir / "nvidia-smi-dmon.log"),
            "--output-path",
            str(case_dir / "summary.json"),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--phase", required=True, choices=["baseline", "nsys", "rewrite"])
    parser.add_argument("--case-id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    matrix = load_matrix(Path(args.matrix_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = [case for case in matrix["cases"] if case_matches(case, args.phase, args.case_id)]
    if not matched:
        raise SystemExit(f"No cases found for phase={args.phase!r} case_id={args.case_id!r}")

    for case in matched:
        model_cfg = matrix["models"][case["model"]]
        run_case(case, model_cfg, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
