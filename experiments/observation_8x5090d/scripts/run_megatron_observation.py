#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_SCRIPT = REPO_ROOT / "experiments" / "observation_8x5090d" / "scripts" / "summarize_megatron_run.py"


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_model_args(model_cfg: dict) -> list[str]:
    args = [
        "--num-layers",
        str(model_cfg["num_layers"]),
        "--hidden-size",
        str(model_cfg["hidden_size"]),
        "--ffn-hidden-size",
        str(model_cfg["ffn_hidden_size"]),
        "--num-attention-heads",
        str(model_cfg["num_attention_heads"]),
        "--num-query-groups",
        str(model_cfg["num_query_groups"]),
        "--kv-channels",
        str(model_cfg["kv_channels"]),
        "--seq-length",
        str(model_cfg["seq_length"]),
        "--max-position-embeddings",
        str(model_cfg["max_position_embeddings"]),
        "--vocab-size",
        str(model_cfg["vocab_size"]),
        "--make-vocab-size-divisible-by",
        str(model_cfg["make_vocab_size_divisible_by"]),
        "--data-cache-path",
        str(REPO_ROOT / "benchmark_cache_observation"),
    ]
    args.extend(model_cfg["common_flags"])
    if "moe" in model_cfg:
        moe = model_cfg["moe"]
        args.extend(
            [
                "--num-experts",
                str(moe["num_experts"]),
                "--moe-ffn-hidden-size",
                str(moe["moe_ffn_hidden_size"]),
                "--moe-router-topk",
                str(moe["moe_router_topk"]),
                "--moe-router-load-balancing-type",
                str(moe["moe_router_load_balancing_type"]),
                "--moe-token-dispatcher-type",
                str(moe["moe_token_dispatcher_type"]),
                "--moe-layer-freq",
                str(moe["moe_layer_freq"]),
            ]
        )
    return args


def build_dataset_args(dataset_cfg: dict, model_cfg: dict) -> list[str]:
    if dataset_cfg["kind"] == "mock":
        return dataset_cfg.get("flags", ["--mock-data", "--tokenizer-type", "NullTokenizer"])

    tokenizer_model = dataset_cfg.get("tokenizer_model") or model_cfg.get("tokenizer_model")
    if not tokenizer_model:
        raise SystemExit(f"Dataset spec {dataset_cfg['id']} requires tokenizer_model.")
    if not dataset_cfg.get("data_path"):
        raise SystemExit(f"Dataset spec {dataset_cfg['id']} requires data_path for non-mock runs.")

    args = [
        "--tokenizer-type",
        dataset_cfg.get("tokenizer_type", "HuggingFaceTokenizer"),
        "--tokenizer-model",
        tokenizer_model,
        "--split",
        dataset_cfg.get("split", "99,1,0"),
        "--data-path",
        dataset_cfg["data_path"],
    ]
    if dataset_cfg.get("extra_flags"):
        args.extend(dataset_cfg["extra_flags"])
    return args


def case_matches(case: dict, phase: str, case_id: str | None) -> bool:
    if case["phase"] != phase:
        return False
    if case_id and case["id"] != case_id:
        return False
    return True


def maybe_start_dmon(repeat_dir: Path) -> tuple[subprocess.Popen | None, object | None]:
    output_path = repeat_dir / "nvidia-smi-dmon.log"
    handle = output_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle


def stop_process(proc: subprocess.Popen | None, handle: object | None) -> None:
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if handle is not None:
        handle.close()


def build_profiler_args(case: dict) -> tuple[list[str], dict]:
    profile_cfg = case.get("profiler_policy") or case.get("profile") or {}
    if not profile_cfg.get("nsys", False):
        return [], profile_cfg

    profile_ranks = profile_cfg.get("profile_ranks", [0])
    if isinstance(profile_ranks, list):
        profile_ranks_value = ",".join(str(rank) for rank in profile_ranks)
    else:
        profile_ranks_value = str(profile_ranks)
    args = [
        "--profile",
        "--profile-step-start",
        str(profile_cfg.get("step_start", 6)),
        "--profile-step-end",
        str(profile_cfg.get("step_end", 8)),
        "--profile-ranks",
        profile_ranks_value,
    ]
    return args, profile_cfg


def export_nsys_stats(repeat_dir: Path) -> None:
    nsys_rep = repeat_dir / "nsys_profile.nsys-rep"
    if not nsys_rep.exists():
        return
    output_prefix = repeat_dir / "nsys_stats"
    cmd = [
        "nsys",
        "stats",
        "--report",
        "gpukernsum,cudaapisum,nvtxsum",
        "--format",
        "csv",
        "--output",
        str(output_prefix),
        str(nsys_rep),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def snapshot_environment(repeat_dir: Path) -> dict:
    payload = {
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": str(REPO_ROOT),
        "python": subprocess.run(["python3", "--version"], capture_output=True, text=True, check=False).stdout.strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_env": {key: value for key, value in os.environ.items() if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_")},
        "selected_env": {
            key: os.environ.get(key)
            for key in [
                "HOSTNAME",
                "KUBERNETES_SERVICE_HOST",
                "NODE_NAME",
                "PHASE",
                "RUN_ID",
                "RESULT_ROOT",
                "REPO_DIR",
            ]
        },
    }
    (repeat_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_case_run_config(case: dict, model_cfg: dict, dataset_cfg: dict, repeat_index: int) -> dict:
    return {
        **case,
        "repeat_index": repeat_index,
        "model_spec": {
            key: model_cfg[key]
            for key in [
                "num_layers",
                "hidden_size",
                "ffn_hidden_size",
                "num_attention_heads",
                "num_query_groups",
                "kv_channels",
                "seq_length",
                "max_position_embeddings",
            ]
        },
        "dataset_spec_detail": dataset_cfg,
    }


def run_case(case: dict, matrix: dict, output_dir: Path) -> None:
    case_dir = output_dir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_config.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

    model_cfg = matrix["models"][case["model"]]
    dataset_cfg = matrix["datasets"][case["dataset_spec"]]
    parallelism = case["parallelism"]
    runtime = case["runtime"]
    repeat_count = case["repeat_policy"]["independent_runs"]
    seed_base = case["repeat_policy"].get("seed_base", 1234)
    warmup_steps = case["warmup_steps"]
    measure_steps = case["measure_steps"]
    train_iters = runtime.get("train_iters", warmup_steps + measure_steps)
    profiler_args, profile_cfg = build_profiler_args(case)

    for repeat_index in range(1, repeat_count + 1):
        repeat_dir = case_dir / f"repeat_{repeat_index:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        run_case_cfg = build_case_run_config(case, model_cfg, dataset_cfg, repeat_index)
        (repeat_dir / "case_config.json").write_text(json.dumps(run_case_cfg, indent=2), encoding="utf-8")
        snapshot_environment(repeat_dir)

        cmd = [
            "torchrun",
            "--nproc_per_node",
            "8",
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(29500 + (int(hashlib.sha1(f'{case["id"]}-{repeat_index}'.encode("utf-8")).hexdigest()[:6], 16) % 3000)),
            "pretrain_gpt.py",
        ]
        cmd.extend(build_model_args(model_cfg))
        cmd.extend(build_dataset_args(dataset_cfg, model_cfg))
        cmd.extend(
            [
                "--tensor-model-parallel-size",
                str(parallelism["tp"]),
                "--pipeline-model-parallel-size",
                str(parallelism["pp"]),
                "--expert-model-parallel-size",
                str(parallelism["ep"]),
                "--micro-batch-size",
                str(runtime["micro_batch_size"]),
                "--global-batch-size",
                str(runtime["global_batch_size"]),
                "--train-iters",
                str(train_iters),
                "--lr",
                str(runtime.get("lr", 1e-4)),
                "--min-lr",
                str(runtime.get("min_lr", 1e-5)),
                "--lr-decay-style",
                runtime.get("lr_decay_style", "cosine"),
                "--weight-decay",
                str(runtime.get("weight_decay", 0.01)),
                "--clip-grad",
                str(runtime.get("clip_grad", 1.0)),
                "--adam-beta1",
                str(runtime.get("adam_beta1", 0.9)),
                "--adam-beta2",
                str(runtime.get("adam_beta2", 0.95)),
                "--seed",
                str(seed_base + repeat_index - 1),
                "--log-interval",
                "1",
                "--timing-log-level",
                "2",
                "--log-throughput",
                "--eval-iters",
                "0",
                "--eval-interval",
                "100000",
                "--save-interval",
                "100000",
                "--distributed-timeout-minutes",
                "90",
            ]
        )

        if parallelism["tp"] > 1:
            cmd.append("--sequence-parallel")
        if case.get("pipeline_model_parallel_layout"):
            cmd.extend(["--pipeline-model-parallel-layout", case["pipeline_model_parallel_layout"]])
        elif case.get("layers_per_virtual_stage"):
            cmd.extend(["--num-layers-per-virtual-pipeline-stage", str(case["layers_per_virtual_stage"])])
        if not case.get("overlap_p2p_comm", False):
            cmd.append("--no-overlap-p2p-communication")

        fsdp = case.get("fsdp", {})
        if fsdp.get("enabled", False):
            cmd.extend(
                [
                    "--use-megatron-fsdp",
                    "--data-parallel-sharding-strategy",
                    fsdp.get("sharding_strategy", "optim_grads_params"),
                    "--no-gradient-accumulation-fusion",
                    "--calculate-per-token-loss",
                    "--init-model-with-meta-device",
                    "--ckpt-format",
                    "fsdp_dtensor",
                    "--grad-reduce-in-bf16",
                ]
            )
        cmd.extend(profiler_args)
        cmd.extend(case.get("extra_flags", []))

        if profile_cfg.get("nsys", False):
            cmd = [
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
                str(repeat_dir / "nsys_profile"),
            ] + cmd

        (repeat_dir / "command.sh").write_text(" ".join(shlex.quote(arg) for arg in cmd) + "\n", encoding="utf-8")

        env = os.environ.copy()
        stdout_path = repeat_dir / "stdout.log"
        stderr_path = repeat_dir / "stderr.log"
        combined_path = repeat_dir / "combined.log"
        dmon_proc, dmon_handle = maybe_start_dmon(repeat_dir)
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            return_code = proc.wait()
        stop_process(dmon_proc, dmon_handle)

        combined_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
        combined_path.write_text(
            combined_text + ("\n\n=== STDERR ===\n" + stderr_text if stderr_text else ""),
            encoding="utf-8",
        )
        (repeat_dir / "return_code.txt").write_text(f"{return_code}\n", encoding="utf-8")
        if profile_cfg.get("nsys", False):
            export_nsys_stats(repeat_dir)

        subprocess.run(
            [
                "python3",
                str(SUMMARY_SCRIPT),
                "--case-id",
                case["id"],
                "--log-path",
                str(combined_path),
                "--dmon-path",
                str(repeat_dir / "nvidia-smi-dmon.log"),
                "--case-config-path",
                str(repeat_dir / "case_config.json"),
                "--output-path",
                str(repeat_dir / "summary.json"),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--phase", required=True, choices=["proxy", "baseline", "nsys", "rewrite"])
    parser.add_argument("--case-id")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    matrix = load_matrix(Path(args.matrix_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix_snapshot.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    matched = [case for case in matrix["cases"] if case_matches(case, args.phase, args.case_id)]
    if not matched:
        raise SystemExit(f"No cases found for phase={args.phase!r} case_id={args.case_id!r}")

    for case in matched:
        run_case(case, matrix, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
