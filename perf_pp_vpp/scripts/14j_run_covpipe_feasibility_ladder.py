#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path("/root/chenny-megatron-clean")
DEFAULT_EXPERIMENT_YAML = "perf_pp_vpp/configs/experiments/phase2_qwen32b.yaml"
DEFAULT_EXPERIMENT_NAME = "q32_pp8_dp1_tp2_vpp4"
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "perf_pp_vpp/outputs/covpipe_shortlist/q32_pp8_vpp4/"
    / "covpipe_left_plus__head_heavy__recv_wait__recv_wait.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--experiment-yaml", default=DEFAULT_EXPERIMENT_YAML)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--campaign-name", default=f"q32_covpipe_ladder_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--repeat-base", type=int, default=7100)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 384, 256])
    parser.add_argument("--recompute-num-layers", type=int, nargs="+", default=[2, 1])
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=6)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=2)
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)


def clean_remote_workers(sleep_seconds: int) -> None:
    remote_cleanup = (
        "pkill -9 -f torchrun >/dev/null 2>&1 || true; "
        "pkill -9 -f torchrun_rank_entry.py >/dev/null 2>&1 || true; "
        "pkill -9 -f pretrain_gpt.py >/dev/null 2>&1 || true; "
        "pkill -9 -f 'nsys profile' >/dev/null 2>&1 || true"
    )
    base = [
        "timeout",
        "20s",
        "bash",
        "/usr/local/bin/kubectl",
        "--insecure-skip-tls-verify=true",
        "exec",
        "-n",
        "default",
    ]
    for pod in ("chenny-dist-0", "chenny-dist-1"):
        run(base + [pod, "--", "sh", "-lc", remote_cleanup], os.environ.copy())
    time.sleep(sleep_seconds)


def run_dir(campaign_root: Path, experiment_name: str, seq_len: int, repeat_id: int, run_tag: str) -> Path:
    return campaign_root / "runs" / experiment_name / f"seq{seq_len}" / f"repeat{repeat_id}" / f"baseline__{run_tag}"


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {"num_rows": 0, "oom": False, "errors": [f"missing {summary_path}"]}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def is_success(summary: dict) -> bool:
    return bool(summary.get("num_rows")) and not summary.get("oom") and summary.get("tflops_mean") is not None


def launch_baseline(env: dict[str, str], seq_len: int, repeat_id: int) -> None:
    run(
        [
            "bash",
            "perf_pp_vpp/scripts/03_run_megatron_experiment.sh",
            "--experiment-yaml",
            env["PERF_EXPERIMENT_YAML"],
            "--experiment-name",
            env["PERF_EXPERIMENT_NAME"],
            "--seq-len",
            str(seq_len),
            "--repeat-id",
            str(repeat_id),
            "--overlap-mode",
            "baseline",
            "--profile-mode",
            "none",
            "--profile-ranks",
            "none",
        ],
        env,
    )


def launch_candidate(env: dict[str, str], candidate_json: str, seq_len: int, repeat_id: int) -> None:
    run(
        [
            "bash",
            "perf_pp_vpp/scripts/14c_run_covpipe_candidate.sh",
            candidate_json,
            env["PERF_EXPERIMENT_YAML"],
            env["PERF_EXPERIMENT_NAME"],
            str(seq_len),
            str(repeat_id),
            "baseline",
        ],
        env,
    )


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_json)
    if not candidate_path.exists():
        raise SystemExit(f"candidate does not exist: {candidate_path}")

    campaign_root = PROJECT_ROOT / "perf_pp_vpp/outputs/campaigns" / args.campaign_name
    remote_campaign_root = Path("/workspace/code/Megatron-LM/perf_pp_vpp/outputs/campaigns") / args.campaign_name
    campaign_root.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_root / "data_manifest.json"
    shared_manifest = PROJECT_ROOT / "perf_pp_vpp/outputs/data_manifest.json"

    base_env = os.environ.copy()
    base_env.update(
        {
            "CAMPAIGN_NAME": args.campaign_name,
            "OUTPUT_ROOT": str(campaign_root),
            "DATA_ROOT": str(campaign_root / "data"),
            "CHECKPOINT_ROOT": str(campaign_root / "checkpoints"),
            "PERF_RUNS_ROOT_LOCAL": str(campaign_root / "runs"),
            "PERF_RUNS_ROOT_REMOTE": str(remote_campaign_root / "runs"),
            "PERF_EXPERIMENT_YAML": args.experiment_yaml,
            "PERF_EXPERIMENT_NAME": args.experiment_name,
            "PERF_GLOBAL_BATCH_SIZE": str(args.global_batch_size),
            "PERF_MICRO_BATCH_SIZE": str(args.micro_batch_size),
            "PERF_TRAIN_ITERS": str(args.train_iters),
            "PERF_WARMUP_ITERS": str(args.warmup_iters),
            "PERF_PROFILE_START_ITER": str(args.warmup_iters + 1),
            "PERF_PROFILE_END_ITER": str(args.train_iters),
            "PERF_TRANSFORMER_IMPL": base_env.get("PERF_TRANSFORMER_IMPL", "transformer_engine"),
            "PERF_USE_PRECISION_AWARE_OPTIMIZER": "0",
            "PERF_ENABLE_DELAY_WGRAD_COVPIPE": "0",
            "KUBECTL_BIN": "bash /usr/local/bin/kubectl --insecure-skip-tls-verify=true",
        }
    )

    if not manifest_path.exists():
        if shared_manifest.exists():
            shutil.copy2(shared_manifest, manifest_path)
        else:
            run(["bash", "perf_pp_vpp/scripts/02_prepare_mock_data.sh"], base_env)

    attempts: list[dict] = []
    repeat_id = args.repeat_base
    for recompute_layers in args.recompute_num_layers:
        for seq_len in args.seq_lens:
            clean_remote_workers(args.sleep_seconds)
            baseline_env = base_env | {
                "PERF_RECOMPUTE_GRANULARITY": "full",
                "PERF_RECOMPUTE_METHOD": "uniform",
                "PERF_RECOMPUTE_NUM_LAYERS": str(recompute_layers),
                "PERF_RUN_TAG": f"uniform_rc{recompute_layers}",
            }
            launch_baseline(baseline_env, seq_len, repeat_id)
            baseline_summary = load_summary(
                run_dir(campaign_root, args.experiment_name, seq_len, repeat_id, baseline_env["PERF_RUN_TAG"])
                / "step_summary.json"
            )
            attempts.append(
                {
                    "kind": "baseline",
                    "seq_len": seq_len,
                    "recompute_num_layers": recompute_layers,
                    "repeat_id": repeat_id,
                    "run_tag": baseline_env["PERF_RUN_TAG"],
                    "summary": baseline_summary,
                }
            )
            repeat_id += 1
            if not is_success(baseline_summary):
                continue

            clean_remote_workers(args.sleep_seconds)
            candidate_env = baseline_env | {"PERF_RUN_TAG": f"covpipe_rc{recompute_layers}"}
            launch_candidate(candidate_env, str(candidate_path), seq_len, repeat_id)
            candidate_summary = load_summary(
                run_dir(campaign_root, args.experiment_name, seq_len, repeat_id, candidate_env["PERF_RUN_TAG"])
                / "step_summary.json"
            )
            attempts.append(
                {
                    "kind": "candidate",
                    "candidate": candidate_path.stem,
                    "seq_len": seq_len,
                    "recompute_num_layers": recompute_layers,
                    "repeat_id": repeat_id,
                    "run_tag": candidate_env["PERF_RUN_TAG"],
                    "summary": candidate_summary,
                }
            )
            repeat_id += 1
            if is_success(candidate_summary):
                break
        else:
            continue
        break

    result = {
        "campaign_root": str(campaign_root),
        "candidate_json": str(candidate_path),
        "attempts": attempts,
    }
    output_path = campaign_root / "covpipe_ladder_summary.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
