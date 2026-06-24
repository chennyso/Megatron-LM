#!/usr/bin/env python3
"""Run the SeamPipe paper experiment matrix on BBT g5+g6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "experiments" / "bbt_16gpu"
RESULTS_ROOT = EXP_ROOT / "results"
PROFILE_TEMPLATE = EXP_ROOT / "volcano-megatron-seampipe-16gpu.yaml"
FINAL_TEMPLATE = EXP_ROOT / "volcano-megatron-seampipe-16gpu-final.yaml"
KUBECTL_ENV_PREFIX = [
    "env",
    "-u",
    "http_proxy",
    "-u",
    "https_proxy",
    "-u",
    "all_proxy",
    "-u",
    "HTTP_PROXY",
    "-u",
    "HTTPS_PROXY",
    "-u",
    "ALL_PROXY",
    "kubectl",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    run_slug: str
    hf_model_ckpt: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    default_vp: int
    seq_length: int = 4096
    max_position_embeddings: int = 40960
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 4
    micro_batch_size: int = 1
    global_batch_size: int = 16


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-8B": ModelSpec(
        key="qwen3-8B",
        run_slug="qwen3-8b",
        hf_model_ckpt="/models/qwen3-8B",
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        num_query_groups=8,
        default_vp=3,
    ),
    "qwen3-14B": ModelSpec(
        key="qwen3-14B",
        run_slug="qwen3-14b",
        hf_model_ckpt="/models/qwen3-14B",
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=40,
        num_query_groups=8,
        default_vp=5,
    ),
    "qwen3-32B": ModelSpec(
        key="qwen3-32B",
        run_slug="qwen3-32b",
        hf_model_ckpt="/models/qwen3-32B",
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=25600,
        num_attention_heads=64,
        num_query_groups=8,
        default_vp=4,
    ),
}

PROFILE_STRATEGY_BLOCK = """--num-layers-per-virtual-pipeline-stage 3
              --pipeline-strategy-policy default
              --pipeline-strategy-runtime fixed
              --pipeline-strategy-profile-steps 8
              --pipeline-strategy-trace-path /workspace/runs/seampipe/qwen3-8b/final/traces/rank{pp_rank}.json"""

FINAL_STRATEGY_BLOCK = """--num-layers-per-virtual-pipeline-stage 3
              --pipeline-strategy-policy default
              --pipeline-strategy-runtime fixed
              --pipeline-strategy-profile-steps 8
              --pipeline-strategy-trace-path /workspace/runs/seampipe/qwen3-8b/final-plan/traces/rank{pp_rank}.json"""


@dataclass
class JobResult:
    job_name: str
    app_label: str
    pods: list[str]
    phases: dict[str, str]
    logs_dir: Path
    metrics: dict[str, Any]


def run(cmd: list[str], *, check: bool = True, capture: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=text)


def kubectl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return run(KUBECTL_ENV_PREFIX + args, check=check)


def kubectl_json(args: list[str]) -> Any:
    return json.loads(kubectl(args).stdout)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def short_job_token(text: str, limit: int = 16) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
    base = cleaned[: max(limit - 7, 1)].rstrip("-") if cleaned else "run"
    return f"{base}-{digest}"


def current_branch() -> str:
    return run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def replace_all(text: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text


def build_strategy_block(
    *,
    layers_per_virtual_stage: int | None,
    trace_dir: str | None,
) -> str:
    if layers_per_virtual_stage is None:
        return ""
    block = [
        f"--num-layers-per-virtual-pipeline-stage {layers_per_virtual_stage}",
        "--pipeline-strategy-policy default",
        "--pipeline-strategy-runtime fixed",
        "--pipeline-strategy-profile-steps 8",
    ]
    if trace_dir is not None:
        block.append(f"--pipeline-strategy-trace-path {trace_dir}/rank{{pp_rank}}.json")
    return "\n              ".join(block)


def render_manifest(template_path: Path, replacements: dict[str, str]) -> Path:
    rendered = replace_all(template_path.read_text(encoding="utf-8"), replacements)
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        handle.write(rendered)
        handle.flush()
        return Path(handle.name)
    finally:
        handle.close()


def template_replacements(
    *,
    model_spec: ModelSpec,
    workspace_claim: str,
    model_claim: str,
    hf_model_ckpt: str,
    run_root_from: str,
    run_root_to: str,
    strategy_block_from: str,
    strategy_block_to: str,
    branch: str,
) -> dict[str, str]:
    return {
        run_root_from: run_root_to,
        "claimName: chenny-workspace": f"claimName: {workspace_claim}",
        "claimName: chenny-models-nfs": f"claimName: {model_claim}",
        "/workspace/models/qwen3-8B": hf_model_ckpt,
        "value: codex/agentpipe-vp-search": f"value: {branch}",
        "--num-layers 36": f"--num-layers {model_spec.num_layers}",
        "--hidden-size 4096": f"--hidden-size {model_spec.hidden_size}",
        "--ffn-hidden-size 12288": f"--ffn-hidden-size {model_spec.ffn_hidden_size}",
        "--num-attention-heads 32": f"--num-attention-heads {model_spec.num_attention_heads}",
        "--num-query-groups 8": f"--num-query-groups {model_spec.num_query_groups}",
        "--tensor-model-parallel-size 4": f"--tensor-model-parallel-size {model_spec.tensor_parallel_size}",
        "--pipeline-model-parallel-size 4": f"--pipeline-model-parallel-size {model_spec.pipeline_parallel_size}",
        "--micro-batch-size 1": f"--micro-batch-size {model_spec.micro_batch_size}",
        "--global-batch-size 16": f"--global-batch-size {model_spec.global_batch_size}",
        "--seq-length 4096": f"--seq-length {model_spec.seq_length}",
        "--max-position-embeddings 40960": f"--max-position-embeddings {model_spec.max_position_embeddings}",
        "--tokenizer-model /workspace/models/qwen3-8B": f"--tokenizer-model {hf_model_ckpt}",
        strategy_block_from: strategy_block_to,
    }


def pod_names_for_label(app_label: str) -> list[str]:
    payload = kubectl_json(["get", "pods", "-n", "default", "-l", f"app={app_label}", "-o", "json"])
    return [item["metadata"]["name"] for item in payload.get("items", [])]


def wait_for_pods(app_label: str, replicas: int, timeout_s: int) -> list[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pods = pod_names_for_label(app_label)
        if len(pods) >= replicas:
            return pods
        time.sleep(5)
    raise TimeoutError(f"pods for app={app_label} did not appear within {timeout_s}s")


def wait_for_job_terminal(app_label: str, timeout_s: int) -> tuple[list[str], dict[str, str]]:
    deadline = time.time() + timeout_s
    seen_pods: set[str] = set()
    while time.time() < deadline:
        payload = kubectl_json(["get", "pods", "-n", "default", "-l", f"app={app_label}", "-o", "json"])
        items = payload.get("items", [])
        if items:
            phases = {item["metadata"]["name"]: item["status"]["phase"] for item in items}
            seen_pods.update(phases)
            if all(phase in {"Succeeded", "Failed"} for phase in phases.values()):
                return sorted(seen_pods), phases
        elif seen_pods:
            return sorted(seen_pods), {}
        time.sleep(10)
    raise TimeoutError(f"pods for app={app_label} did not finish within {timeout_s}s")


def current_phases(app_label: str) -> dict[str, str]:
    payload = kubectl_json(["get", "pods", "-n", "default", "-l", f"app={app_label}", "-o", "json"])
    return {item["metadata"]["name"]: item["status"]["phase"] for item in payload.get("items", [])}


def collect_workspace_logs(inspect_pod: str, remote_log_dir: str, logs_dir: Path) -> list[str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    find_cmd = (
        f"find {remote_log_dir} -maxdepth 1 -type f -name 'launcher*.log' | sort"
    )
    result = kubectl(
        ["exec", "-n", "default", inspect_pod, "--", "bash", "-lc", find_cmd],
        check=True,
    )
    remote_logs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for remote_path in remote_logs:
        content = kubectl(
            ["exec", "-n", "default", inspect_pod, "--", "bash", "-lc", f"cat {remote_path}"],
            check=True,
        ).stdout
        (logs_dir / Path(remote_path).name).write_text(content, encoding="utf-8")
    return remote_logs


def extract_metrics_from_logs(logs_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "iter1_ms": None,
        "after_training_done": False,
        "validation_line": None,
        "test_line": None,
        "error_markers": [],
    }
    err_markers = ["Traceback", "IndexError", "ChildFailedError", "RuntimeError"]
    iter_re = re.compile(r"iteration\s+1/\s*20 .*?elapsed time per iteration \(ms\):\s*([0-9.]+)")
    for log_path in sorted(logs_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if metrics["iter1_ms"] is None:
            match = iter_re.search(text)
            if match:
                metrics["iter1_ms"] = float(match.group(1))
        if "[after training is done]" in text:
            metrics["after_training_done"] = True
        if metrics["validation_line"] is None:
            match = re.search(r"validation loss at iteration 20 on validation set .*", text)
            if match:
                metrics["validation_line"] = match.group(0)
        if metrics["test_line"] is None:
            match = re.search(r"validation loss at iteration 20 on test set .*", text)
            if match:
                metrics["test_line"] = match.group(0)
        for marker in err_markers:
            if marker in text:
                metrics["error_markers"].append({"log": log_path.name, "marker": marker})
    return metrics


def cleanup_job(job_name: str, service_name: str) -> None:
    kubectl(["delete", "job.batch.volcano.sh", job_name, "-n", "default"], check=False)
    kubectl(["delete", "service", service_name, "-n", "default"], check=False)


def ensure_inspect_repo(pod_name: str, branch: str) -> str:
    repo_dir = "/workspace/repos/Megatron-LM-paper-matrix"
    script = f"""
set -euo pipefail
export GIT_SSL_NO_VERIFY=1
if [ ! -d "{repo_dir}/.git" ]; then
  git clone https://github.com/chennyso/Megatron-LM.git "{repo_dir}"
fi
cd "{repo_dir}"
git fetch origin
git checkout -q {branch}
git reset --hard origin/{branch} >/dev/null
printf '%s' "{repo_dir}"
"""
    result = kubectl(
        ["exec", "-n", "default", pod_name, "--", "bash", "-lc", script],
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def run_search(
    pod_name: str,
    branch: str,
    trace_glob: str,
    output_dir: str,
    runtime: str,
    model_spec: ModelSpec,
) -> dict[str, Any]:
    repo_dir = ensure_inspect_repo(pod_name, branch)
    cmd = f"""
set -euo pipefail
cd "{repo_dir}"
python3 tools/run_bcp_vpp_loop.py \
  --trace {trace_glob} \
  --output-dir {output_dir} \
  --num-microbatches 16 \
  --num-model-chunks {model_spec.default_vp} \
  --microbatch-group-size 4 \
  --pipeline-parallel-size {model_spec.pipeline_parallel_size} \
  --num-layers {model_spec.num_layers} \
  --runtime {runtime} \
  --objective bcp \
  --candidate-budget 24 \
  --bcp-activation-budget-mb 28000 \
  --bcp-p2p-credit-budget 2 \
  --bcp-fb-delay-budget 24
"""
    result = kubectl(["exec", "-n", "default", pod_name, "--", "bash", "-lc", cmd], check=True)
    manifest = json.loads(result.stdout)
    manifest["runtime"] = runtime
    return manifest


def submit_and_wait(
    manifest_path: Path,
    app_label: str,
    job_name: str,
    service_name: str,
    inspect_pod: str,
    remote_log_dir: str,
    logs_dir: Path,
    timeout_s: int,
) -> JobResult:
    kubectl(["apply", "-f", str(manifest_path)])
    wait_for_pods(app_label, replicas=2, timeout_s=180)
    pods, terminal_phases = wait_for_job_terminal(app_label, timeout_s=timeout_s)
    remote_logs = collect_workspace_logs(inspect_pod, remote_log_dir, logs_dir)
    phases = current_phases(app_label) or terminal_phases
    metrics = extract_metrics_from_logs(logs_dir)
    metrics["remote_logs"] = remote_logs
    cleanup_job(job_name, service_name)
    return JobResult(job_name=job_name, app_label=app_label, pods=pods, phases=phases, logs_dir=logs_dir, metrics=metrics)


def build_profile_manifest(
    run_root: str,
    branch: str,
    batch_token: str,
    trial_tag: str,
    model_spec: ModelSpec,
    profile_dirname: str,
    enable_strategy_profile: bool,
    workspace_claim: str,
    model_claim: str,
    hf_model_ckpt: str,
) -> tuple[Path, str, str, str]:
    job_name = f"spm-{batch_token}-{trial_tag}-prof"
    service_name = f"{job_name}-svc"
    app_label = job_name
    layers_per_virtual_stage = None
    if enable_strategy_profile:
        layers_per_virtual_stage = model_spec.num_layers // (
            model_spec.pipeline_parallel_size * model_spec.default_vp
        )
    strategy_block = build_strategy_block(
        layers_per_virtual_stage=layers_per_virtual_stage,
        trace_dir=f"{run_root}/{profile_dirname}/traces" if enable_strategy_profile else None,
    )
    manifest = render_manifest(
        PROFILE_TEMPLATE,
        {
            "seampipe-megatron-svc": service_name,
            "seampipe-megatron-16gpu": job_name,
            "seampipe-megatron": app_label,
            "/workspace/runs/seampipe/qwen3-8b/plans/best-plan.json": f"{run_root}/plans/profile-placeholder.json",
            **template_replacements(
                model_spec=model_spec,
                workspace_claim=workspace_claim,
                model_claim=model_claim,
                hf_model_ckpt=hf_model_ckpt,
                run_root_from="/workspace/runs/seampipe/qwen3-8b/final",
                run_root_to=f"{run_root}/{profile_dirname}",
                strategy_block_from=PROFILE_STRATEGY_BLOCK,
                strategy_block_to=strategy_block,
                branch=branch,
            ),
        },
    )
    return manifest, app_label, job_name, service_name


def build_final_manifest(
    run_root: str,
    branch: str,
    batch_token: str,
    trial_tag: str,
    plan_path: str,
    final_dirname: str,
    job_suffix: str,
    model_spec: ModelSpec,
    workspace_claim: str,
    model_claim: str,
    hf_model_ckpt: str,
) -> tuple[Path, str, str, str]:
    job_name = f"spm-{batch_token}-{trial_tag}-{job_suffix}"
    service_name = f"{job_name}-svc"
    app_label = job_name
    manifest = render_manifest(
        FINAL_TEMPLATE,
        {
            "seampipe-megatron-final-svc": service_name,
            "seampipe-megatron-16gpu-final": job_name,
            "seampipe-megatron-final": app_label,
            "/workspace/runs/seampipe/qwen3-8b/plans/best-plan.json": plan_path,
            **template_replacements(
                model_spec=model_spec,
                workspace_claim=workspace_claim,
                model_claim=model_claim,
                hf_model_ckpt=hf_model_ckpt,
                run_root_from="/workspace/runs/seampipe/qwen3-8b/final-plan",
                run_root_to=f"{run_root}/{final_dirname}",
                strategy_block_from=FINAL_STRATEGY_BLOCK,
                strategy_block_to=FINAL_STRATEGY_BLOCK.replace(
                    "/workspace/runs/seampipe/qwen3-8b/final-plan",
                    f"{run_root}/{final_dirname}",
                ),
                branch=branch,
            ),
        },
    )
    return manifest, app_label, job_name, service_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default=current_branch())
    parser.add_argument("--inspect-pod", default="seampipe-workspace-inspect-live")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-minutes", type=int, default=45)
    parser.add_argument("--prefix", default="paper-qwen3-8b")
    parser.add_argument("--model-key", default="qwen3-8B", choices=sorted(MODEL_SPECS))
    parser.add_argument("--workspace-claim", default="chenny-workspace")
    parser.add_argument("--model-claim", default="chenny-models-nfs")
    parser.add_argument("--hf-model-ckpt", default=None)
    parser.add_argument("--run-1f1b-baseline", action="store_true")
    args = parser.parse_args()
    model_spec = MODEL_SPECS[args.model_key]
    hf_model_ckpt = args.hf_model_ckpt or model_spec.hf_model_ckpt

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    batch_tag = f"{args.prefix}-{now_tag()}"
    batch_root = RESULTS_ROOT / batch_tag
    batch_root.mkdir(parents=True, exist_ok=True)
    timeout_s = args.timeout_minutes * 60
    batch_token = short_job_token(batch_tag)

    batch_summary: dict[str, Any] = {
        "batch_tag": batch_tag,
        "branch": args.branch,
        "model_key": model_spec.key,
        "hf_model_ckpt": hf_model_ckpt,
        "inspect_pod": args.inspect_pod,
        "repeats": args.repeats,
        "trials": [],
    }

    for idx in range(1, args.repeats + 1):
        trial_tag = f"{batch_tag}-r{idx:02d}"
        local_trial_dir = batch_root / f"r{idx:02d}"
        local_trial_dir.mkdir(parents=True, exist_ok=True)
        run_root = f"/workspace/runs/seampipe/paper/{model_spec.run_slug}/{trial_tag}"

        trial_summary: dict[str, Any] = {
            "trial_tag": trial_tag,
            "run_root": run_root,
        }

        profile_manifest, profile_app, profile_job, profile_svc = build_profile_manifest(
            run_root,
            args.branch,
            batch_token,
            f"{idx:02d}-p",
            model_spec,
            "profile",
            True,
            args.workspace_claim,
            args.model_claim,
            hf_model_ckpt,
        )
        profile_result = submit_and_wait(
            profile_manifest,
            profile_app,
            profile_job,
            profile_svc,
            args.inspect_pod,
            f"{run_root}/profile",
            local_trial_dir / "profile-logs",
            timeout_s,
        )
        trial_summary["profile"] = {
            "job_name": profile_result.job_name,
            "pods": profile_result.pods,
            "phases": profile_result.phases,
            "metrics": profile_result.metrics,
        }
        if not profile_result.metrics["iter1_ms"]:
            write_json(local_trial_dir / "trial-summary.json", trial_summary)
            batch_summary["trials"].append(trial_summary)
            continue

        if args.run_1f1b_baseline:
            onef1b_manifest, onef1b_app, onef1b_job, onef1b_svc = build_profile_manifest(
                run_root,
                args.branch,
                batch_token,
                f"{idx:02d}-b",
                model_spec,
                "baseline-1f1b",
                False,
                args.workspace_claim,
                args.model_claim,
                hf_model_ckpt,
            )
            onef1b_result = submit_and_wait(
                onef1b_manifest,
                onef1b_app,
                onef1b_job,
                onef1b_svc,
                args.inspect_pod,
                f"{run_root}/baseline-1f1b",
                local_trial_dir / "baseline-1f1b-logs",
                timeout_s,
            )
            trial_summary["baseline_1f1b"] = {
                "job_name": onef1b_result.job_name,
                "pods": onef1b_result.pods,
                "phases": onef1b_result.phases,
                "metrics": onef1b_result.metrics,
            }

        search_fixed = run_search(
            args.inspect_pod,
            args.branch,
            f"{run_root}/profile/traces/rank*.json",
            f"{run_root}/search-fixed",
            "fixed",
            model_spec,
        )
        search_bcp_ready = run_search(
            args.inspect_pod,
            args.branch,
            f"{run_root}/profile/traces/rank*.json",
            f"{run_root}/search-bcp-ready",
            "bcp-ready",
            model_spec,
        )
        trial_summary["search_fixed"] = search_fixed
        trial_summary["search_bcp_ready"] = search_bcp_ready

        final_plan_path = f"{run_root}/search-fixed/best_strategy.json"
        final_manifest, final_app, final_job, final_svc = build_final_manifest(
            run_root,
            args.branch,
            batch_token,
            f"{idx:02d}-f",
            final_plan_path,
            "final-fixed",
            "final",
            model_spec,
            args.workspace_claim,
            args.model_claim,
            hf_model_ckpt,
        )
        final_result = submit_and_wait(
            final_manifest,
            final_app,
            final_job,
            final_svc,
            args.inspect_pod,
            f"{run_root}/final-fixed",
            local_trial_dir / "final-fixed-logs",
            timeout_s,
        )
        trial_summary["final_fixed"] = {
            "job_name": final_result.job_name,
            "pods": final_result.pods,
            "phases": final_result.phases,
            "metrics": final_result.metrics,
        }

        final_bcp_ready_plan_path = f"{run_root}/search-bcp-ready/best_strategy.json"
        final_bcp_ready_manifest, final_bcp_ready_app, final_bcp_ready_job, final_bcp_ready_svc = build_final_manifest(
            run_root,
            args.branch,
            batch_token,
            f"{idx:02d}-r",
            final_bcp_ready_plan_path,
            "final-bcp-ready",
            "ready",
            model_spec,
            args.workspace_claim,
            args.model_claim,
            hf_model_ckpt,
        )
        final_bcp_ready_result = submit_and_wait(
            final_bcp_ready_manifest,
            final_bcp_ready_app,
            final_bcp_ready_job,
            final_bcp_ready_svc,
            args.inspect_pod,
            f"{run_root}/final-bcp-ready",
            local_trial_dir / "final-bcp-ready-logs",
            timeout_s,
        )
        trial_summary["final_bcp_ready"] = {
            "job_name": final_bcp_ready_result.job_name,
            "pods": final_bcp_ready_result.pods,
            "phases": final_bcp_ready_result.phases,
            "metrics": final_bcp_ready_result.metrics,
        }
        write_json(local_trial_dir / "trial-summary.json", trial_summary)
        batch_summary["trials"].append(trial_summary)
        write_json(batch_root / "batch-summary.json", batch_summary)

    write_json(batch_root / "batch-summary.json", batch_summary)
    print(json.dumps({"batch_summary": str(batch_root / "batch-summary.json")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - operational script
        print(f"fatal: {exc}", file=sys.stderr)
        raise
