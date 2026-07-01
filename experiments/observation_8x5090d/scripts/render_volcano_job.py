#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "experiments" / "observation_8x5090d" / "k8s" / "volcano_single_node_8gpu.yaml.tmpl"
KUBECTL_PREFIX = [
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


def current_branch() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def render(args: argparse.Namespace) -> str:
    replacements = {
        "__JOB_NAME__": args.job_name,
        "__NODE_NAME__": args.node,
        "__IMAGE__": args.image,
        "__PHASE__": args.phase,
        "__RUN_ID__": args.run_id,
        "__CASE_ID__": args.case_id or "",
        "__GIT_REMOTE_URL__": args.git_remote_url,
        "__GIT_BRANCH__": args.git_branch,
        "__WORKSPACE_PVC__": args.workspace_pvc,
        "__MODEL_PVC__": args.model_pvc,
        "__CPU_REQUEST__": args.cpu_request,
        "__CPU_LIMIT__": args.cpu_limit,
        "__MEM_REQUEST__": args.mem_request,
        "__MEM_LIMIT__": args.mem_limit,
        "__SHM_SIZE__": args.shm_size,
    }
    rendered = TEMPLATE_PATH.read_text(encoding="utf-8")
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["hardware", "baseline", "nsys", "rewrite"])
    parser.add_argument("--node", default="g5")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--git-branch", default=current_branch())
    parser.add_argument("--git-remote-url", default="https://github.com/chennyso/Megatron-LM.git")
    parser.add_argument("--workspace-pvc", default="chenny-workspace")
    parser.add_argument("--model-pvc", default="chenny-models-nfs")
    parser.add_argument("--image", default="harbor.bbt.sspu.edu.cn/nvcr/nvidia/pytorch:26.05-py3")
    parser.add_argument("--cpu-request", default="24")
    parser.add_argument("--cpu-limit", default="48")
    parser.add_argument("--mem-request", default="160Gi")
    parser.add_argument("--mem-limit", default="240Gi")
    parser.add_argument("--shm-size", default="128Gi")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest = render(args)
    print(manifest)

    if not args.apply:
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(manifest)
        manifest_path = Path(handle.name)

    subprocess.run(KUBECTL_PREFIX + ["apply", "-f", str(manifest_path)], check=True)
    print(f"\nApplied manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
