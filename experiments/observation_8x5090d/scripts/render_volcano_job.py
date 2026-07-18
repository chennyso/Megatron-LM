#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = (
    REPO_ROOT
    / "experiments"
    / "observation_8x5090d"
    / "k8s"
    / "volcano_single_node_8gpu.yaml.tmpl"
)
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
    sriov_network_annotation = "k8s.v1.cni.cncf.io/networks: sriov-ib-network"
    mlnxnics_quantity = "1"
    nccl_ib_disable = "0"
    if args.disable_sriov_ib_network:
        sriov_network_annotation = 'k8s.v1.cni.cncf.io/networks: ""'
        mlnxnics_quantity = "0"
        nccl_ib_disable = "1"
    stage0_toleration = "# stage0-excluded toleration disabled"
    if args.tolerate_stage0_excluded:
        stage0_toleration = (
            "tolerations:\n"
            "        - key: bbt.sspu.edu.cn/stage0-excluded\n"
            "          operator: Equal\n"
            '          value: "true"\n'
            "          effect: NoSchedule"
        )
    repo_dir = f"/workspace/code/Megatron-LM-observation/{args.run_id}"
    if args.phase in {"hardware", "motif"}:
        repo_dir = f"/opt/observation-code/{args.run_id}"

    replacements = {
        "__JOB_NAME__": args.job_name,
        "__NODE_NAME__": args.node,
        "__IMAGE__": args.image,
        "__GPU_RESOURCE_NAME__": args.gpu_resource_name,
        "__GPU_COUNT__": args.gpu_count,
        "__PHASE__": args.phase,
        "__RUN_ID__": args.run_id,
        "__CASE_ID__": args.case_id or "",
        "__MATRIX_PATH__": args.matrix_path,
        "__GIT_REMOTE_URL__": args.git_remote_url,
        "__GIT_REF__": getattr(args, "git_ref", None) or args.git_branch,
        "__REPO_DIR__": repo_dir,
        "__WORKSPACE_PVC__": args.workspace_pvc,
        "__MODEL_PVC__": args.model_pvc,
        "__CPU_REQUEST__": args.cpu_request,
        "__CPU_LIMIT__": args.cpu_limit,
        "__MEM_REQUEST__": args.mem_request,
        "__MEM_LIMIT__": args.mem_limit,
        "__SHM_SIZE__": args.shm_size,
        "__OBS_REPEAT_COUNT_OVERRIDE__": args.obs_repeat_count_override or "",
        "__OBS_SEED_BASE_OVERRIDE__": args.obs_seed_base_override or "",
        "__OBS_MOTIF_TARGET__": getattr(args, "motif_target", "all"),
        "__OBS_MOTIF_NSYS__": "1" if getattr(args, "motif_nsys", False) else "0",
        "__OBS_COMPUTE_COMM_CASE_ID__": getattr(args, "compute_comm_case_id", None) or "",
        "__OBS_COMPUTE_COMM_LOCATIONS__": getattr(args, "compute_comm_locations", None) or "",
        "__SRIOV_NETWORK_ANNOTATION__": sriov_network_annotation,
        "__MLNXNICS_QUANTITY__": mlnxnics_quantity,
        "__NCCL_IB_DISABLE__": nccl_ib_disable,
        "__NCCL_P2P_DISABLE__": args.nccl_p2p_disable,
        "__STAGE0_TOLERATION__": stage0_toleration,
    }
    rendered = TEMPLATE_PATH.read_text(encoding="utf-8")
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["hardware", "motif", "proxy", "baseline", "nsys", "rewrite"],
    )
    parser.add_argument("--node", default="g5")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--matrix-path",
        default="experiments/observation_8x5090d/configs/observation_matrix.json",
    )
    parser.add_argument("--git-branch", default=current_branch())
    parser.add_argument("--git-ref")
    parser.add_argument("--git-remote-url", default="https://github.com/chennyso/Megatron-LM.git")
    parser.add_argument("--workspace-pvc", default="seampipe-paper-workspace")
    parser.add_argument("--model-pvc", default="chenny-models-nfs")
    parser.add_argument("--image", default="harbor.bbt.sspu.edu.cn/nvcr/nvidia/pytorch:26.04-py3")
    parser.add_argument("--gpu-resource-name", default="nvidia.com/gpu")
    parser.add_argument("--gpu-count", default="8")
    parser.add_argument("--cpu-request", default="24")
    parser.add_argument("--cpu-limit", default="48")
    parser.add_argument("--mem-request", default="160Gi")
    parser.add_argument("--mem-limit", default="240Gi")
    parser.add_argument("--shm-size", default="128Gi")
    parser.add_argument("--disable-sriov-ib-network", action="store_true")
    parser.add_argument(
        "--nccl-p2p-disable",
        choices=["0", "1"],
        default="1",
        help="Explicitly select the intra-node NCCL P2P path for matched baselines.",
    )
    parser.add_argument(
        "--tolerate-stage0-excluded",
        action="store_true",
        help="Tolerate the explicit g5/g6 stage0 exclusion without mutating the node taint.",
    )
    parser.add_argument("--obs-repeat-count-override")
    parser.add_argument("--obs-seed-base-override")
    parser.add_argument("--motif-target", choices=["all", "compute-comm"], default="all")
    parser.add_argument("--motif-nsys", action="store_true")
    parser.add_argument("--compute-comm-case-id")
    parser.add_argument(
        "--compute-comm-locations",
        help="Comma-separated sender,receiver,disjoint subset.",
    )
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
