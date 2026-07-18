#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "experiments/bbt_16gpu/configs/observation_16gpu.json"
LAUNCH_SCRIPT = REPO_ROOT / "experiments/bbt_16gpu/scripts/run_dual_node_observation.sh"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dns_label(value: str, limit: int = 50) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    if not cleaned:
        raise ValueError(f"cannot derive a DNS label from {value!r}")
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: limit - len(digest) - 1].rstrip('-')}-{digest}"


def q(value: object) -> str:
    return json.dumps(str(value))


def indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def env_block(values: dict[str, object], spaces: int = 10) -> str:
    prefix = " " * spaces
    lines: list[str] = []
    for name, value in values.items():
        lines.extend(
            [
                f"{prefix}- name: {name}",
                f"{prefix}  value: {q(value)}",
            ]
        )
    return "\n".join(lines)


def render_task(
    *,
    task_name: str,
    node_name: str,
    role: str,
    node_rank: int,
    image: str,
    app: str,
    env: dict[str, object],
    workspace_pvc: str,
    model_pvc: str,
    tolerate_stage0: bool,
) -> str:
    tolerations = ""
    if tolerate_stage0:
        tolerations = """
        tolerations:
        - key: bbt.sspu.edu.cn/stage0-excluded
          operator: Equal
          value: "true"
          effect: NoSchedule"""
    task_env = dict(env)
    task_env["NODE_RANK"] = node_rank
    return f"""  - name: {task_name}
    replicas: 1
    maxRetry: 0
    template:
      metadata:
        labels:
          app: {app}
          observation-role: {role}
        annotations:
          k8s.v1.cni.cncf.io/networks: sriov-ib-network
      spec:
        restartPolicy: Never
        nodeSelector:
          kubernetes.io/hostname: {node_name}{tolerations}
        containers:
        - name: observation
          image: {image}
          imagePullPolicy: IfNotPresent
          command: ["bash", "/opt/observation/run_dual_node_observation.sh"]
          env:
{env_block(task_env, spaces=10)}
          resources:
            requests:
              cpu: "24"
              memory: 160Gi
              nvidia.com/gpu: "8"
              nvidia.com/mlnxnics: "1"
            limits:
              cpu: "48"
              memory: 240Gi
              nvidia.com/gpu: "8"
              nvidia.com/mlnxnics: "1"
          securityContext:
            capabilities:
              add: [IPC_LOCK]
          volumeMounts:
          - name: launcher
            mountPath: /opt/observation
            readOnly: true
          - name: workspace
            mountPath: /workspace
          - name: models
            mountPath: /models
            readOnly: true
          - name: dev-infiniband
            mountPath: /dev/infiniband
          - name: dshm
            mountPath: /dev/shm
        volumes:
        - name: launcher
          configMap:
            name: {app}-launcher
            defaultMode: 493
        - name: workspace
          persistentVolumeClaim:
            claimName: {workspace_pvc}
        - name: models
          persistentVolumeClaim:
            claimName: {model_pvc}
        - name: dev-infiniband
          hostPath:
            path: /dev/infiniband
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 128Gi"""


def render(args: argparse.Namespace) -> str:
    config = load_config(args.config)
    infrastructure = config["infrastructure"]
    workload = config["workload"]
    case = config["cases"].get(args.case_id)
    if case is None:
        raise SystemExit(f"unknown case id: {args.case_id}")
    if args.profile_mode not in {"throughput", "trace", "nsys"}:
        raise SystemExit(f"unsupported profile mode: {args.profile_mode}")

    app = dns_label(f"obs16-{args.run_id}-{args.case_id}-r{args.repeat_id}")
    job_name = app
    service_name = f"{app}-master"
    warmup_steps = args.warmup_steps or workload["warmup_steps"]
    measure_steps = args.measure_steps or workload["measure_steps"]
    env = {
        "CASE_ID": args.case_id,
        "RUN_ID": args.run_id,
        "REPEAT_ID": args.repeat_id,
        "MASTER_ADDR": service_name,
        "MASTER_PORT": args.master_port,
        "GIT_REMOTE": args.git_remote or infrastructure["git_remote"],
        "GIT_REF": args.git_ref,
        "WORKSPACE_ROOT": infrastructure["result_root"],
        "MODEL_PATH": workload["model_path"],
        "DATA_PATH": workload["data_path"],
        "VPP_SIZE": case["vpp_size"],
        "OVERLAP_P2P": int(case["overlap_p2p"]),
        "WARMUP_FLUSH_OVERLAP": int(case["warmup_flush_overlap"]),
        "MICROBATCH_GROUP_SIZE": case["microbatch_group_size"] or 0,
        "PROFILE_MODE": args.profile_mode,
        "WARMUP_STEPS": warmup_steps,
        "MEASURE_STEPS": measure_steps,
        "NCCL_DEBUG_LEVEL": "INFO" if args.profile_mode != "throughput" else "WARN",
    }
    launcher = LAUNCH_SCRIPT.read_text(encoding="utf-8").rstrip()
    g5_task = render_task(
        task_name="worker-g5",
        node_name="g5",
        role="master",
        node_rank=0,
        image=args.image or infrastructure["image"],
        app=app,
        env=env,
        workspace_pvc=args.workspace_pvc or infrastructure["workspace_pvc"],
        model_pvc=args.model_pvc or infrastructure["model_pvc"],
        tolerate_stage0=False,
    )
    g6_task = render_task(
        task_name="worker-g6",
        node_name="g6",
        role="worker",
        node_rank=1,
        image=args.image or infrastructure["image"],
        app=app,
        env=env,
        workspace_pvc=args.workspace_pvc or infrastructure["workspace_pvc"],
        model_pvc=args.model_pvc or infrastructure["model_pvc"],
        tolerate_stage0=True,
    )
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {app}-launcher
  namespace: {infrastructure['namespace']}
data:
  run_dual_node_observation.sh: |-
{indent_block(launcher, 4)}
---
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {infrastructure['namespace']}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    app: {app}
    observation-role: master
  ports:
  - name: torch
    port: {args.master_port}
    targetPort: {args.master_port}
---
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: {job_name}
  namespace: {infrastructure['namespace']}
  labels:
    app: {app}
    observation-phase: {case['phase']}
    observation-mode: {args.profile_mode}
spec:
  minAvailable: 2
  schedulerName: volcano
  queue: default
  maxRetry: 0
  tasks:
{g5_task}
{g6_task}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-id", type=int, required=True)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--profile-mode", default="throughput")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--measure-steps", type=int)
    parser.add_argument("--git-remote")
    parser.add_argument("--image")
    parser.add_argument("--workspace-pvc")
    parser.add_argument("--model-pvc")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest = render(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest, encoding="utf-8")
    if args.apply:
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest,
            text=True,
            check=True,
        )
    elif not args.output:
        sys.stdout.write(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
