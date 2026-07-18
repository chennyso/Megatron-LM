#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "experiments/bbt_16gpu/configs/observation_16gpu.json"
RENDERER = REPO_ROOT / "experiments/bbt_16gpu/scripts/render_dual_node_observation.py"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def job_name(run_id: str, case_id: str, repeat_id: int) -> str:
    import re

    value = f"obs16-{run_id}-{case_id}-r{repeat_id}".lower()
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return re.sub(r"-+", "-", cleaned)[:50].rstrip("-")


def kubectl_json(args: list[str]) -> dict:
    result = subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def wait_for_terminal(name: str, timeout_s: int, pending_timeout_s: int) -> str:
    started = time.time()
    first_running = None
    while time.time() - started < timeout_s:
        payload = kubectl_json(
            ["get", "job.batch.volcano.sh", name, "-n", "default", "-o", "json"]
        )
        phase = payload.get("status", {}).get("state", {}).get("phase", "Pending")
        print(f"job={name} phase={phase}", flush=True)
        if phase in {"Completed", "Failed", "Aborted", "Terminated"}:
            return phase
        if phase == "Running" and first_running is None:
            first_running = time.time()
        if first_running is None and time.time() - started > pending_timeout_s:
            events = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "events",
                    "-n",
                    "default",
                    "--sort-by=.lastTimestamp",
                ],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
            raise RuntimeError(f"job {name} remained pending\n{events[-8000:]}")
        time.sleep(15)
    raise TimeoutError(f"job {name} exceeded timeout {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--phase", choices=["screening", "diagnostic", "tuning", "all"], default="screening")
    parser.add_argument("--profile-mode", choices=["throughput", "trace", "nsys"], default="throughput")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--pending-timeout-s", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-jobs", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repeat_key = f"{args.profile_mode}_repeats"
    repeats = args.repeats or config["repeat_policy"][repeat_key]
    seed = args.seed or config["repeat_policy"]["randomization_seed"]
    case_ids = [
        case_id
        for case_id, case in config["cases"].items()
        if args.phase == "all" or case["phase"] == args.phase
    ]
    work = [(case_id, repeat_id) for repeat_id in range(1, repeats + 1) for case_id in case_ids]
    random.Random(seed).shuffle(work)

    order_path = REPO_ROOT / "experiments/bbt_16gpu/results" / args.run_id / f"order_{args.profile_mode}.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    order_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "git_ref": args.git_ref,
                "phase": args.phase,
                "profile_mode": args.profile_mode,
                "seed": seed,
                "work": [{"case_id": case, "repeat_id": repeat} for case, repeat in work],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(order_path)
    if args.dry_run:
        for case_id, repeat_id in work:
            print(case_id, repeat_id)
        return 0

    for case_id, repeat_id in work:
        name = job_name(args.run_id, case_id, repeat_id)
        command = [
            "python3",
            str(RENDERER),
            "--config",
            str(args.config),
            "--case-id",
            case_id,
            "--run-id",
            args.run_id,
            "--repeat-id",
            str(repeat_id),
            "--git-ref",
            args.git_ref,
            "--profile-mode",
            args.profile_mode,
            "--apply",
        ]
        subprocess.run(command, check=True)
        phase = wait_for_terminal(name, args.timeout_s, args.pending_timeout_s)
        if phase != "Completed":
            raise RuntimeError(f"job {name} ended in phase {phase}")
        if not args.keep_jobs:
            subprocess.run(
                ["kubectl", "delete", "job.batch.volcano.sh", name, "-n", "default"],
                check=False,
            )
            subprocess.run(
                ["kubectl", "delete", "service", f"{name}-master", "-n", "default"],
                check=False,
            )
            subprocess.run(
                ["kubectl", "delete", "configmap", f"{name}-launcher", "-n", "default"],
                check=False,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
