#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_nvtx(rows: list[dict], prefix: str) -> float:
    for row in rows:
        if row["name"].startswith(prefix):
            return float(row["total_ns"])
    return 0.0


def score_profile(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    nvtx = data.get("nvtx_summary", [])
    send_ns = _find_nvtx(nvtx, "Send")
    recv_ns = _find_nvtx(nvtx, "Recv")
    f_ns = _find_nvtx(nvtx, "task=F|")
    b_ns = _find_nvtx(nvtx, "task=B|")
    comm_ns = send_ns + recv_ns
    covering_window_ns = f_ns + b_ns
    exposed_comm_proxy_ns = max(0.0, comm_ns - covering_window_ns)
    covering_ratio = 1.0 if comm_ns <= 0 else min(1.0, covering_window_ns / comm_ns)

    kernel_breakdown = data.get("kernel_breakdown", [])
    kernel_summary = kernel_breakdown[0] if kernel_breakdown else None
    return {
        "profile": str(path),
        "send_ns": send_ns,
        "recv_ns": recv_ns,
        "comm_ns": comm_ns,
        "forward_ns": f_ns,
        "backward_ns": b_ns,
        "covering_window_ns": covering_window_ns,
        "exposed_comm_proxy_ns": exposed_comm_proxy_ns,
        "covering_ratio": covering_ratio,
        "kernel_breakdown": kernel_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a COVPipe candidate from existing nsys analysis JSON.")
    parser.add_argument("analysis_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_profile(args.analysis_json), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
