#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_weights(chunks_per_rank: list[list[int]]) -> dict[int, list[float]]:
    weights: dict[int, list[float]] = {}
    for rank, chunks in enumerate(chunks_per_rank):
        total = sum(chunks)
        weights[rank] = [chunk / total for chunk in chunks]
    return weights


def estimate(candidate: dict[str, Any], baseline_profile: dict[str, Any]) -> dict[str, Any]:
    chunks_per_rank = candidate["chunks_per_pp_rank"]
    weights = chunk_weights(chunks_per_rank)

    send_ns = float(baseline_profile["send_ns"])
    recv_ns = float(baseline_profile["recv_ns"])
    comm_ns = float(baseline_profile["comm_ns"])
    forward_ns = float(baseline_profile["forward_ns"])
    backward_ns = float(baseline_profile["backward_ns"])

    # The boundary ranks are 3 and 4 for pp=8. We use their tail chunk sizes
    # as a proxy for the immediately-available covering window around cross-node
    # communication.
    left_boundary_weight = weights[3][-1]
    right_boundary_weight = weights[4][-1]
    local_cover_forward_ns = forward_ns * (left_boundary_weight + right_boundary_weight) * 0.5
    local_cover_backward_ns = backward_ns * (left_boundary_weight + right_boundary_weight) * 0.5

    shielding_bonus = 0.0
    for placement in candidate.get("annotations", {}).get("wgrad_placements", []):
        target = placement["target"]
        if target == "recv_wait":
            shielding_bonus += 0.10
        elif target == "send_inflight":
            shielding_bonus += 0.08

    effective_cover_ns = (local_cover_forward_ns + local_cover_backward_ns) * (1.0 + shielding_bonus)
    exposed_comm_proxy_ns = max(0.0, comm_ns - effective_cover_ns)
    covering_ratio = 1.0 if comm_ns <= 0 else min(1.0, effective_cover_ns / comm_ns)

    return {
        "candidate_name": candidate["name"],
        "layout": candidate["layout"],
        "counts_per_pp_rank": candidate["counts_per_pp_rank"],
        "chunks_per_pp_rank": chunks_per_rank,
        "estimated_comm_ns": comm_ns,
        "estimated_send_ns": send_ns,
        "estimated_recv_ns": recv_ns,
        "estimated_cover_forward_ns": local_cover_forward_ns,
        "estimated_cover_backward_ns": local_cover_backward_ns,
        "estimated_effective_cover_ns": effective_cover_ns,
        "estimated_exposed_comm_proxy_ns": exposed_comm_proxy_ns,
        "estimated_covering_ratio": covering_ratio,
        "shielding_bonus": shielding_bonus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a COVPipe candidate using a measured baseline profile.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--compiled", action="store_true", help="Candidate input is already compiled output.")
    parser.add_argument("--baseline-profile", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    if not args.compiled:
        import importlib.util

        compiler_path = Path(__file__).with_name("14_generate_covpipe_candidate.py")
        spec = importlib.util.spec_from_file_location("covpipe_compiler", compiler_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load compiler from {compiler_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        candidate = module.compile_candidate(candidate)

    baseline_profile = load_json(args.baseline_profile)
    if "send_ns" not in baseline_profile:
        baseline_profile = json.loads(
            __import__("subprocess").check_output(
                [
                    "python",
                    str(Path(__file__).with_name("14b_score_covpipe_profile.py")),
                    str(args.baseline_profile),
                ],
                text=True,
            )
        )
    print(json.dumps(estimate(candidate, baseline_profile), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
