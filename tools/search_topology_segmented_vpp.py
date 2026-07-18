#!/usr/bin/env python3
"""Enumerate verified topology-transition budgets for segmented VPP routes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_route_policy():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "megatron" / "core" / "pipeline_parallel" / "route_policy.py"
    spec = importlib.util.spec_from_file_location("topology_route_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_metrics(route, node_by_rank: tuple[str, ...]) -> dict:
    transition_count = len(route.cross_node_edges(node_by_rank))
    endpoint_counts = route.cross_node_endpoint_counts(node_by_rank)
    reuse_gaps = route.rank_reuse_gaps()
    all_gaps = [gap for gaps in reuse_gaps.values() for gap in gaps]
    return {
        "topology_transition_count": transition_count,
        "endpoint_peak_load": max(endpoint_counts.values(), default=0),
        "endpoint_squared_load": sum(count * count for count in endpoint_counts.values()),
        "endpoint_load_by_rank": dict(sorted(endpoint_counts.items())),
        "rank_reuse_pressure": route.rank_reuse_pressure(),
        "minimum_rank_reuse_gap": min(all_gaps) if all_gaps else None,
        "mean_rank_reuse_gap": sum(all_gaps) / len(all_gaps) if all_gaps else None,
    }


def build_candidates(
    pipeline_size: int, virtual_chunks: int, node_by_rank: tuple[str, ...]
) -> list[dict]:
    route_policy = load_route_policy()
    candidates: list[dict] = []
    seen_routes: set[tuple[tuple[int, int], ...]] = set()
    for hierarchy_factor in range(1, virtual_chunks + 1):
        if virtual_chunks % hierarchy_factor != 0:
            continue
        for rotate_endpoints in (False, True):
            route = route_policy.PipelineRoute.topology_segmented(
                pipeline_size,
                virtual_chunks,
                node_by_rank,
                hierarchy_factor,
                rotate_endpoints=rotate_endpoints,
            )
            route.verify()
            route_word = tuple(
                (stage.virtual_chunk, stage.physical_rank) for stage in route.stages
            )
            if route_word in seen_routes:
                continue
            seen_routes.add(route_word)
            name = f"segmented-h{hierarchy_factor}"
            if rotate_endpoints:
                name += "-rotated"
            candidates.append(
                {
                    "name": name,
                    "hierarchy_factor": hierarchy_factor,
                    "rotate_endpoints": rotate_endpoints,
                    **candidate_metrics(route, node_by_rank),
                    "route": [
                        {
                            "virtual_chunk": virtual_chunk,
                            "physical_rank": physical_rank,
                        }
                        for virtual_chunk, physical_rank in route_word
                    ],
                }
            )
    return candidates


def dominates(left: dict, right: dict) -> bool:
    objectives = (
        "topology_transition_count",
        "endpoint_peak_load",
        "rank_reuse_pressure",
    )
    no_worse = all(float(left[key]) <= float(right[key]) for key in objectives)
    strictly_better = any(float(left[key]) < float(right[key]) for key in objectives)
    return no_worse and strictly_better


def pareto_front(candidates: list[dict]) -> list[dict]:
    return [
        candidate
        for candidate in candidates
        if not any(
            dominates(other, candidate)
            for other in candidates
            if other["name"] != candidate["name"]
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-size", type=int, required=True)
    parser.add_argument("--virtual-chunks", type=int, required=True)
    parser.add_argument(
        "--node-by-rank",
        required=True,
        help="Comma-separated topology-domain label for every physical PP rank.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    node_by_rank = tuple(item.strip() for item in args.node_by_rank.split(","))
    if len(node_by_rank) != args.pipeline_size:
        raise ValueError("node-by-rank length must match pipeline-size")
    candidates = build_candidates(
        args.pipeline_size, args.virtual_chunks, node_by_rank
    )
    pareto = pareto_front(candidates)
    payload = {
        "pipeline_size": args.pipeline_size,
        "virtual_chunks": args.virtual_chunks,
        "node_by_rank": node_by_rank,
        "candidate_count": len(candidates),
        "pareto_candidate_names": [candidate["name"] for candidate in pareto],
        "pareto_objectives": [
            "minimize topology_transition_count",
            "minimize endpoint_peak_load",
            "minimize rank_reuse_pressure",
        ],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
