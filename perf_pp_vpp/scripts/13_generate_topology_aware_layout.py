#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LayoutSpec:
    pp: int
    vpp: int
    num_layers: int
    mode: str


def weighted_counts(total: int, weights: List[float]) -> List[int]:
    total_weight = sum(weights)
    raw = [total * w / total_weight for w in weights]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def topology_guarded_counts(spec: LayoutSpec) -> List[int]:
    slots = spec.pp * spec.vpp
    if spec.num_layers < slots:
        raise ValueError("topology_guarded requires num_layers >= pp * vpp")

    counts = [1] * slots
    extras = spec.num_layers - slots

    # Prefer adding work to the interior ranks on each node. Avoid the
    # embedding/output endpoints and the cross-node boundary ranks first.
    left_boundary = (spec.pp // 2) - 1
    right_boundary = spec.pp // 2
    preferred = [
        rank
        for rank in range(1, spec.pp - 1)
        if rank not in (left_boundary, right_boundary)
    ]
    boundary = [rank for rank in (left_boundary, right_boundary) if 0 <= rank < spec.pp]
    rank_priority = preferred + boundary + [0, spec.pp - 1]

    rank_cursor = 0
    rank_visits = [0] * spec.pp
    while extras > 0:
        pp_rank = rank_priority[rank_cursor % len(rank_priority)]
        rank_cursor += 1
        vp = rank_visits[pp_rank] % spec.vpp
        rank_visits[pp_rank] += 1
        slot = vp * spec.pp + pp_rank
        counts[slot] += 1
        extras -= 1
    return counts


def stage_weights(spec: LayoutSpec) -> List[float]:
    slots = spec.pp * spec.vpp
    weights = [1.0] * slots
    left_boundary = (spec.pp // 2) - 1
    right_boundary = spec.pp // 2
    for vp in range(spec.vpp):
        offset = vp * spec.pp
        if spec.mode == "uniform":
            continue
        if spec.mode == "boundary_light":
            weights[offset + left_boundary] = 0.65
            weights[offset + right_boundary] = 0.65
            for rank in (left_boundary - 1, left_boundary - 2, right_boundary + 1, right_boundary + 2):
                if 0 <= rank < spec.pp:
                    weights[offset + rank] = max(weights[offset + rank], 1.15)
            continue
        if spec.mode == "boundary_dilated":
            weights[offset + left_boundary] = 0.45
            weights[offset + right_boundary] = 0.45
            for rank in (left_boundary - 1, right_boundary + 1):
                if 0 <= rank < spec.pp:
                    weights[offset + rank] = max(weights[offset + rank], 0.80)
            for rank in (left_boundary - 2, left_boundary - 3, right_boundary + 2, right_boundary + 3):
                if 0 <= rank < spec.pp:
                    weights[offset + rank] = max(weights[offset + rank], 1.20)
            continue
        raise ValueError(f"unsupported mode: {spec.mode}")
    return weights


def counts_to_layout(counts: List[int]) -> str:
    stages = []
    for idx, count in enumerate(counts):
        stage = ""
        if idx == 0:
            stage += "E"
        stage += "t" * count
        if idx == len(counts) - 1:
            stage += "L"
        stages.append(stage)
    return "|".join(stages)


def per_rank_totals(counts: List[int], pp: int, vpp: int) -> List[int]:
    return [sum(counts[vp * pp + pp_rank] for vp in range(vpp)) for pp_rank in range(pp)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a topology-aware Megatron pipeline layout.")
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--vpp", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("uniform", "boundary_light", "boundary_dilated", "topology_guarded"),
        required=True,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = LayoutSpec(pp=args.pp, vpp=args.vpp, num_layers=args.num_layers, mode=args.mode)
    if spec.mode == "topology_guarded":
        counts = topology_guarded_counts(spec)
    else:
        counts = weighted_counts(spec.num_layers, stage_weights(spec))
    layout = counts_to_layout(counts)

    if args.json:
        payload = {
            "layout": layout,
            "counts_per_stage": counts,
            "counts_per_pp_rank": per_rank_totals(counts, spec.pp, spec.vpp),
            "pp": spec.pp,
            "vpp": spec.vpp,
            "num_layers": spec.num_layers,
            "mode": spec.mode,
        }
        print(json.dumps(payload, indent=2))
        return

    print(layout)


if __name__ == "__main__":
    main()
