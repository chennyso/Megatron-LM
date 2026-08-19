#!/usr/bin/env python3
"""Search globally ordered, non-uniform VPP layouts.

The search space is the decoder-count vector for PP*VPP virtual stages.  The
first and last stage carry the embedding/loss markers; their decoder counts
remain variables.  Unlike a per-rank load balancer, the score keeps the
ordered VPP seam positions, because Megatron executes virtual stages in that
order during warmup, steady state, and cooldown.

This is an offline candidate generator.  Every emitted layout still needs the
typed strategy verifier and a real training run before it is accepted.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    decoder_counts: tuple[int, ...]
    rank_counts: tuple[int, ...]
    rank_load_ms: tuple[float, ...]
    max_rank_load_ms: float
    seam_score_ms: float
    objective_ms: float
    layout: str


def bounded_compositions(total: int, parts: int, minimum: int, maximum: int):
    """Enumerate only bounded local layouts; unrestricted compositions explode."""
    def visit(prefix: tuple[int, ...], remaining: int):
        slots = parts - len(prefix)
        if slots == 0:
            if remaining == 0:
                yield prefix
            return
        low = max(minimum, remaining - maximum * (slots - 1))
        high = min(maximum, remaining - minimum * (slots - 1))
        for value in range(low, high + 1):
            yield from visit(prefix + (value,), remaining - value)

    yield from visit((), total)


def layout_string(counts: tuple[int, ...]) -> str:
    stages = []
    for index, count in enumerate(counts):
        marker = "E" if index == 0 else ""
        suffix = "L" if index == len(counts) - 1 else ""
        stages.append(marker + "t" * count + suffix)
    return "|".join(stages)


def score(
    counts: tuple[int, ...],
    pp: int,
    decoder_ms: float,
    embedding_ms: float,
    loss_ms: float,
    seam_weight_ms: float,
) -> Candidate:
    stages = []
    for index, count in enumerate(counts):
        cost = count * decoder_ms
        if index == 0:
            cost += embedding_ms
        if index == len(counts) - 1:
            cost += loss_ms
        stages.append(cost)
    rank_load = tuple(sum(stages[rank::pp]) for rank in range(pp))
    # The seam term preserves global VPP order.  A stage that is much heavier
    # than the preceding virtual stage creates a larger wavefront mismatch;
    # weighting the absolute adjacent jump is a conservative proxy for the
    # measured P2P wait exposed at VPP seams.
    seam = seam_weight_ms * sum(abs(a - b) for a, b in zip(stages, stages[1:]))
    objective = max(rank_load) + seam
    return Candidate(counts, tuple(sum(counts[r::pp]) for r in range(pp)), rank_load,
                     max(rank_load), seam, objective, layout_string(counts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--vpp", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--decoder-ms", type=float, required=True)
    parser.add_argument("--embedding-ms", type=float, default=0.0)
    parser.add_argument("--loss-ms", type=float, default=0.0)
    parser.add_argument("--seam-weight-ms", type=float, default=0.0)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_count = args.pp * args.vpp
    candidates = [
        score(c, args.pp, args.decoder_ms, args.embedding_ms, args.loss_ms, args.seam_weight_ms)
        for c in bounded_compositions(args.layers, stage_count, args.min_count, args.max_count)
    ]
    candidates.sort(key=lambda x: (x.objective_ms, x.max_rank_load_ms, x.decoder_counts))
    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "candidates": [asdict(candidate) for candidate in candidates[: args.top_k]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for candidate in candidates[: args.top_k]:
        print(f"{candidate.objective_ms:.3f} {candidate.layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
