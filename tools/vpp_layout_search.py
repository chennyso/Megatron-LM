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
    schedule_makespan_ms: float
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
    microbatches: int,
    group_size: int,
    comm_ms: float,
    layer_costs: tuple[float, ...] | None = None,
) -> Candidate:
    stages = []
    layer_offset = 0
    for index, count in enumerate(counts):
        if layer_costs:
            cost = sum(layer_costs[layer_offset : layer_offset + count])
            layer_offset += count
        else:
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
    schedule_makespan = simulate_schedule(tuple(stages), pp, microbatches, group_size, comm_ms)
    objective = schedule_makespan + seam
    return Candidate(counts, tuple(sum(counts[r::pp]) for r in range(pp)), rank_load,
                     max(rank_load), seam, schedule_makespan, objective, layout_string(counts))


def simulate_schedule(stage_costs, pp: int, microbatches: int, group_size: int, comm_ms: float) -> float:
    """Greedy complete-period simulator matching Megatron's grouped VPP order."""
    vpp = len(stage_costs) // pp
    table = [(m, c) for start in range(0, microbatches, group_size)
             for c in range(vpp) for m in range(start, min(microbatches, start + group_size))]
    tasks = {}
    for m in range(microbatches):
        for s in range(pp * vpp):
            rank, chunk = s % pp, s // pp
            deps = []
            if rank > 0:
                deps.append((f"F:{s-1}:{m}", comm_ms))
            elif chunk > 0:
                deps.append((f"F:{s-pp}:{m}", 0.0))
            tasks[f"F:{s}:{m}"] = ("F", rank, stage_costs[s] / 2.0, deps)
        for s in reversed(range(pp * vpp)):
            rank, chunk = s % pp, s // pp
            deps = [(f"F:{s}:{m}", 0.0)]
            if rank < pp - 1:
                deps.append((f"B:{s+1}:{m}", comm_ms))
            elif chunk < vpp - 1:
                deps.append((f"B:{s+pp}:{m}", 0.0))
            tasks[f"B:{s}:{m}"] = ("B", rank, stage_costs[s] / 2.0, deps)
    local = {r: [] for r in range(pp)}
    for r in range(pp):
        warmup = min(len(table), (pp - r - 1) * 2 + (vpp - 1) * max(1, group_size))
        remaining = len(table) - warmup
        actions = [("F", table[k][0], table[k][1]) for k in range(warmup)]
        for k in range(remaining):
            actions.append(("F", table[k + warmup][0], table[k + warmup][1]))
            actions.append(("B", table[k][0], vpp - 1 - table[k][1]))
        actions.extend(("B", table[k][0], vpp - 1 - table[k][1]) for k in range(remaining, len(table)))
        for kind, m, c in actions:
            local[r].append(f"{kind}:{c * pp + r}:{m}")
    predecessor = {}
    for seq in local:
        for a, b in zip(local[seq], local[seq][1:]):
            predecessor[b] = a
    end, available = {}, [0.0] * pp
    pending = set(tasks)
    while pending:
        ready = []
        for task_id in pending:
            kind, rank, duration, deps = tasks[task_id]
            if not all(dep in end for dep, _ in deps):
                continue
            prev = predecessor.get(task_id)
            if prev is not None and prev not in end:
                continue
            start = max([available[rank]] + [end[d] + delay for d, delay in deps] + ([end[prev]] if prev else [0.0]))
            ready.append((start, 0 if kind == "B" else 1, task_id, rank, duration))
        if not ready:
            raise ValueError("cyclic VPP schedule")
        start, _, task_id, rank, duration = min(ready)
        end[task_id] = start + duration
        available[rank] = end[task_id]
        pending.remove(task_id)
    return max(end.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--vpp", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--decoder-ms", type=float, required=True)
    parser.add_argument("--embedding-ms", type=float, default=0.0)
    parser.add_argument("--loss-ms", type=float, default=0.0)
    parser.add_argument("--seam-weight-ms", type=float, default=0.0)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--comm-ms", type=float, default=4.84)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--profile-candidates",
        type=int,
        default=256,
        help="Run the complete grouped-1F1B simulator only for this many lower-bound candidates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layer-costs",
        type=str,
        default="",
        help="Optional comma-separated per-decoder costs; overrides --decoder-ms in stage scoring.",
    )
    args = parser.parse_args()
    stage_count = args.pp * args.vpp
    layer_costs = tuple(float(x) for x in args.layer_costs.split(",") if x.strip())
    if layer_costs and len(layer_costs) != args.layers:
        parser.error(f"--layer-costs must contain exactly {args.layers} values")
    raw = list(bounded_compositions(args.layers, stage_count, args.min_count, args.max_count))

    def cheap_ordering_key(counts: tuple[int, ...]) -> tuple[float, float, float, tuple[int, ...]]:
        """Cheap ordering key before the complete-period simulation.

        The first term is a rank-load lower bound.  The second term is the
        unavoidable boundary hotspot contribution, and the third is a small
        ordered-seam proxy.  This is used only to choose profiles; the final
        objective is always computed by ``score`` and no lower bound is
        presented as a performance result.
        """
        stages = []
        layer_offset = 0
        for index, count in enumerate(counts):
            if layer_costs:
                cost = sum(layer_costs[layer_offset : layer_offset + count])
                layer_offset += count
            else:
                cost = count * args.decoder_ms
            if index == 0:
                cost += args.embedding_ms
            if index == stage_count - 1:
                cost += args.loss_ms
            stages.append(cost)
        rank_load = [sum(stages[r::args.pp]) for r in range(args.pp)]
        seam_lb = args.seam_weight_ms * sum(
            abs(a - b) for a, b in zip(stages, stages[1:])
        )
        return (max(rank_load), max(stages), seam_lb, counts)

    raw.sort(key=cheap_ordering_key)
    profile_limit = max(args.top_k, args.profile_candidates)
    candidates = [
        score(
            c,
            args.pp,
            args.decoder_ms,
            args.embedding_ms,
            args.loss_ms,
            args.seam_weight_ms,
            args.microbatches,
            args.group_size,
            args.comm_ms,
            layer_costs or None,
        )
        for c in raw[:profile_limit]
    ]
    candidates.sort(key=lambda x: (x.objective_ms, x.max_rank_load_ms, x.decoder_counts))
    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "enumerated_candidates": len(raw),
        "profiled_candidates": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates[: args.top_k]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for candidate in candidates[: args.top_k]:
        print(f"{candidate.objective_ms:.3f} {candidate.layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
