#!/usr/bin/env python3
"""Audit real Megatron collective traces before proposing a strategy change.

This tool deliberately makes no performance recommendation.  It joins rank-local
collective issue order with rank-0 complete-step logs and reports only patterns
that repeat across measured iterations.  A pattern is evidence for a research
problem only after a controlled intervention reproduces its step-time effect.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


STEP_RE = re.compile(r"iteration\s+(\d+)/\s+\d+ .*?elapsed time per iteration \(ms\):\s*([0-9.]+)")
OP_CODE = {
    "all_gather_into_tensor": "AG",
    "reduce_scatter_tensor": "RS",
    "all_reduce": "AR",
}
P2P_OP_CODE = {
    "p2p_issue": "P2P_ISSUE",
    "p2p_wait": "P2P_WAIT",
}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))]


def load_step_times(run: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for log in run.glob("node.*.log"):
        for iteration, value in STEP_RE.findall(log.read_text(errors="ignore")):
            result[int(iteration)] = float(value)
    return result


def summarize_iteration(events: list[dict]) -> dict[str, object]:
    events.sort(key=lambda event: event["ts_ns"])
    sequence = [event["op"] for event in events]
    phases = Counter(event["phase"] for event in events)
    chunks = Counter(event["chunk"] for event in events)
    transitions = Counter(f"{a}>{b}" for a, b in zip(sequence, sequence[1:]))
    api_total = sum(event["api_ms"] for event in events)
    p2p_wait_by_edge: dict[str, float] = defaultdict(float)
    p2p_issue_count: Counter[str] = Counter()
    p2p_wait_count: Counter[str] = Counter()
    for event in events:
        edge = event.get("p2p_edge")
        if edge is None:
            continue
        if event["op"] == "P2P_ISSUE":
            p2p_issue_count[edge] += 1
        elif event["op"] == "P2P_WAIT":
            p2p_wait_count[edge] += 1
            p2p_wait_by_edge[edge] += event["wait_ms"]
    first = events[0]["ts_ns"]
    last = max(event["ts_ns"] + int(event["api_ms"] * 1e6) for event in events)
    return {
        "events": len(events),
        "api_total_ms": api_total,
        "issue_span_ms": (last - first) / 1e6,
        "first_ops": "-".join(sequence[:24]),
        "phases": dict(sorted(phases.items())),
        "chunks": {str(key): value for key, value in sorted(chunks.items(), key=lambda item: str(item[0]))},
        "transitions": dict(sorted(transitions.items())),
        "p2p_issue_count": int(sum(p2p_issue_count.values())),
        "p2p_wait_ms": sum(p2p_wait_by_edge.values()),
        "p2p_by_edge": {
            edge: {
                "issue_count": p2p_issue_count[edge],
                "wait_count": p2p_wait_count[edge],
                "wait_ms": p2p_wait_by_edge[edge],
            }
            for edge in sorted(set(p2p_issue_count) | set(p2p_wait_count))
        },
    }


def parse_rank_trace(path: Path) -> dict[int, dict[str, object]]:
    by_iteration: dict[int, list[dict]] = defaultdict(list)
    for raw in path.read_text(errors="ignore").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        context = item.get("context") or {}
        iteration = context.get("iteration")
        op = OP_CODE.get(item.get("op")) or P2P_OP_CODE.get(item.get("op"))
        if iteration is None or op is None:
            continue
        is_p2p = op.startswith("P2P_")
        if not is_p2p and item.get("api_ms") is None:
            continue
        direction = "".join(
            label
            for label, enabled in (
                ("sp", item.get("send_prev")),
                ("rp", item.get("recv_prev")),
                ("sn", item.get("send_next")),
                ("rn", item.get("recv_next")),
            )
            if enabled
        ) or "none"
        p2p_edge = None
        if is_p2p:
            p2p_edge = (
                f"{item.get('action_class', 'PP_UNKNOWN')}|"
                f"{context.get('phase', 'unknown')}|"
                f"chunk={context.get('vp_chunk', 'unknown')}|{direction}"
            )
        by_iteration[int(iteration)].append(
            {
                "ts_ns": int(item.get("issue_ts_ns", 0)),
                "op": op,
                "api_ms": float(item.get("api_ms", 0.0)),
                "wait_ms": float(item.get("wait_ms", 0.0)),
                "phase": str(context.get("phase", "unknown")),
                "chunk": context.get("vp_chunk"),
                "p2p_edge": p2p_edge,
            }
        )
    return {iteration: summarize_iteration(events) for iteration, events in by_iteration.items() if events}


def rank_from_trace_name(path: Path) -> int:
    match = re.search(r"rank(\d+)\.json\.phase\.jsonl$", path.name)
    if not match:
        raise ValueError(f"unexpected trace filename: {path}")
    return int(match.group(1))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def audit_run(run: Path, warmup: int, trace_iteration_offset: int) -> dict[str, object]:
    step_times = load_step_times(run)
    traces = sorted((run / "traces").glob("rank*.json.phase.jsonl"))
    ranks = {rank_from_trace_name(path): parse_rank_trace(path) for path in traces}
    measured = sorted(iteration for iteration in step_times if iteration >= warmup)
    step_values = [step_times[iteration] for iteration in measured]
    median = percentile(step_values, 0.5)
    slow_threshold = percentile(step_values, 0.8)
    rows = []
    for iteration in measured:
        trace_iteration = iteration - trace_iteration_offset
        rank_rows = {
            str(rank): trace[trace_iteration]
            for rank, trace in ranks.items()
            if trace_iteration in trace
        }
        rows.append(
            {
                "iteration": iteration,
                "trace_iteration": trace_iteration,
                "step_ms": step_times[iteration],
                "slow": step_times[iteration] >= slow_threshold,
                "rank_trace": rank_rows,
            }
        )

    # A pattern is keyed by its exact collective prefix and transition counts.
    # It must appear in both slow and non-slow iterations before it can be a
    # candidate explanation rather than a one-off warmup artifact.
    pattern_bins: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for row in rows:
        for rank, summary in row["rank_trace"].items():
            signature = json.dumps(
                {
                    "rank": rank,
                    "first_ops": summary["first_ops"],
                    "phases": summary["phases"],
                    "chunks": summary["chunks"],
                },
                sort_keys=True,
            )
            pattern_bins[signature].append((bool(row["slow"]), float(row["step_ms"])))

    associations = []
    for signature, values in pattern_bins.items():
        slow = [value for is_slow, value in values if is_slow]
        fast = [value for is_slow, value in values if not is_slow]
        if slow and fast:
            associations.append(
                {
                    "signature": json.loads(signature),
                    "n": len(values),
                    "slow_n": len(slow),
                    "fast_n": len(fast),
                    "slow_mean_ms": statistics.fmean(slow),
                    "fast_mean_ms": statistics.fmean(fast),
                    "delta_ms": statistics.fmean(slow) - statistics.fmean(fast),
                }
            )
    associations.sort(key=lambda item: (-item["delta_ms"], -item["n"]))

    # This is exploratory association, not a causal attribution.  A bucket
    # appears even when absent in an iteration (as a zero) so slow/fast means
    # and correlations use the same complete-step population.
    p2p_series: dict[str, list[tuple[bool, float, float]]] = defaultdict(list)
    p2p_keys = {
        (rank, edge)
        for row in rows
        for rank, summary in row["rank_trace"].items()
        for edge in summary.get("p2p_by_edge", {})
    }
    for rank, edge in p2p_keys:
        key = f"rank={rank}|{edge}"
        for row in rows:
            value = float(
                row["rank_trace"].get(rank, {}).get("p2p_by_edge", {}).get(edge, {}).get("wait_ms", 0.0)
            )
            p2p_series[key].append((bool(row["slow"]), float(row["step_ms"]), value))
    p2p_associations = []
    for key, values in p2p_series.items():
        slow = [value for is_slow, _, value in values if is_slow]
        fast = [value for is_slow, _, value in values if not is_slow]
        waits = [value for _, _, value in values]
        times = [value for _, value, _ in values]
        if not slow or not fast or not any(waits):
            continue
        p2p_associations.append(
            {
                "edge": key,
                "n": len(values),
                "slow_n": len(slow),
                "fast_n": len(fast),
                "slow_mean_wait_ms": statistics.fmean(slow),
                "fast_mean_wait_ms": statistics.fmean(fast),
                "delta_wait_ms": statistics.fmean(slow) - statistics.fmean(fast),
                "step_wait_pearson_r": pearson(times, waits),
            }
        )
    p2p_associations.sort(key=lambda item: (-item["delta_wait_ms"], -item["n"]))
    return {
        "run": run.name,
        "trace_ranks": sorted(ranks),
        "warmup_excluded_through": warmup - 1,
        "trace_iteration_offset": trace_iteration_offset,
        "step_times_ms": {str(key): value for key, value in step_times.items()},
        "step_summary": {
            "n": len(step_values),
            "mean_ms": statistics.fmean(step_values) if step_values else None,
            "median_ms": median,
            "p80_ms": slow_threshold,
            "p95_ms": percentile(step_values, 0.95),
        },
        "iteration_rows": rows,
        "repeating_slow_fast_associations": associations[:20],
        "p2p_slow_fast_associations": p2p_associations[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--trace-iteration-offset",
        type=int,
        default=1,
        help="log iteration minus trace context iteration (Megatron training defaults to one)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        run.name: audit_run(run, args.warmup, args.trace_iteration_offset)
        for run in args.runs
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in report.items():
        summary = result["step_summary"]
        print(f"{name}: n={summary['n']} mean={summary['mean_ms']} p95={summary['p95_ms']}")
        print(f"  repeated slow/fast signatures: {len(result['repeating_slow_fast_associations'])}")


if __name__ == "__main__":
    main()
