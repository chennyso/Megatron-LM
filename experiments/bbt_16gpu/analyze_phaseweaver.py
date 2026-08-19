#!/usr/bin/env python3
"""Summarize complete-step phase traces for the PhaseWeaver study.

This deliberately reports measured API timing and issue-phase structure as
separate quantities.  API timing is not claimed to be device makespan; the
distinction is important when constructing the interaction-surplus argument.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "cvar95": statistics.fmean(sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1) :])
        if values
        else None,
    }


def parse_step_times(run: Path) -> dict[str, object]:
    # Only one worker log normally emits the rank-0 timing line.  Deduplicate
    # by iteration in case both node logs contain the same line.
    pattern = re.compile(
        r"iteration\s+(\d+)/\s+\d+ .*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
    )
    by_iter: dict[int, float] = {}
    for log in sorted(run.glob("node.*.log")):
        for match in pattern.finditer(log.read_text(errors="ignore")):
            by_iter[int(match.group(1))] = float(match.group(2))
    values = [by_iter[key] for key in sorted(by_iter)]
    return {"iterations": sorted(by_iter), "times_ms": values, **stats(values)}


def classify(op: str, group_size: int | None) -> str:
    if op == "all_gather_into_tensor":
        return "tp_sp_all_gather" if group_size == 2 else "all_gather"
    if op == "reduce_scatter_tensor":
        return "dp_zero_reduce_scatter" if group_size == 2 else "reduce_scatter"
    if op == "all_reduce":
        return "all_reduce"
    return op


def parse_phase(run: Path) -> dict[str, object]:
    by_class: dict[str, list[float]] = defaultdict(list)
    by_iteration: dict[int, list[dict[str, object]]] = defaultdict(list)
    pair_counts: Counter[str] = Counter()
    files = sorted((run / "traces").glob("*.phase.jsonl"))
    for path in files:
        events = []
        for line in path.open(encoding="utf-8"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            context = event.get("context") or {}
            if context.get("iteration") is None:
                continue
            iteration = int(context["iteration"])
            op = str(event.get("op"))
            group_size = event.get("group_size")
            cls = classify(op, int(group_size) if group_size is not None else None)
            api_ms = event.get("api_ms")
            if api_ms is None:
                continue
            api_ms = float(api_ms)
            by_class[cls].append(api_ms)
            item = {
                "ts_ns": int(event.get("issue_ts_ns", 0)),
                "api_ms": api_ms,
                "class": cls,
                "op": op,
                "group_size": group_size,
                "vp_chunk": context.get("vp_chunk"),
                "phase": context.get("phase"),
            }
            by_iteration[iteration].append(item)
            events.append((iteration, item))
        # Near-issue pairs are a structural phase-aliasing proxy.  They do not
        # assert device overlap; they identify where a global scheduler must
        # reason about multiple action types in one short issue window.
        events.sort(key=lambda pair: pair[1]["ts_ns"])
        for (it_a, a), (it_b, b) in zip(events, events[1:]):
            if it_a != it_b:
                continue
            gap_ms = (b["ts_ns"] - a["ts_ns"]) / 1e6
            if gap_ms <= 0.25 and a["class"] != b["class"]:
                pair_counts[f"{a['class']}->{b['class']}"] += 1

    iteration_summary = {}
    for iteration, items in sorted(by_iteration.items()):
        items.sort(key=lambda item: item["ts_ns"])
        first = items[0]["ts_ns"]
        last = max(item["ts_ns"] + int(item["api_ms"] * 1e6) for item in items)
        iteration_summary[str(iteration)] = {
            "events": len(items),
            "classes": dict(Counter(item["class"] for item in items)),
            "issue_span_ms": (last - first) / 1e6,
            "api_sum_ms": sum(float(item["api_ms"]) for item in items),
            "near_issue_pairs": sum(
                1
                for a, b in zip(items, items[1:])
                if (b["ts_ns"] - a["ts_ns"]) / 1e6 <= 0.25
                and a["class"] != b["class"]
            ),
        }
    return {
        "files": len(files),
        "events_by_class": {key: stats(value) for key, value in sorted(by_class.items())},
        "iterations": iteration_summary,
        "near_issue_pairs_0p25ms": dict(pair_counts),
    }


def parse_schedule(run: Path) -> dict[str, object]:
    p2p_wait: list[float] = []
    p2p_issue: list[float] = []
    p2p_api: list[float] = []
    p2p_by_name: dict[str, list[float]] = defaultdict(list)
    memory: list[float] = []
    for path in sorted((run / "traces").glob("rank*.json")):
        try:
            events = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(events, list):
            continue
        for event in events:
            name = event.get("name", "")
            elapsed = event.get("elapsed_ms")
            if str(name).startswith("p2p_") and elapsed is not None:
                value = float(elapsed)
                p2p_by_name[str(name)].append(value)
                # The high-level schedule trace and the low-level communicator
                # trace are complementary: recv/send entries measure the
                # synchronous schedule API path, while comm_issue/wait expose
                # the nonblocking request and its exposed completion wait.
                p2p_api.append(value)
                if name == "p2p_comm_wait":
                    p2p_wait.append(value)
                elif name == "p2p_comm_issue":
                    p2p_issue.append(value)
            if event.get("memory_mb") is not None:
                memory.append(float(event["memory_mb"]))
    return {
        "p2p_wait_ms": stats(p2p_wait),
        "p2p_issue_ms": stats(p2p_issue),
        "p2p_api_ms": stats(p2p_api),
        "p2p_by_name": {name: stats(values) for name, values in sorted(p2p_by_name.items())},
        "max_trace_memory_mb": max(memory) if memory else None,
    }


def summarize(run: Path) -> dict[str, object]:
    return {
        "run": run.name,
        "step": parse_step_times(run),
        "phase": parse_phase(run),
        "schedule": parse_schedule(run),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {run.name: summarize(run) for run in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
