#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ITER_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\s*\|"
    r".*?elapsed time per iteration \(ms\):\s*(?P<iter_ms>[0-9.]+)\s*\|"
    r".*?global batch size:\s*(?P<global_batch_size>\d+)\s*\|"
    r".*?lm loss:\s*(?P<loss>[0-9.Ee+-]+)\s*\|"
    r".*?grad norm:\s*(?P<grad_norm>[0-9.Ee+-]+)\s*\|",
    re.IGNORECASE,
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "cv_percent": None, "p95": None}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "cv_percent": stdev / mean * 100.0 if mean else None,
        "p95": percentile(values, 0.95),
    }


def parse_steps(log_path: Path, warmup_steps: int, seq_length: int) -> tuple[list[dict], dict]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict] = []
    for match in ITER_RE.finditer(text):
        iteration = int(match.group("iteration"))
        iter_ms = float(match.group("iter_ms"))
        global_batch_size = int(match.group("global_batch_size"))
        rows.append(
            {
                "iteration": iteration,
                "iter_time_ms": iter_ms,
                "global_batch_size": global_batch_size,
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "tokens_per_second": global_batch_size * seq_length * 1000.0 / iter_ms,
                "steady": iteration > warmup_steps,
            }
        )
    steady = [row for row in rows if row["steady"]]
    iter_summary = summarize([row["iter_time_ms"] for row in steady])
    token_summary = summarize([row["tokens_per_second"] for row in steady])
    result = {
        "completed_steps": len(rows),
        "steady_steps": len(steady),
        "iteration_time_ms": iter_summary,
        "tokens_per_second": token_summary,
        "first_loss": rows[0]["loss"] if rows else None,
        "last_loss": rows[-1]["loss"] if rows else None,
        "last_grad_norm": rows[-1]["grad_norm"] if rows else None,
    }
    return rows, result


def interval_union_ms(events: list[dict]) -> float:
    intervals = sorted(
        (float(event["start_ts"]), float(event["end_ts"]))
        for event in events
        if event.get("start_ts") is not None and event.get("end_ts") is not None
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total * 1000.0


def analyze_traces(trace_dir: Path) -> dict:
    events: list[dict] = []
    for path in sorted(trace_dir.glob("rank*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            events.extend(event for event in payload if isinstance(event, dict))
    by_rank: dict[int, list[dict]] = defaultdict(list)
    by_rank_chunk: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in events:
        rank = int(event.get("pp_rank", -1))
        by_rank[rank].append(event)
        chunk = event.get("model_chunk_id")
        if chunk is not None and event.get("name") in {"forward_step", "backward_step"}:
            by_rank_chunk[(rank, int(chunk))].append(event)

    rank_rows = []
    compute_totals = []
    wait_totals = []
    for rank, rank_events in sorted(by_rank.items()):
        compute = [event for event in rank_events if event.get("name") in {"forward_step", "backward_step"}]
        waits = [event for event in rank_events if "wait" in str(event.get("name", ""))]
        compute_ms = sum(float(event.get("elapsed_ms", 0.0)) for event in compute)
        exposed_wait_ms = sum(float(event.get("wait_ms", event.get("elapsed_ms", 0.0))) for event in waits)
        compute_totals.append(compute_ms)
        wait_totals.append(exposed_wait_ms)
        rank_rows.append(
            {
                "pp_rank": rank,
                "compute_ms": compute_ms,
                "compute_union_ms": interval_union_ms(compute),
                "exposed_wait_ms": exposed_wait_ms,
                "p2p_issue_count": sum(event.get("name") == "p2p_comm_issue" for event in rank_events),
                "p2p_wait_count": sum(event.get("name") == "p2p_comm_wait" for event in rank_events),
                "peak_memory_mb": max((float(event.get("memory_mb") or 0.0) for event in rank_events), default=0.0),
            }
        )

    chunk_rows = []
    for (rank, chunk), chunk_events in sorted(by_rank_chunk.items()):
        chunk_rows.append(
            {
                "pp_rank": rank,
                "model_chunk_id": chunk,
                "forward_ms": sum(float(event.get("elapsed_ms", 0.0)) for event in chunk_events if event.get("name") == "forward_step"),
                "backward_ms": sum(float(event.get("elapsed_ms", 0.0)) for event in chunk_events if event.get("name") == "backward_step"),
                "event_count": len(chunk_events),
            }
        )

    boundary_events = [
        event
        for event in events
        if int(event.get("pp_rank", -1)) in {3, 4} and event.get("name") in {"p2p_comm_issue", "p2p_comm_wait"}
    ]
    return {
        "event_count": len(events),
        "rank_rows": rank_rows,
        "chunk_rows": chunk_rows,
        "stage_compute_imbalance_percent": (
            (max(compute_totals) - min(compute_totals)) / statistics.fmean(compute_totals) * 100.0
            if compute_totals and statistics.fmean(compute_totals) > 0
            else None
        ),
        "rank_exposed_wait_ms": summarize(wait_totals),
        "boundary_p2p_event_count": len(boundary_events),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repeat_rows: list[dict] = []
    case_values: dict[str, list[float]] = defaultdict(list)
    for repeat_dir in sorted(args.run_root.glob("*/repeat_*")):
        case_id = repeat_dir.parent.name
        log_path = repeat_dir / "node_0" / "training.log"
        if not log_path.exists():
            continue
        steps, summary = parse_steps(log_path, args.warmup_steps, args.seq_length)
        trace_summary = analyze_traces(repeat_dir / "strategy_traces")
        repeat_id = int(repeat_dir.name.split("_")[-1])
        median_ms = summary["iteration_time_ms"]["median"]
        if median_ms is not None:
            case_values[case_id].append(float(median_ms))
        repeat_rows.append(
            {
                "case_id": case_id,
                "repeat_id": repeat_id,
                "completed_steps": summary["completed_steps"],
                "steady_steps": summary["steady_steps"],
                "median_iter_ms": median_ms,
                "mean_iter_ms": summary["iteration_time_ms"]["mean"],
                "within_run_cv_percent": summary["iteration_time_ms"]["cv_percent"],
                "median_tokens_per_second": summary["tokens_per_second"]["median"],
                "first_loss": summary["first_loss"],
                "last_loss": summary["last_loss"],
                "last_grad_norm": summary["last_grad_norm"],
                "trace_event_count": trace_summary["event_count"],
                "stage_compute_imbalance_percent": trace_summary["stage_compute_imbalance_percent"],
                "boundary_p2p_event_count": trace_summary["boundary_p2p_event_count"],
            }
        )
        write_csv(args.output_dir / "per_step" / f"{case_id}_r{repeat_id:02d}.csv", steps)
        write_csv(args.output_dir / "trace_rank" / f"{case_id}_r{repeat_id:02d}.csv", trace_summary["rank_rows"])
        write_csv(args.output_dir / "trace_chunk" / f"{case_id}_r{repeat_id:02d}.csv", trace_summary["chunk_rows"])

    case_rows = []
    for case_id, values in sorted(case_values.items()):
        aggregate = summarize(values)
        case_rows.append(
            {
                "case_id": case_id,
                "repeat_count": aggregate["count"],
                "median_of_run_medians_ms": aggregate["median"],
                "mean_of_run_medians_ms": aggregate["mean"],
                "between_run_cv_percent": aggregate["cv_percent"],
                "p95_run_median_ms": aggregate["p95"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "repeat_summary.csv", repeat_rows)
    write_csv(args.output_dir / "case_summary.csv", case_rows)
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "sequence_length": args.seq_length,
                "repeat_rows": len(repeat_rows),
                "case_rows": len(case_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
