#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Classify effective PP/VPP communication overlap from traces.

The goal is not to maximize visual overlap on a timeline. The useful signal for
PP/VPP search is whether communication is hidden by compute without slowing the
critical compute kernels. This tool consumes either Megatron strategy traces or
SQLite files exported from Nsight Systems and emits a stable JSON report that
can be joined with BCP-VPP candidate reports.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, List


COMPUTE_HINTS = (
    "forward",
    "backward",
    "dgrad",
    "wgrad",
    "gemm",
    "matmul",
    "attention",
    "layernorm",
    "softmax",
    "kernel",
)
COMM_HINTS = (
    "p2p",
    "send",
    "recv",
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "broadcast",
    "memcpy",
)


@dataclass(frozen=True)
class TimelineEvent:
    name: str
    kind: str
    start_ms: float
    end_ms: float
    rank: int | None = None
    wait_ms: float = 0.0
    source: str = ""

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class OverlapPair:
    comm_name: str
    compute_name: str
    overlap_ms: float
    comm_duration_ms: float
    compute_duration_ms: float
    compute_slowdown: float
    category: str
    rank: int | None = None


@dataclass(frozen=True)
class EffectiveOverlapReport:
    span_ms: float
    compute_ms: float
    comm_ms: float
    useful_overlap_ms: float
    harmful_overlap_ms: float
    fake_overlap_ms: float
    exposed_wait_ms: float
    hidden_comm_ratio: float
    harmful_overlap_ratio: float
    top_pairs: List[OverlapPair]


def _read_json_payloads(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    payloads: List[Any] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            payload, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            if payloads:
                break
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        payloads.append(payload)
        idx = end
    return payloads


def _flatten_json_events(payloads: Iterable[Any]) -> List[dict]:
    events: List[dict] = []
    for payload in payloads:
        if isinstance(payload, list):
            events.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
            events.extend(item for item in payload["events"] if isinstance(item, dict))
        elif isinstance(payload, dict):
            events.append(payload)
    return events


def _event_kind(name: str) -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in COMM_HINTS):
        return "comm"
    if any(hint in lowered for hint in COMPUTE_HINTS):
        return "compute"
    return "other"


def _event_rank(event: dict) -> int | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    raw = event.get("rank", event.get("pp_rank", event.get("pp_group_rank", metadata.get("rank"))))
    return None if raw is None else int(raw)


def load_megatron_trace(paths: Iterable[str]) -> List[TimelineEvent]:
    timeline: List[TimelineEvent] = []
    for raw_path in paths:
        path = Path(raw_path)
        for event in _flatten_json_events(_read_json_payloads(path)):
            if event.get("start_ts") is None or event.get("end_ts") is None:
                continue
            start = float(event["start_ts"])
            end = float(event["end_ts"])
            if end < start:
                continue
            name = str(event.get("name", "unknown"))
            timeline.append(
                TimelineEvent(
                    name=name,
                    kind=_event_kind(name),
                    start_ms=start * 1000.0,
                    end_ms=end * 1000.0,
                    rank=_event_rank(event),
                    wait_ms=float(event.get("wait_ms", 0.0) or 0.0),
                    source=str(path),
                )
            )
    return timeline


def _sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "select name from sqlite_master where type='table' order by name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()}


def _pick_column(columns: set[str], candidates: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _nsys_string_map(conn: sqlite3.Connection) -> dict[int, str]:
    values: dict[int, str] = {}
    for table in _sqlite_tables(conn):
        if table.upper() not in {"STRINGIDS", "NVTX_STRINGS"}:
            continue
        columns = _sqlite_columns(conn, table)
        id_col = _pick_column(columns, ("id", "key"))
        value_col = _pick_column(columns, ("value", "string", "text"))
        if not id_col or not value_col:
            continue
        for raw_id, raw_value in conn.execute(f"select {id_col}, {value_col} from {table}"):
            try:
                values[int(raw_id)] = str(raw_value)
            except (TypeError, ValueError):
                continue
    return values


def _resolve_name(raw: Any, string_map: dict[int, str]) -> str:
    if raw is None:
        return "unknown"
    try:
        raw_int = int(raw)
    except (TypeError, ValueError):
        return str(raw)
    return string_map.get(raw_int, str(raw))


def load_nsys_sqlite(path: str) -> List[TimelineEvent]:
    db_path = Path(path)
    conn = sqlite3.connect(str(db_path))
    try:
        string_map = _nsys_string_map(conn)
        timeline: List[TimelineEvent] = []
        for table in _sqlite_tables(conn):
            columns = _sqlite_columns(conn, table)
            start_col = _pick_column(columns, ("start", "startNs", "start_ns", "startTime"))
            end_col = _pick_column(columns, ("end", "endNs", "end_ns", "endTime"))
            if not start_col or not end_col:
                continue
            name_col = _pick_column(
                columns,
                ("name", "shortName", "demangledName", "text", "message", "globalTid"),
            )
            rank_col = _pick_column(columns, ("rank", "deviceId", "gpuId", "globalTid"))
            kind_hint = table.lower()
            if "kernel" in kind_hint:
                default_kind = "compute"
            elif "memcpy" in kind_hint or "nccl" in kind_hint or "runtime" in kind_hint:
                default_kind = "comm"
            else:
                default_kind = "other"
            if default_kind == "other" and not any(hint in kind_hint for hint in COMM_HINTS):
                continue

            query_cols = [start_col, end_col]
            if name_col:
                query_cols.append(name_col)
            if rank_col and rank_col != name_col:
                query_cols.append(rank_col)
            query = f"select {', '.join(query_cols)} from {table}"
            for row in conn.execute(query):
                start_ns = float(row[0])
                end_ns = float(row[1])
                if end_ns < start_ns:
                    continue
                raw_name = row[2] if name_col else table
                name = _resolve_name(raw_name, string_map)
                kind = _event_kind(name)
                if kind == "other":
                    kind = default_kind
                rank = None
                if rank_col and rank_col != name_col and len(row) > 3:
                    try:
                        rank = int(row[3])
                    except (TypeError, ValueError):
                        rank = None
                timeline.append(
                    TimelineEvent(
                        name=name,
                        kind=kind,
                        start_ms=start_ns / 1_000_000.0,
                        end_ms=end_ns / 1_000_000.0,
                        rank=rank,
                        source=str(db_path),
                    )
                )
        return timeline
    finally:
        conn.close()


def _overlap_ms(left: TimelineEvent, right: TimelineEvent) -> float:
    return max(0.0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def _median_solo_compute(events: List[TimelineEvent]) -> dict[str, float]:
    compute = [event for event in events if event.kind == "compute"]
    comm = [event for event in events if event.kind == "comm"]
    solo_by_name: dict[str, List[float]] = {}
    for compute_event in compute:
        if any(_overlap_ms(compute_event, comm_event) > 0 for comm_event in comm):
            continue
        solo_by_name.setdefault(compute_event.name, []).append(compute_event.duration_ms)
    return {name: median(values) for name, values in solo_by_name.items() if values}


def classify_effective_overlap(
    events: List[TimelineEvent],
    *,
    harmful_slowdown_threshold: float = 0.15,
    max_pairs: int = 25,
) -> EffectiveOverlapReport:
    timed = [event for event in events if event.duration_ms > 0]
    compute = [event for event in timed if event.kind == "compute"]
    comm = [event for event in timed if event.kind == "comm"]
    if timed:
        span_ms = max(event.end_ms for event in timed) - min(event.start_ms for event in timed)
    else:
        span_ms = 0.0
    solo_medians = _median_solo_compute(timed)

    useful = 0.0
    harmful = 0.0
    fake = 0.0
    pairs: List[OverlapPair] = []
    for comm_event in comm:
        comm_overlap = 0.0
        comm_harmful = 0.0
        for compute_event in compute:
            overlap = _overlap_ms(comm_event, compute_event)
            if overlap <= 0:
                continue
            baseline = solo_medians.get(compute_event.name)
            if baseline and baseline > 0:
                slowdown = compute_event.duration_ms / baseline
            else:
                slowdown = 1.0
            category = "useful"
            if slowdown > 1.0 + harmful_slowdown_threshold:
                category = "harmful"
                comm_harmful += overlap
            comm_overlap += overlap
            pairs.append(
                OverlapPair(
                    comm_name=comm_event.name,
                    compute_name=compute_event.name,
                    overlap_ms=overlap,
                    comm_duration_ms=comm_event.duration_ms,
                    compute_duration_ms=compute_event.duration_ms,
                    compute_slowdown=slowdown,
                    category=category,
                    rank=comm_event.rank if comm_event.rank is not None else compute_event.rank,
                )
            )
        covered = min(comm_event.duration_ms, comm_overlap)
        harmful_part = min(covered, comm_harmful)
        useful += max(0.0, covered - harmful_part)
        harmful += harmful_part
        fake += max(0.0, comm_event.duration_ms - covered)

    exposed_wait = sum(max(0.0, event.wait_ms) for event in comm)
    comm_ms = sum(event.duration_ms for event in comm)
    pairs = sorted(pairs, key=lambda item: item.overlap_ms, reverse=True)[:max_pairs]
    return EffectiveOverlapReport(
        span_ms=span_ms,
        compute_ms=sum(event.duration_ms for event in compute),
        comm_ms=comm_ms,
        useful_overlap_ms=useful,
        harmful_overlap_ms=harmful,
        fake_overlap_ms=fake,
        exposed_wait_ms=exposed_wait,
        hidden_comm_ratio=(useful + harmful) / max(comm_ms, 1e-6),
        harmful_overlap_ratio=harmful / max(useful + harmful, 1e-6),
        top_pairs=pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", default=[], help="Megatron strategy trace JSON files")
    parser.add_argument("--nsys-sqlite", nargs="+", default=[], help="Nsight Systems SQLite exports")
    parser.add_argument("--output", required=True, help="Path to write overlap report JSON")
    parser.add_argument("--harmful-slowdown-threshold", type=float, default=0.15)
    parser.add_argument("--max-pairs", type=int, default=25)
    args = parser.parse_args()

    events: List[TimelineEvent] = []
    if args.trace:
        events.extend(load_megatron_trace(args.trace))
    for db_path in args.nsys_sqlite:
        events.extend(load_nsys_sqlite(db_path))
    if not events:
        raise RuntimeError("no timeline events loaded from --trace or --nsys-sqlite")

    report = classify_effective_overlap(
        events,
        harmful_slowdown_threshold=args.harmful_slowdown_threshold,
        max_pairs=args.max_pairs,
    )
    payload = {
        "summary": asdict(report),
        "num_events": len(events),
        "num_compute_events": sum(1 for event in events if event.kind == "compute"),
        "num_comm_events": sum(1 for event in events if event.kind == "comm"),
        "sources": sorted({event.source for event in events if event.source}),
        "method": {
            "harmful_slowdown_threshold": args.harmful_slowdown_threshold,
            "classification": (
                "useful=comm overlapped by compute without observed compute slowdown; "
                "harmful=overlap with slower-than-solo compute; fake=comm time not covered by compute"
            ),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
