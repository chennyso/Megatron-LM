#!/usr/bin/env python3
"""Audit PipePerf-style semantic invariants in Megatron phase JSONL traces.

This is an evidence tool, not a performance optimizer.  It checks message
pairing, per-direction payloads, collective ownership, and replicated DP
bucket shape.  A failed check is a candidate execution-contract violation;
large durations or unequal stage payloads are reported separately because
they are performance symptoms, not atomicity errors.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Iterable


def _context(event: dict) -> tuple:
    context = event.get("context") or {}
    return (
        context.get("iteration"),
        context.get("microbatch_id"),
        context.get("virtual_microbatch_id"),
        context.get("vp_chunk"),
    )


def _read(paths: Iterable[str]) -> list[dict]:
    events = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event["_path"] = str(path)
            events.append(event)
    return events


def _p2p_check(events: list[dict]) -> dict:
    checks = {}
    for direction, send_field, recv_field, recv_peer_field, send_peer_field in (
        ("forward", "send_next_bytes", "recv_prev_bytes", "prev_pipeline_rank", "next_pipeline_rank"),
        ("backward", "send_prev_bytes", "recv_next_bytes", "next_pipeline_rank", "prev_pipeline_rank"),
    ):
        sends = collections.Counter()
        receives = collections.Counter()
        for event in events:
            if event.get("op") != "p2p_issue":
                continue
            if event.get(send_field, 0):
                sends[
                    (
                        event.get("rank"),
                        event.get(send_peer_field),
                        event.get(send_field),
                        event.get("action_class"),
                    )
                ] += 1
            if event.get(recv_field, 0):
                receives[
                    (
                        event.get(recv_peer_field),
                        event.get("rank"),
                        event.get(recv_field),
                        event.get("action_class"),
                    )
                ] += 1
        checks[direction] = {
            "send_messages": sum(sends.values()),
            "recv_messages": sum(receives.values()),
            "edge_payload_mismatch": [
                {"key": list(key), "send": sends[key], "recv": receives[key]}
                for key in sorted(set(sends) | set(receives), key=str)
                if sends[key] != receives[key]
            ],
        }
    issue_by_id = {
        event.get("event_id"): event
        for event in events
        if event.get("op") == "p2p_issue" and event.get("event_kind") == "issue"
    }
    waits_by_parent = collections.Counter()
    unknown_requests = 0
    for event in events:
        if event.get("op") != "p2p_wait":
            continue
        unknown_requests += int(event.get("unknown_request_count", 0) or 0)
        for parent_id in event.get("request_parent_event_ids", []) or []:
            waits_by_parent[(event.get("rank"), parent_id)] += 1
    request_lifecycle_errors = []
    for event_id, event in issue_by_id.items():
        request_count = int(event.get("request_count", 0) or 0)
        if request_count <= 0:
            continue
        observed = waits_by_parent[(event.get("rank"), event_id)]
        waited = event.get("p2p_waited_count")
        double_waits = int(event.get("p2p_double_wait_count", 0) or 0)
        if observed != request_count or (waited is not None and int(waited) != request_count) or double_waits:
            request_lifecycle_errors.append(
                {
                    "rank": event.get("rank"),
                    "event_id": event_id,
                    "request_count": request_count,
                    "wait_parent_count": observed,
                    "p2p_waited_count": waited,
                    "p2p_double_wait_count": double_waits,
                    "wait_on_reqs": event.get("wait_on_reqs"),
                    "context": event.get("context"),
                }
            )
    checks["request_lifecycle"] = {
        "issue_events_with_requests": sum(
            int(event.get("request_count", 0) or 0) > 0 for event in issue_by_id.values()
        ),
        "unknown_request_count": unknown_requests,
        "errors": request_lifecycle_errors,
        "pass": not request_lifecycle_errors and unknown_requests == 0,
    }
    return checks


def _ticket_check(events: list[dict]) -> dict:
    """Check local collective ticket monotonicity and DP replica signatures."""
    violations = []
    by_stream = collections.defaultdict(list)
    for event in events:
        if event.get("name") != "collective_issue":
            continue
        if event.get("group_key") is None or event.get("group_ticket") is None:
            continue
        key = (event.get("rank"), event.get("group_key"))
        by_stream[key].append((event.get("group_ticket"), event.get("event_id")))
    for key, tickets in by_stream.items():
        values = [ticket for ticket, _ in tickets]
        if values != sorted(values) or len(values) != len(set(values)):
            violations.append({"key": list(key), "tickets": tickets})

    return {
        "groups": len(by_stream),
        "monotonicity_violations": violations,
        "pass": not violations,
    }


def _dp_shape_check(events: list[dict]) -> dict:
    # Compare the ordered bucket payload sequence across the two DP replicas
    # that share a PP/TP coordinate.  Different PP coordinates are expected
    # to have different amounts of state (e.g. embeddings or LM head).
    by_coordinate = collections.defaultdict(lambda: collections.defaultdict(list))
    for event in events:
        if event.get("action_class") != "DP_RS":
            continue
        context = event.get("context") or {}
        coordinate = (context.get("pp_rank"), context.get("tp_rank"), context.get("iteration"))
        by_coordinate[coordinate][event.get("rank")].append(event.get("payload_bytes"))
    mismatches = []
    for coordinate, ranks in by_coordinate.items():
        sequences = {tuple(values) for values in ranks.values()}
        if len(sequences) > 1:
            mismatches.append({"coordinate": list(coordinate), "ranks": ranks})
    return {
        "coordinates": len(by_coordinate),
        "replica_shape_mismatches": mismatches,
        "pass": not mismatches,
    }


def audit(paths: Iterable[str]) -> dict:
    events = _read(paths)
    classes = collections.Counter(event.get("action_class", "UNKNOWN") for event in events)
    bad_completion = [
        event
        for event in events
        if isinstance(event.get("issue_ts_ns"), int)
        and isinstance(event.get("complete_ts_ns"), int)
        and event["complete_ts_ns"] < event["issue_ts_ns"]
    ]
    return {
        "trace_files": len(list(paths)) if not isinstance(paths, list) else len(paths),
        "events": len(events),
        "action_classes": dict(classes),
        "unknown_collective_events": sum(k.startswith("UNKNOWN_") for k in classes for _ in range(classes[k])),
        "negative_duration_events": len(bad_completion),
        "p2p": _p2p_check(events),
        "collective_tickets": _ticket_check(events),
        "dp_bucket_replica_shape": _dp_shape_check(events),
        "claim_boundary": (
            "No novel atomicity finding. Stage payload asymmetry is a known performance symptom; "
            "async stream/order semantics require additional instrumentation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(path for pattern in args.trace_glob for path in glob.glob(pattern, recursive=True))
    result = audit(paths)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
