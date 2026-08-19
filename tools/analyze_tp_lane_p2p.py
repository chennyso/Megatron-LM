#!/usr/bin/env python3
"""Pair traced PP P2P messages and report lane-specific exposed waits.

This tool intentionally reports host-observed issue and wait timing, not a
claim about wire latency.  Pairing is based on trace-only semantic identities
and is therefore safe to use while the runtime retains Megatron's FIFO P2P
semantics.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _read(paths: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("op") == "p2p_issue":
                event["_path"] = path
                events.append(event)
    return events


def _request_wait(event: dict[str, Any], label: str, tag: str) -> float | None:
    waits = [
        float(item["wait_ms"])
        for item in event.get("p2p_request_waits", [])
        if item.get("request_label") in {label, tag} and not item.get("double_wait")
    ]
    return max(waits) if waits else None


def analyze(events: list[dict[str, Any]]) -> dict[str, Any]:
    sends: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    untagged = 0
    for event in events:
        tags = event.get("p2p_message_tags") or {}
        if not tags:
            untagged += 1
        for label, tag in tags.items():
            record = {
                "rank": event.get("rank"),
                "issue_ts_ns": event.get("issue_ts_ns"),
                "wait_ms": _request_wait(event, label, tag),
                "context": event.get("context") or {},
                "label": label,
            }
            if label in {"send_prev", "send_next"}:
                sends[tag].append(record)
            elif label in {"recv_prev", "recv_next"}:
                receives[tag].append(record)

    paired: list[dict[str, Any]] = []
    unmatched_send = 0
    unmatched_recv = 0
    for tag in sorted(set(sends) | set(receives)):
        source = sends[tag]
        target = receives[tag]
        matched = min(len(source), len(target))
        unmatched_send += len(source) - matched
        unmatched_recv += len(target) - matched
        for sender, receiver in zip(source[:matched], target[:matched]):
            lane = sender["context"].get("tp_rank")
            paired.append(
                {
                    "tag": tag,
                    "tp_lane": lane,
                    "sender_rank": sender["rank"],
                    "receiver_rank": receiver["rank"],
                    "sender_wait_ms": sender["wait_ms"],
                    "receiver_wait_ms": receiver["wait_ms"],
                    "receiver_issue_after_sender_issue_ms": (
                        (receiver["issue_ts_ns"] - sender["issue_ts_ns"]) / 1e6
                        if isinstance(receiver["issue_ts_ns"], int)
                        and isinstance(sender["issue_ts_ns"], int)
                        else None
                    ),
                }
            )

    by_lane: dict[str, dict[str, Any]] = {}
    for lane in sorted({item["tp_lane"] for item in paired}, key=str):
        rows = [item for item in paired if item["tp_lane"] == lane]
        sender_waits = [item["sender_wait_ms"] for item in rows if item["sender_wait_ms"] is not None]
        receiver_waits = [item["receiver_wait_ms"] for item in rows if item["receiver_wait_ms"] is not None]
        issue_offsets = [
            item["receiver_issue_after_sender_issue_ms"]
            for item in rows
            if item["receiver_issue_after_sender_issue_ms"] is not None
        ]
        by_lane[str(lane)] = {
            "paired_messages": len(rows),
            "sender_wait": _stats(sender_waits),
            "receiver_wait": _stats(receiver_waits),
            "receiver_issue_after_sender_issue": _stats(issue_offsets),
        }
    return {
        "claim_boundary": (
            "Host issue/wait correlation only. A receiver wait can include producer readiness, "
            "communication service, and local stream synchronization; it is not wire latency."
        ),
        "p2p_issue_events": len(events),
        "untagged_issue_events": untagged,
        "paired_messages": len(paired),
        "unmatched_sends": unmatched_send,
        "unmatched_receives": unmatched_recv,
        "by_tp_lane": by_lane,
        "pairs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for pattern in args.trace_glob for path in glob.glob(pattern, recursive=True))
    result = analyze(_read(paths))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
