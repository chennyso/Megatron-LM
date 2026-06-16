#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Trace-guided PP/VPP strategy proposal tool.

This is the deterministic local agent used by the strategy-synthesis loop. An
LLM-backed proposer can replace the `propose_rewrites` function while preserving
the same JSON input/output contract and verifier-facing rewrite actions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class Bottleneck:
    kind: str
    score: float
    evidence: str
    affected_tasks: List[str] | None = None


@dataclass
class RewriteProposal:
    action: str
    reason: str
    expected_effect: str
    policy: str = "default"
    target: dict | None = None
    requires_runtime_support: bool = False


def load_events(paths: Iterable[str]) -> List[dict]:
    events: List[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        events.extend(_load_events_from_path(path))
    return events


def _load_events_from_path(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    payloads: List[object] = []

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
            payloads = _load_ndjson_events(text, path)
            break
        payloads.append(payload)
        idx = end

    events: List[dict] = []
    for payload in payloads:
        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError(f"{path} contains a non-dict event entry")
            events.extend(payload)
        elif isinstance(payload, dict):
            nested_events = payload.get("events")
            if isinstance(nested_events, list):
                if not all(isinstance(item, dict) for item in nested_events):
                    raise ValueError(f"{path} contains a non-dict nested event entry")
                events.extend(nested_events)
            else:
                events.append(payload)
        else:
            raise ValueError(f"{path} contains unsupported trace payload type {type(payload).__name__}")

    if not events:
        raise ValueError(f"{path} does not contain trace events")
    return events


def _load_ndjson_events(text: str, path: Path) -> List[dict]:
    events: List[dict] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} contains invalid JSON at line {line_no}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_no} is not a JSON object")
        events.append(payload)
    return events


def attribute_bottlenecks(events: List[dict]) -> List[Bottleneck]:
    by_rank_name: Dict[tuple, List[float]] = defaultdict(list)
    by_chunk_name: Dict[tuple, List[float]] = defaultdict(list)
    total_elapsed = sum(float(event.get("elapsed_ms", 0.0)) for event in events)
    for event in events:
        name = event.get("name", "unknown")
        elapsed = float(event.get("elapsed_ms", 0.0))
        pp_rank = event.get("pp_rank")
        chunk = event.get("model_chunk_id")
        pp_rank = -1 if pp_rank is None else int(pp_rank)
        by_rank_name[(pp_rank, name)].append(elapsed)
        if chunk is not None:
            by_chunk_name[(int(chunk), name)].append(elapsed)

    bottlenecks: List[Bottleneck] = []
    timed_events = [
        event
        for event in events
        if event.get("start_ts") is not None
        and event.get("end_ts") is not None
        and float(event.get("end_ts", 0.0)) >= float(event.get("start_ts", 0.0))
    ]
    if timed_events:
        first_start = min(float(event["start_ts"]) for event in timed_events)
        last_end = max(float(event["end_ts"]) for event in timed_events)
        span_ms = max(0.0, (last_end - first_start) * 1000.0)
        accounted_ms = sum(float(event.get("elapsed_ms", 0.0)) for event in timed_events)
        exposed_wait = sum(
            float(event.get("wait_ms", 0.0))
            for event in timed_events
            if float(event.get("wait_ms", 0.0)) > 0
        )
        if span_ms > 0:
            bottlenecks.append(
                Bottleneck(
                    kind="critical_path_bubble",
                    score=exposed_wait / max(span_ms, 1e-6),
                    evidence=(
                        f"step span {span_ms:.3f} ms with {exposed_wait:.3f} ms exposed wait "
                        f"and {accounted_ms:.3f} ms accounted task time"
                    ),
                    affected_tasks=[
                        str(event.get("metadata", {}).get("task_id", event.get("name")))
                        for event in sorted(
                            timed_events,
                            key=lambda item: float(item.get("wait_ms", 0.0)),
                            reverse=True,
                        )[:3]
                        if float(event.get("wait_ms", 0.0)) > 0
                    ],
                )
            )

    readiness_events = [
        event
        for event in events
        if event.get("ready_ts") is not None
        and event.get("start_ts") is not None
        and float(event.get("start_ts", 0.0)) > float(event.get("ready_ts", 0.0))
    ]
    if readiness_events:
        total_ready_delay = sum(
            (float(event["start_ts"]) - float(event["ready_ts"])) * 1000.0
            for event in readiness_events
        )
        top_ready_delay = max(
            readiness_events,
            key=lambda event: float(event["start_ts"]) - float(event["ready_ts"]),
        )
        bottlenecks.append(
            Bottleneck(
                kind="readiness_miss",
                score=total_ready_delay / max(total_elapsed, 1e-6),
                evidence=(
                    f"ready tasks waited {total_ready_delay:.3f} ms before scheduling; "
                    f"largest delay event={top_ready_delay.get('name')} "
                    f"rank={top_ready_delay.get('pp_rank')}"
                ),
                affected_tasks=[
                    str(top_ready_delay.get("metadata", {}).get("task_id", top_ready_delay.get("name")))
                ],
            )
        )

    rank_totals = defaultdict(float)
    for (rank, _name), values in by_rank_name.items():
        rank_totals[rank] += sum(values)
    if rank_totals:
        slow_rank, slow_time = max(rank_totals.items(), key=lambda item: item[1])
        avg_rank_time = sum(rank_totals.values()) / len(rank_totals)
        imbalance = slow_time / max(avg_rank_time, 1e-6)
        bottlenecks.append(
            Bottleneck(
                kind="rank_imbalance",
                score=imbalance,
                evidence=f"rank {slow_rank} accumulated {slow_time:.3f} ms vs avg {avg_rank_time:.3f} ms",
            )
        )

    chunk_totals = defaultdict(float)
    for (chunk, _name), values in by_chunk_name.items():
        if chunk >= 0:
            chunk_totals[chunk] += sum(values)
    if chunk_totals:
        slow_chunk, slow_time = max(chunk_totals.items(), key=lambda item: item[1])
        avg_chunk_time = sum(chunk_totals.values()) / len(chunk_totals)
        imbalance = slow_time / max(avg_chunk_time, 1e-6)
        bottlenecks.append(
            Bottleneck(
                kind="chunk_imbalance",
                score=imbalance,
                evidence=(
                    f"chunk {slow_chunk} accumulated {slow_time:.3f} ms vs "
                    f"avg {avg_chunk_time:.3f} ms"
                ),
            )
        )

    forward_time = sum(
        float(event.get("elapsed_ms", 0.0)) for event in events if event.get("name") == "forward_step"
    )
    backward_time = sum(
        float(event.get("elapsed_ms", 0.0)) for event in events if event.get("name") == "backward_step"
    )
    if forward_time and backward_time:
        ratio = backward_time / forward_time
        bottlenecks.append(
            Bottleneck(
                kind="backward_forward_skew",
                score=ratio,
                evidence=f"backward total {backward_time:.3f} ms / forward total {forward_time:.3f} ms",
            )
        )

    p2p_waits = [
        event
        for event in events
        if str(event.get("name", "")).startswith("p2p_") and "wait" in str(event.get("name", ""))
    ]
    total_wait = sum(float(event.get("wait_ms", event.get("elapsed_ms", 0.0))) for event in p2p_waits)
    if total_wait > 0:
        wait_ratio = total_wait / max(total_elapsed, 1e-6)
        top_wait = max(p2p_waits, key=lambda event: float(event.get("wait_ms", event.get("elapsed_ms", 0.0))))
        bottlenecks.append(
            Bottleneck(
                kind="p2p_dominated_bubble",
                score=wait_ratio,
                evidence=(
                    f"P2P wait {total_wait:.3f} ms is {wait_ratio:.2%} of traced time; "
                    f"largest event {top_wait.get('name')} rank={top_wait.get('pp_rank')}"
                ),
                affected_tasks=[str(top_wait.get("metadata", {}).get("task_id", top_wait.get("name")))],
            )
        )

    wgrad_time = sum(
        float(event.get("elapsed_ms", 0.0)) for event in events if event.get("name") == "wgrad_compute"
    )
    if total_wait > 0 and wgrad_time > 0:
        fillability = min(total_wait, wgrad_time) / max(wgrad_time, 1e-6)
        bottlenecks.append(
            Bottleneck(
                kind="wgrad_fillable_bubble",
                score=fillability,
                evidence=(
                    f"{min(total_wait, wgrad_time):.3f} ms of WGrad could fit into "
                    f"{total_wait:.3f} ms P2P wait window"
                ),
            )
        )

    memory_events = [
        event for event in events if event.get("memory_mb") is not None and event.get("memory_mb") >= 0
    ]
    if memory_events:
        peak = max(float(event.get("memory_mb", 0.0)) for event in memory_events)
        avg = sum(float(event.get("memory_mb", 0.0)) for event in memory_events) / len(memory_events)
        if peak > 1.2 * max(avg, 1.0):
            bottlenecks.append(
                Bottleneck(
                    kind="memory_limited_schedule",
                    score=peak / max(avg, 1.0),
                    evidence=f"peak activation/runtime memory {peak:.1f} MB vs avg {avg:.1f} MB",
                )
            )

    chunk_sensitivity = defaultdict(lambda: {"count": 0, "memory": 0.0, "time": 0.0})
    for event in events:
        chunk = event.get("model_chunk_id", -1)
        if chunk is None or int(chunk) < 0:
            continue
        item = chunk_sensitivity[int(chunk)]
        item["count"] += 1
        item["time"] += float(event.get("elapsed_ms", 0.0))
        if event.get("memory_mb") is not None:
            item["memory"] = max(item["memory"], float(event.get("memory_mb", 0.0)))
    if len(chunk_sensitivity) > 1:
        chunk_scores = {
            chunk: values["time"] + 0.01 * values["memory"]
            for chunk, values in chunk_sensitivity.items()
        }
        slow_chunk, slow_score = max(chunk_scores.items(), key=lambda item: item[1])
        avg_score = sum(chunk_scores.values()) / len(chunk_scores)
        if slow_score > 1.15 * max(avg_score, 1e-6):
            bottlenecks.append(
                Bottleneck(
                    kind="layout_induced_imbalance",
                    score=slow_score / max(avg_score, 1e-6),
                    evidence=(
                        f"chunk {slow_chunk} has time+memory pressure score "
                        f"{slow_score:.3f} vs avg {avg_score:.3f}"
                    ),
                    affected_tasks=[f"chunk:{slow_chunk}"],
                )
            )

    return sorted(bottlenecks, key=lambda item: item.score, reverse=True)


def propose_rewrites(bottlenecks: List[Bottleneck]) -> List[RewriteProposal]:
    proposals: List[RewriteProposal] = []
    kinds = {b.kind: b for b in bottlenecks}

    if kinds.get("chunk_imbalance") and kinds["chunk_imbalance"].score > 1.08:
        proposals.append(
            RewriteProposal(
                action="move_layer_boundary",
                reason=kinds["chunk_imbalance"].evidence,
                expected_effect="reduce B-critical and chunk compute imbalance",
                policy="default",
                target={"bottleneck": "chunk_imbalance"},
            )
        )
        proposals.append(
            RewriteProposal(
                action="move_layer_boundary",
                reason="chunk-local pressure suggests non-uniform PP layout instead of equal layer split",
                expected_effect="generate asymmetric pipeline layout candidates around the hot chunk",
                policy="default",
                target={"search": "pipeline_layout", "bottleneck": "chunk_imbalance"},
            )
        )
        proposals.append(
            RewriteProposal(
                action="change_schedule_order",
                reason="chunk imbalance can leave repeated local bubbles under fixed VPP order",
                expected_effect="requires tagged out-of-order P2P before it can run in Megatron",
                policy="default",
                target={"bottleneck": "chunk_imbalance"},
                requires_runtime_support=True,
            )
        )
        proposals.append(
            RewriteProposal(
                action="change_schedule_order",
                reason="chunk imbalance can leave repeated local bubbles under fixed VPP order",
                expected_effect="test whether a shorter initial VPP group reduces warmup skew",
                policy="front-loaded",
                target={"policy": "front-loaded"},
            )
        )

    if kinds.get("rank_imbalance") and kinds["rank_imbalance"].score > 1.08:
        proposals.append(
            RewriteProposal(
                action="swap_stage_placement",
                reason=kinds["rank_imbalance"].evidence,
                expected_effect="shift layer or communication cost away from straggler rank",
                policy="default",
                target={"bottleneck": "rank_imbalance"},
            )
        )
        proposals.append(
            RewriteProposal(
                action="swap_stage_placement",
                reason="rank skew often reflects bad cross-node PP boundaries, not just local compute imbalance",
                expected_effect="generate placement candidates that cluster critical neighbors on the same node",
                policy="default",
                target={"search": "placement", "bottleneck": "rank_imbalance"},
            )
        )

    if kinds.get("p2p_dominated_bubble") and kinds["p2p_dominated_bubble"].score > 0.05:
        proposals.append(
            RewriteProposal(
                action="change_microbatch_group_size",
                reason=kinds["p2p_dominated_bubble"].evidence,
                expected_effect="search VPP grouping to reduce exposed P2P wait",
                policy="default",
                target={"search": "microbatch_group_size"},
            )
        )
        proposals.append(
            RewriteProposal(
                action="move_layer_boundary",
                reason="P2P-heavy traces imply that the current PP cut likely crosses a bad boundary",
                expected_effect="search layouts that reduce cross-stage activation traffic on hot boundaries",
                policy="default",
                target={"search": "boundary_aware_layout", "bottleneck": "p2p_dominated_bubble"},
            )
        )

    if kinds.get("critical_path_bubble") and kinds["critical_path_bubble"].score > 0.05:
        proposals.append(
            RewriteProposal(
                action="change_schedule_order",
                reason=kinds["critical_path_bubble"].evidence,
                expected_effect="prioritize tasks that reduce exposed critical-path wait",
                policy="front-loaded",
                target={"bottleneck": "critical_path_bubble"},
            )
        )
        proposals.append(
            RewriteProposal(
                action="change_microbatch_group_size",
                reason="critical-path bubbles may shrink if VPP chunk granularity is matched to the bottlenecked stage",
                expected_effect="search smaller/larger VPP chunk splits instead of only schedule order tweaks",
                policy="default",
                target={"search": "vpp_chunk_size", "bottleneck": "critical_path_bubble"},
            )
        )

    if kinds.get("readiness_miss") and kinds["readiness_miss"].score > 0.01:
        proposals.append(
            RewriteProposal(
                action="enable_ready_set_dispatch",
                reason=kinds["readiness_miss"].evidence,
                expected_effect="use conservative ready-set dispatch when hinted task is not ready",
                policy="default",
                target={"runtime": "ready-set", "allow_out_of_order_p2p": False},
            )
        )

    if kinds.get("wgrad_fillable_bubble") and kinds["wgrad_fillable_bubble"].score > 0.2:
        proposals.append(
            RewriteProposal(
                action="split_wgrad",
                reason=kinds["wgrad_fillable_bubble"].evidence,
                expected_effect="expose WGrad as bubble-filling work in P2P wait windows",
                policy="default",
                target={"policy": "wgrad_wait_fill"},
                requires_runtime_support=True,
            )
        )

    if kinds.get("memory_limited_schedule") and kinds["memory_limited_schedule"].score > 1.2:
        proposals.append(
            RewriteProposal(
                action="change_checkpoint_window",
                reason=kinds["memory_limited_schedule"].evidence,
                expected_effect="reduce activation peak before enabling more aggressive schedule rewrites",
                policy="default",
                target={"policy": "memory_guard"},
            )
        )

    if kinds.get("backward_forward_skew") and kinds["backward_forward_skew"].score > 1.25:
        proposals.append(
            RewriteProposal(
                action="split_wgrad",
                reason=kinds["backward_forward_skew"].evidence,
                expected_effect="expose W-gradient work as bubble-filling tasks",
                policy="default",
            )
        )

    if not proposals:
        proposals.append(
            RewriteProposal(
                action="change_schedule_order",
                reason="no dominant imbalance found in trace",
                expected_effect="avoid unproductive strategy churn",
                policy="default",
                target={},
            )
        )
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True, help="Trace JSON files from PP ranks")
    parser.add_argument("--output", required=True, help="Path to write proposal JSON")
    args = parser.parse_args()

    events = load_events(args.trace)
    bottlenecks = attribute_bottlenecks(events)
    proposals = propose_rewrites(bottlenecks)
    payload = {
        "bottlenecks": [asdict(item) for item in bottlenecks],
        "proposals": [asdict(item) for item in proposals],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
