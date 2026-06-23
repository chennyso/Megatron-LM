#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Search executable Megatron PP/VPP StrategyPlan candidates from trace files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class BcpBudget:
    """Budgets used by bounded-critical-path VPP search.

    These are soft budgets in the offline searcher. The executable Megatron
    verifier still guards hard schedule legality; this model ranks candidates by
    how much critical-path exposure they are expected to create.
    """

    activation_peak_mb: float | None = None
    p2p_credit: int | None = None
    fb_delay_steps: int | None = None


@dataclass(frozen=True)
class BcpStats:
    """Critical-path pressure summary for a candidate VPP strategy."""

    critical_path_ms: float
    exposed_p2p_wait_ms: float
    activation_peak_mb: float
    fb_delay_steps: int
    chunk_skew: float
    p2p_credit_pressure: int
    score: float


@dataclass(frozen=True)
class OverlapStats:
    """Profiler-derived overlap categories used by BCP reports."""

    useful_overlap_ms: float
    harmful_overlap_ms: float
    fake_overlap_ms: float
    exposed_wait_ms: float


@dataclass(frozen=True)
class CandidateSpec:
    """Search-space point plus the rewrite that produced it."""

    policy: str
    group_size: int
    vpp_size: int
    layout: str | None
    placement: Tuple[int, ...]
    rewrite: str
    rationale: str
    quick_score: float = 0.0
    twincut_name: str = ""
    segment_decoder_counts: Tuple[int, ...] = ()
    segment_boundaries: Tuple[int, ...] = ()
    cross_node_boundaries: Tuple[int, ...] = ()
    node_assignment: Tuple[int, ...] = ()
    vpp_packing: Tuple[Tuple[int, ...], ...] = ()
    memory_actions: Tuple[Tuple[int, str], ...] = ()
    topology_objective: float = 0.0


BCP_REWRITE_ALGEBRA = {
    "baseline": "Preserve the default Megatron interleaved VPP schedule.",
    "front_loaded_group": "Shrink the first VPP group to reduce warmup critical-path exposure.",
    "seam_staggered_group": (
        "Shrink warmup/cooldown groups and thicken the steady-state wave to reduce "
        "dual-node seam exposure without violating Megatron chunk ordering."
    ),
    "change_group_size": "Change microbatch_group_size_per_vp_stage under Megatron legality checks.",
    "change_vp_degree": "Search a different virtual pipeline degree when layer divisibility permits.",
    "boundary_layout": "Move safe layer boundaries to reduce hot chunk or P2P boundary pressure.",
    "node_local_placement": "Prefer neighbor placement that keeps critical PP edges local when possible.",
    "twincut_partition": (
        "Synthesize topology-aware stage boundaries, seam-local VPP packing, and "
        "memory actions before compiling the candidate into an executable layout."
    ),
}


def _event_metadata(event: dict) -> dict:
    metadata = event.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _event_rank(event: dict) -> int:
    metadata = _event_metadata(event)
    raw_rank = event.get("pp_rank", event.get("pp_group_rank", metadata.get("pp_group_rank", 0)))
    return int(raw_rank or 0)


def _event_chunk(event: dict) -> int:
    metadata = _event_metadata(event)
    raw = event.get("model_chunk_id", metadata.get("model_chunk_id", metadata.get("vp_chunk", 0)))
    return int(raw if raw is not None else 0)


def _event_microbatch(event: dict) -> int:
    metadata = _event_metadata(event)
    raw = event.get("microbatch_id", metadata.get("microbatch_id", metadata.get("microbatch", 0)))
    return int(raw if raw is not None else 0)


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _event_key(event: dict) -> Tuple[int, int, int]:
    return (_event_rank(event), _event_chunk(event), _event_microbatch(event))


def _average_event_time(events: List[dict], names: set[str], default_ms: float) -> Dict[Tuple[int, int, int], float]:
    values: Dict[Tuple[int, int, int], List[float]] = {}
    for event in events:
        if str(event.get("name")) in names:
            values.setdefault(_event_key(event), []).append(float(event.get("elapsed_ms", 0.0)))
    return {key: sum(items) / max(len(items), 1) for key, items in values.items()} or {
        (0, 0, 0): default_ms
    }


def _lookup_time(table: Dict[Tuple[int, int, int], float], rank: int, chunk: int, microbatch: int) -> float:
    if (rank, chunk, microbatch) in table:
        return table[(rank, chunk, microbatch)]
    same_chunk = [value for (r, c, _m), value in table.items() if r == rank and c == chunk]
    if same_chunk:
        return sum(same_chunk) / len(same_chunk)
    same_rank = [value for (r, _c, _m), value in table.items() if r == rank]
    if same_rank:
        return sum(same_rank) / len(same_rank)
    return sum(table.values()) / max(len(table), 1)


def _estimate_memory(events: List[dict]) -> float:
    memories = [float(event["memory_mb"]) for event in events if event.get("memory_mb") is not None]
    return max(memories) if memories else 0.0


def _iter_timed_events(events: Iterable[dict]) -> Iterable[dict]:
    for event in events:
        if event.get("start_ts") is None or event.get("end_ts") is None:
            continue
        if float(event.get("end_ts", 0.0)) < float(event.get("start_ts", 0.0)):
            continue
        yield event


def _estimate_fb_delay_steps(events: List[dict]) -> int:
    """Estimate the largest F/B separation in per-rank event order.

    Megatron traces are not yet a full tagged task DAG. This proxy gives the
    BCP objective a conservative signal: large gaps between a chunk's forward
    and backward events indicate activation residency and version-delay risk.
    """

    forward_index: Dict[Tuple[int, int, int], int] = {}
    max_delay = 0
    for idx, event in enumerate(events):
        key = _event_key(event)
        name = str(event.get("name", ""))
        if name in {"forward_step", "F"}:
            forward_index.setdefault(key, idx)
        elif name in {"backward_step", "B_DGRAD", "dgrad_compute", "wgrad_compute"}:
            if key in forward_index:
                max_delay = max(max_delay, idx - forward_index[key])
    return max_delay


def _p2p_credit_pressure(events: List[dict]) -> int:
    """Approximate maximum outstanding P2P requests from trace issue/wait events."""

    timed = sorted(
        _iter_timed_events(events),
        key=lambda event: (float(event.get("start_ts", 0.0)), float(event.get("end_ts", 0.0))),
    )
    active: List[float] = []
    peak = 0
    for event in timed:
        name = str(event.get("name", ""))
        if not name.startswith("p2p_"):
            continue
        start = float(event["start_ts"])
        end = float(event["end_ts"])
        active = [item for item in active if item > start]
        active.append(end)
        peak = max(peak, len(active))
    return peak


def _chunk_pressure(events: List[dict], vpp_size: int) -> Dict[int, float]:
    pressure = {chunk: 0.0 for chunk in range(max(vpp_size, 1))}
    for event in events:
        chunk = _event_chunk(event)
        if chunk not in pressure:
            continue
        elapsed = float(event.get("elapsed_ms", 0.0))
        wait = float(event.get("wait_ms", 0.0))
        memory = float(event.get("memory_mb", 0.0) or 0.0)
        pressure[chunk] += elapsed + wait + 0.005 * memory
    return pressure


def _critical_path_ms(events: List[dict]) -> float:
    timed_events = list(_iter_timed_events(events))
    if timed_events:
        first_start = min(float(event["start_ts"]) for event in timed_events)
        last_end = max(float(event["end_ts"]) for event in timed_events)
        return max(0.0, (last_end - first_start) * 1000.0)

    rank_totals: Dict[int, float] = {}
    for event in events:
        rank = _event_rank(event)
        rank_totals[rank] = rank_totals.get(rank, 0.0) + float(event.get("elapsed_ms", 0.0))
    return max(rank_totals.values()) if rank_totals else 1.0


def _bcp_stats(
    events: List[dict],
    *,
    group_size: int,
    policy: str,
    vpp_size: int,
    baseline_vpp_size: int,
    layout: str | None,
    runtime: str,
    budget: BcpBudget,
    memory_actions: Tuple[Tuple[int, str], ...] = (),
    cross_node_boundaries: Tuple[int, ...] = (),
) -> BcpStats:
    critical_path = _critical_path_ms(events)
    exposed_wait = sum(
        float(event.get("wait_ms", event.get("elapsed_ms", 0.0)))
        for event in events
        if str(event.get("name", "")).startswith("p2p_") and "wait" in str(event.get("name", ""))
    )
    activation_peak = _estimate_memory(events)
    fb_delay = _estimate_fb_delay_steps(events)
    credit_pressure = _p2p_credit_pressure(events)
    pressure = _chunk_pressure(events, vpp_size)
    if pressure:
        avg_pressure = sum(pressure.values()) / len(pressure)
        chunk_skew = max(pressure.values()) / max(avg_pressure, 1e-6)
    else:
        chunk_skew = 1.0

    # Candidate effects are deliberately conservative: they bias search toward
    # candidates that shorten exposed critical-path pressure without claiming
    # actual speedup before a short-run validation confirms it.
    if policy == "front-loaded":
        policy_gain = 0.04
    elif policy == "seam-staggered":
        policy_gain = 0.06 if exposed_wait > 0.0 else 0.03
    else:
        policy_gain = 0.0
    layout_gain = 0.05 if layout else 0.0
    runtime_gain = 0.04 if runtime == "bcp-ready" else 0.03 if runtime == "ready-set" else 0.0
    if vpp_size > baseline_vpp_size:
        vpp_gain = min(0.08, 0.02 * (vpp_size - baseline_vpp_size))
    elif vpp_size < baseline_vpp_size:
        vpp_gain = -min(0.06, 0.02 * (baseline_vpp_size - vpp_size))
    else:
        vpp_gain = 0.0
    group_penalty = 0.03 * max(0, 2 - group_size)
    retain_count = sum(1 for _stage_idx, action in memory_actions if action == "retain")
    recompute_count = sum(1 for _stage_idx, action in memory_actions if action == "recompute")
    offload_count = sum(1 for _stage_idx, action in memory_actions if action == "offload")
    seam_gain = min(0.04, 0.01 * len(cross_node_boundaries) + 0.012 * retain_count)
    memory_action_penalty = 0.008 * recompute_count + 0.012 * offload_count

    predicted_critical_path = critical_path * (
        1.0
        - policy_gain
        - layout_gain
        - runtime_gain
        - vpp_gain
        - seam_gain
        + group_penalty
        + memory_action_penalty
    )
    activation_violation = 0.0
    if budget.activation_peak_mb is not None:
        activation_violation = max(0.0, activation_peak - budget.activation_peak_mb)
    fb_violation = 0.0
    if budget.fb_delay_steps is not None:
        fb_violation = max(0.0, float(fb_delay - budget.fb_delay_steps))
    credit_violation = 0.0
    if budget.p2p_credit is not None:
        credit_violation = max(0.0, float(credit_pressure - budget.p2p_credit))

    score = (
        predicted_critical_path
        + 0.35 * exposed_wait
        + 6.0 * max(0.0, chunk_skew - 1.0)
        + 0.02 * activation_violation
        + 4.0 * fb_violation
        + 3.0 * credit_violation
    )
    return BcpStats(
        critical_path_ms=predicted_critical_path,
        exposed_p2p_wait_ms=exposed_wait,
        activation_peak_mb=activation_peak,
        fb_delay_steps=fb_delay,
        chunk_skew=chunk_skew,
        p2p_credit_pressure=credit_pressure,
        score=score,
    )


def _overlap_stats(events: List[dict]) -> OverlapStats:
    """Classify trace overlap into useful, harmful, fake, and exposed wait.

    This is a conservative trace-level approximation. Nsight-level kernel
    attribution can replace this later without changing report schema.
    """

    timed = list(_iter_timed_events(events))
    compute = [
        event
        for event in timed
        if str(event.get("name", ""))
        in {"forward_step", "backward_step", "dgrad_compute", "wgrad_compute"}
    ]
    comm = [event for event in timed if str(event.get("name", "")).startswith("p2p_")]
    useful = 0.0
    harmful = 0.0
    fake = 0.0
    for comm_event in comm:
        c_start = float(comm_event["start_ts"])
        c_end = float(comm_event["end_ts"])
        comm_ms = max(0.0, (c_end - c_start) * 1000.0)
        overlap_ms = 0.0
        for compute_event in compute:
            overlap_start = max(c_start, float(compute_event["start_ts"]))
            overlap_end = min(c_end, float(compute_event["end_ts"]))
            overlap_ms += max(0.0, (overlap_end - overlap_start) * 1000.0)
        wait_ms = float(comm_event.get("wait_ms", 0.0))
        if overlap_ms <= 0:
            fake += comm_ms
        elif wait_ms > 0.1 * max(comm_ms, 1e-6):
            harmful += min(overlap_ms, comm_ms)
        else:
            useful += min(overlap_ms, comm_ms)

    exposed_wait = sum(
        float(event.get("wait_ms", event.get("elapsed_ms", 0.0)))
        for event in events
        if str(event.get("name", "")).startswith("p2p_") and "wait" in str(event.get("name", ""))
    )
    return OverlapStats(
        useful_overlap_ms=useful,
        harmful_overlap_ms=harmful,
        fake_overlap_ms=fake,
        exposed_wait_ms=exposed_wait,
    )


def _trace_diagnostics(events: List[dict], vpp_size: int) -> Dict[str, Any]:
    pressure = _chunk_pressure(events, vpp_size)
    hot_chunks = sorted(pressure.items(), key=lambda item: item[1], reverse=True)
    rank_totals: Dict[int, float] = {}
    p2p_by_name: Dict[str, float] = {}
    for event in events:
        rank = _event_rank(event)
        elapsed = float(event.get("elapsed_ms", 0.0))
        rank_totals[rank] = rank_totals.get(rank, 0.0) + elapsed
        name = str(event.get("name", ""))
        if name.startswith("p2p_"):
            p2p_by_name[name] = p2p_by_name.get(name, 0.0) + float(event.get("wait_ms", elapsed))
    overlap = _overlap_stats(events)
    return {
        "critical_path_ms": _critical_path_ms(events),
        "activation_peak_mb": _estimate_memory(events),
        "fb_delay_steps": _estimate_fb_delay_steps(events),
        "p2p_credit_pressure": _p2p_credit_pressure(events),
        "rank_totals_ms": dict(sorted(rank_totals.items())),
        "hot_chunks": [
            {"chunk": chunk, "pressure": value}
            for chunk, value in hot_chunks[: min(5, len(hot_chunks))]
        ],
        "p2p_wait_by_name_ms": dict(sorted(p2p_by_name.items())),
        "overlap": asdict(overlap),
    }


def _candidate_quick_score(
    *,
    policy: str,
    rewrite: str,
    group_size: int,
    vpp_size: int,
    baseline_group_size: int,
    baseline_vpp_size: int,
    has_layout: bool,
    has_placement: bool,
    diagnostics: Dict[str, Any],
    topology_objective: float = 0.0,
    memory_actions: Tuple[Tuple[int, str], ...] = (),
    cross_node_boundaries: Tuple[int, ...] = (),
) -> float:
    score = 100.0
    exposed_wait = float(diagnostics["overlap"]["exposed_wait_ms"])
    chunk_count = max(1, len(diagnostics.get("hot_chunks", [])))
    if rewrite == "front_loaded_group" or policy == "front-loaded":
        score -= 4.0 + min(8.0, exposed_wait * 0.2)
    if rewrite == "seam_staggered_group" or policy == "seam-staggered":
        score -= 5.0 + min(9.0, exposed_wait * 0.22)
    if rewrite == "change_group_size" and group_size != baseline_group_size:
        score -= 3.0
        if group_size >= 2:
            score -= 1.0
    if rewrite == "change_vp_degree" and vpp_size != baseline_vpp_size:
        score -= 4.0 / chunk_count
    if has_layout:
        score -= 5.0
    if has_placement:
        score -= 2.0
    if rewrite == "twincut_partition":
        score -= 7.0 + min(10.0, exposed_wait * 0.1)
        score += 0.01 * topology_objective
    retain_count = sum(1 for _stage_idx, action in memory_actions if action == "retain")
    offload_count = sum(1 for _stage_idx, action in memory_actions if action == "offload")
    score -= min(3.0, 0.75 * retain_count + 0.35 * len(cross_node_boundaries))
    score += min(2.0, 0.5 * offload_count)
    return score


def _estimate_layer_costs(events: List[dict], num_layers: int, pipeline_parallel_size: int) -> Tuple[float, float]:
    if num_layers <= 0 or pipeline_parallel_size <= 0:
        return 1.0, 1.0
    compute_events = [
        float(event.get("elapsed_ms", 0.0))
        for event in events
        if str(event.get("name", ""))
        in {"forward_step", "backward_step", "dgrad_compute", "wgrad_compute"}
    ]
    average_stage_compute_ms = sum(compute_events) / max(len(compute_events), 1)
    average_layers_per_stage = max(float(num_layers) / float(pipeline_parallel_size), 1.0)
    layer_time_ms = max(average_stage_compute_ms / average_layers_per_stage, 1e-3)
    activation_peak_mb = _estimate_memory(events)
    activation_mb_per_layer = max(activation_peak_mb / average_layers_per_stage, 1e-3)
    return layer_time_ms, activation_mb_per_layer


def _candidate_policies(
    args,
    *,
    seam_pressure: float = 0.0,
    memory_actions: Tuple[Tuple[int, str], ...] = (),
    has_layout: bool = False,
) -> List[str]:
    policies = ["default"]
    runtime = getattr(args, "runtime", "fixed")
    if runtime == "fixed":
        return policies
    policies.append("front-loaded")
    if seam_pressure > 0.0 or has_layout or memory_actions or runtime == "bcp-ready":
        policies.append("seam-staggered")
    deduped: List[str] = []
    seen = set()
    for item in policies:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _build_uniform_layout(num_layers: int, pipeline_parallel_size: int, vpp_size: int) -> str:
    total_stages = pipeline_parallel_size * max(vpp_size, 1)
    base = num_layers // total_stages
    rem = num_layers % total_stages
    segments: List[str] = []
    for stage in range(total_stages):
        decoder_layers = base + (1 if stage < rem else 0)
        tokens = ["t"] * decoder_layers
        if stage == 0:
            tokens = ["E", *tokens]
        if stage == total_stages - 1:
            tokens = [*tokens, "L"]
        segments.append("".join(tokens))
    return "|".join(segments)


def _build_boundary_aware_layout(
    num_layers: int, pipeline_parallel_size: int, vpp_size: int
) -> str | None:
    total_stages = pipeline_parallel_size * max(vpp_size, 1)
    base = num_layers // total_stages
    rem = num_layers % total_stages
    counts = [base + (1 if stage < rem else 0) for stage in range(total_stages)]
    changed = False
    if total_stages >= 4 and counts[1] > 1:
        counts[0] += 1
        counts[1] -= 1
        changed = True
    if total_stages >= 4 and counts[-2] > 1:
        counts[-1] += 1
        counts[-2] -= 1
        changed = True
    if not changed or any(count < 1 for count in counts):
        return None
    segments: List[str] = []
    for stage, decoder_layers in enumerate(counts):
        tokens = ["t"] * decoder_layers
        if stage == 0:
            tokens = ["E", *tokens]
        if stage == total_stages - 1:
            tokens = [*tokens, "L"]
        segments.append("".join(tokens))
    return "|".join(segments)


def _default_placement(pipeline_parallel_size: int) -> Tuple[int, ...]:
    return tuple(range(pipeline_parallel_size))


def _node_local_placement(pipeline_parallel_size: int) -> Tuple[int, ...]:
    half = pipeline_parallel_size // 2
    left = list(range(0, half))
    right = list(range(half, pipeline_parallel_size))
    return tuple(left + list(reversed(right)))


def _candidate_vp_sizes(args, proposals: List[Any]) -> List[int]:
    vp_sizes = [args.num_model_chunks]
    hinted = {item.target.get("search") for item in proposals if getattr(item, "target", None)}
    if args.num_layers and (
        "vpp_chunk_size" in hinted
        or "pipeline_layout" in hinted
        or "boundary_aware_layout" in hinted
        or "microbatch_group_size" in hinted
    ):
        layers_per_pp_rank = args.num_layers // max(args.pipeline_parallel_size, 1)
        for vp_size in (8, 4, 2, 1):
            if vp_size != args.num_model_chunks and layers_per_pp_rank % vp_size == 0:
                vp_sizes.append(vp_size)
    deduped: List[int] = []
    seen = set()
    for item in vp_sizes:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _candidate_layouts(args, proposals: List[Any], vpp_size: int) -> List[str | None]:
    layouts: List[str | None] = [None]
    if not getattr(args, "num_layers", None):
        return layouts
    hinted = {item.target.get("search") for item in proposals if getattr(item, "target", None)}
    if "pipeline_layout" in hinted or "boundary_aware_layout" in hinted:
        uniform_layout = _build_uniform_layout(args.num_layers, args.pipeline_parallel_size, vpp_size)
        boundary_layout = _build_boundary_aware_layout(
            args.num_layers, args.pipeline_parallel_size, vpp_size
        )
        if boundary_layout is not None and boundary_layout != uniform_layout:
            layouts.append(boundary_layout)
    deduped: List[str | None] = []
    seen = set()
    for item in layouts:
        key = item or "__none__"
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _candidate_placements(args, proposals: List[Any]) -> List[Tuple[int, ...]]:
    placements = [_default_placement(args.pipeline_parallel_size)]
    hinted = {item.target.get("search") for item in proposals if getattr(item, "target", None)}
    if "placement" in hinted:
        placements.append(_node_local_placement(args.pipeline_parallel_size))
    deduped: List[Tuple[int, ...]] = []
    seen = set()
    for item in placements:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _candidate_group_sizes(
    args,
    proposals: List[Any],
    vpp_size: int,
    *,
    seam_pressure: float = 0.0,
    memory_actions: Tuple[Tuple[int, str], ...] = (),
) -> List[int]:
    baseline_group_size = args.microbatch_group_size
    max_group_size = max(1, int(getattr(args, "num_microbatches", baseline_group_size)))
    group_sizes = [baseline_group_size]
    hinted = {item.target.get("search") for item in proposals if getattr(item, "target", None)}
    if (
        "microbatch_group_size" in hinted
        or "vpp_chunk_size" in hinted
        or seam_pressure > 0.0
        or memory_actions
    ):
        for group_size in (1, 2, 4, 8):
            if group_size != baseline_group_size:
                group_sizes.append(group_size)
        for group_size in (
            max(1, baseline_group_size // 2),
            max(1, baseline_group_size - 1),
            min(max_group_size, baseline_group_size + 1),
            min(max_group_size, baseline_group_size + max(1, baseline_group_size // 2)),
        ):
            if group_size != baseline_group_size:
                group_sizes.append(group_size)
    if vpp_size != args.num_model_chunks:
        for group_size in (max(1, args.pipeline_parallel_size // 2), args.pipeline_parallel_size):
            if group_size != baseline_group_size:
                group_sizes.append(group_size)
    if any(action == "retain" for _stage_idx, action in memory_actions):
        group_sizes.append(max(1, min(max_group_size, baseline_group_size // 2)))
    if any(action in {"recompute", "offload"} for _stage_idx, action in memory_actions):
        group_sizes.append(min(max_group_size, baseline_group_size + 1))
    deduped: List[int] = []
    seen = set()
    for item in group_sizes:
        if item < 1 or item > max_group_size:
            continue
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _build_twincut_specs(
    args,
    diagnostics: Dict[str, Any],
    events: List[dict],
    vp_sizes: List[int],
    twincut,
) -> List[CandidateSpec]:
    if not getattr(args, "num_layers", None) or twincut is None or not events:
        return []

    layer_time_ms, activation_mb_per_layer = _estimate_layer_costs(
        events,
        args.num_layers,
        args.pipeline_parallel_size,
    )
    exposed_wait_ms = float(diagnostics["overlap"]["exposed_wait_ms"])
    p2p_credit_pressure = float(diagnostics.get("p2p_credit_pressure", 0.0))
    placement = _default_placement(args.pipeline_parallel_size)
    specs: List[CandidateSpec] = []
    for vpp_size in vp_sizes:
        try:
            proposals = twincut.propose_twincut_specs(
                num_layers=args.num_layers,
                pipeline_parallel_size=args.pipeline_parallel_size,
                vpp_size=vpp_size,
                layer_time_ms=layer_time_ms,
                activation_mb_per_layer=activation_mb_per_layer,
                exposed_wait_ms=exposed_wait_ms,
                p2p_credit_pressure=p2p_credit_pressure,
                memory_budget_mb=args.memory_budget_mb,
            )
        except Exception:
            continue
        for proposal in proposals:
            rewrite = "twincut_partition"
            layout = twincut.build_pipeline_layout(proposal.segment_decoder_counts)
            seam_pressure = exposed_wait_ms + 4.0 * len(proposal.cross_node_boundaries)
            policies = _candidate_policies(
                args,
                seam_pressure=seam_pressure,
                memory_actions=proposal.memory_actions,
                has_layout=True,
            )
            group_sizes = _candidate_group_sizes(
                args,
                [],
                vpp_size,
                seam_pressure=seam_pressure,
                memory_actions=proposal.memory_actions,
            )
            for policy in policies:
                for group_size in group_sizes:
                    quick_score = _candidate_quick_score(
                        policy=policy,
                        rewrite=rewrite,
                        group_size=group_size,
                        vpp_size=vpp_size,
                        baseline_group_size=args.microbatch_group_size,
                        baseline_vpp_size=args.num_model_chunks,
                        has_layout=True,
                        has_placement=False,
                        diagnostics=diagnostics,
                        topology_objective=proposal.objective,
                        memory_actions=proposal.memory_actions,
                        cross_node_boundaries=proposal.cross_node_boundaries,
                    )
                    specs.append(
                        CandidateSpec(
                            policy=policy,
                            group_size=group_size,
                            vpp_size=vpp_size,
                            layout=layout,
                            placement=placement,
                            rewrite=rewrite,
                            rationale=BCP_REWRITE_ALGEBRA["twincut_partition"],
                            quick_score=quick_score,
                            twincut_name=proposal.name,
                            segment_decoder_counts=proposal.segment_decoder_counts,
                            segment_boundaries=proposal.segment_boundaries,
                            cross_node_boundaries=proposal.cross_node_boundaries,
                            node_assignment=proposal.node_assignment,
                            vpp_packing=proposal.vpp_packing,
                            memory_actions=proposal.memory_actions,
                            topology_objective=proposal.objective,
                        )
                    )
    return specs


def _build_candidate_specs(
    args,
    proposals: List[Any],
    diagnostics: Dict[str, Any],
    events: List[dict] | None = None,
    twincut=None,
) -> List[CandidateSpec]:
    placements = _candidate_placements(args, proposals)
    vp_sizes = _candidate_vp_sizes(args, proposals)
    default_placement = _default_placement(args.pipeline_parallel_size)
    seam_pressure = float(diagnostics["overlap"]["exposed_wait_ms"])
    base_policies = _candidate_policies(args, seam_pressure=seam_pressure)
    specs: List[CandidateSpec] = []
    for policy in base_policies:
        if policy == "front-loaded":
            rewrite = "front_loaded_group"
        elif policy == "seam-staggered":
            rewrite = "seam_staggered_group"
        else:
            rewrite = "baseline"
        specs.append(
            CandidateSpec(
                policy=policy,
                group_size=args.microbatch_group_size,
                vpp_size=args.num_model_chunks,
                layout=None,
                placement=default_placement,
                rewrite=rewrite,
                rationale=BCP_REWRITE_ALGEBRA[rewrite],
            )
        )
    prioritized_vp_sizes = [item for item in vp_sizes if item != args.num_model_chunks]
    prioritized_vp_sizes.extend(item for item in vp_sizes if item == args.num_model_chunks)
    for vpp_size in prioritized_vp_sizes:
        layouts = _candidate_layouts(args, proposals, vpp_size)
        group_sizes = _candidate_group_sizes(
            args,
            proposals,
            vpp_size,
            seam_pressure=seam_pressure,
        )
        for group_size in group_sizes:
            for layout in layouts:
                for placement in placements:
                    policies = _candidate_policies(
                        args,
                        seam_pressure=seam_pressure,
                        has_layout=bool(layout),
                    )
                    for policy in policies:
                        rewrite = "change_group_size"
                        if vpp_size != args.num_model_chunks:
                            rewrite = "change_vp_degree"
                        if layout:
                            rewrite = "boundary_layout"
                        if placement != default_placement:
                            rewrite = "node_local_placement"
                        if rewrite == "change_group_size" and group_size == args.microbatch_group_size:
                            if policy == "front-loaded":
                                rewrite = "front_loaded_group"
                            elif policy == "seam-staggered":
                                rewrite = "seam_staggered_group"
                        specs.append(
                            CandidateSpec(
                                policy=policy,
                                group_size=group_size,
                                vpp_size=vpp_size,
                                layout=layout,
                                placement=placement,
                                rewrite=rewrite,
                                rationale=BCP_REWRITE_ALGEBRA[rewrite],
                            )
                        )

    deduped_specs: List[CandidateSpec] = []
    seen_specs = set()
    for spec in specs:
        key = (spec.policy, spec.group_size, spec.vpp_size, spec.layout, spec.placement)
        if key in seen_specs:
            continue
        seen_specs.add(key)
        quick_score = _candidate_quick_score(
            policy=spec.policy,
            rewrite=spec.rewrite,
            group_size=spec.group_size,
            vpp_size=spec.vpp_size,
            baseline_group_size=args.microbatch_group_size,
            baseline_vpp_size=args.num_model_chunks,
            has_layout=bool(spec.layout),
            has_placement=spec.placement != default_placement,
            diagnostics=diagnostics,
            memory_actions=spec.memory_actions,
            cross_node_boundaries=spec.cross_node_boundaries,
        )
        deduped_specs.append(
            CandidateSpec(
                policy=spec.policy,
                group_size=spec.group_size,
                vpp_size=spec.vpp_size,
                layout=spec.layout,
                placement=spec.placement,
                rewrite=spec.rewrite,
                rationale=spec.rationale,
                quick_score=quick_score,
            )
        )

    deduped_specs.extend(_build_twincut_specs(args, diagnostics, events or [], vp_sizes, twincut))

    final_specs: List[CandidateSpec] = []
    final_seen = set()
    for spec in deduped_specs:
        key = (
            spec.policy,
            spec.group_size,
            spec.vpp_size,
            spec.layout,
            spec.placement,
            spec.segment_boundaries,
            spec.cross_node_boundaries,
            spec.node_assignment,
            spec.vpp_packing,
            spec.memory_actions,
        )
        if key in final_seen:
            continue
        final_seen.add(key)
        final_specs.append(spec)
    ranked_specs = sorted(final_specs, key=lambda item: item.quick_score)
    budget = max(1, args.candidate_budget)
    diversified: List[CandidateSpec] = []
    policy_seen = set()
    for spec in ranked_specs:
        if spec.policy in policy_seen:
            continue
        policy_seen.add(spec.policy)
        diversified.append(spec)
        if len(diversified) >= budget:
            return diversified
    diversified_seen = set()
    for spec in diversified:
        diversified_seen.add(
            (
                spec.policy,
                spec.rewrite,
                spec.vpp_size,
                bool(spec.layout),
                bool(spec.segment_boundaries),
            )
        )
    for spec in ranked_specs:
        if spec in diversified:
            continue
        family = (
            spec.policy,
            spec.rewrite,
            spec.vpp_size,
            bool(spec.layout),
            bool(spec.segment_boundaries),
        )
        if family in diversified_seen:
            continue
        diversified_seen.add(family)
        diversified.append(spec)
        if len(diversified) >= budget:
            return diversified
    for spec in ranked_specs:
        if spec in diversified:
            continue
        diversified.append(spec)
        if len(diversified) >= budget:
            break
    return diversified


def _build_task_dag(synth, events: List[dict], args, vpp_size: int) -> Tuple[Any, ...]:
    forward_times = _average_event_time(events, {"forward_step"}, 5.0)
    dgrad_times = _average_event_time(events, {"backward_step", "dgrad_compute"}, 7.0)
    wgrad_times = _average_event_time(events, {"wgrad_compute"}, 3.0)
    comm_times = _average_event_time(
        events,
        {
            "p2p_comm_wait",
            "p2p_recv_wait_forward",
            "p2p_send_wait_forward",
            "p2p_recv_wait_backward",
            "p2p_send_wait_backward",
        },
        1.0,
    )
    memory_mb = _estimate_memory(events)
    tasks = []
    for rank in range(args.pipeline_parallel_size):
        for chunk in range(vpp_size):
            for microbatch in range(args.num_microbatches):
                recv_f = f"RECV_F:r{rank}:c{chunk}:m{microbatch}"
                send_f = f"SEND_F:r{rank}:c{chunk}:m{microbatch}"
                recv_b = f"RECV_B:r{rank}:c{chunk}:m{microbatch}"
                send_b = f"SEND_B:r{rank}:c{chunk}:m{microbatch}"
                forward = f"F:r{rank}:c{chunk}:m{microbatch}"
                dgrad = f"B_DGRAD:r{rank}:c{chunk}:m{microbatch}"
                wgrad = f"B_WGRAD:r{rank}:c{chunk}:m{microbatch}"

                if rank > 0:
                    tasks.append(
                        synth.StrategyTask(
                            task_id=recv_f,
                            kind="RECV_F",
                            pp_rank=rank,
                            vp_chunk=chunk,
                            microbatch=microbatch,
                            est_comm_ms=_lookup_time(comm_times, rank, chunk, microbatch),
                            deps=(f"SEND_F:r{rank - 1}:c{chunk}:m{microbatch}",),
                        )
                    )
                tasks.append(
                    synth.StrategyTask(
                        task_id=forward,
                        kind="F",
                        pp_rank=rank,
                        vp_chunk=chunk,
                        microbatch=microbatch,
                        est_compute_ms=_lookup_time(forward_times, rank, chunk, microbatch),
                        est_memory_mb=memory_mb / max(args.pipeline_parallel_size, 1),
                        deps=(recv_f,) if rank > 0 else (),
                    )
                )
                if rank < args.pipeline_parallel_size - 1:
                    tasks.append(
                        synth.StrategyTask(
                            task_id=send_f,
                            kind="SEND_F",
                            pp_rank=rank,
                            vp_chunk=chunk,
                            microbatch=microbatch,
                            est_comm_ms=_lookup_time(comm_times, rank, chunk, microbatch),
                            deps=(forward,),
                        )
                    )
                    tasks.append(
                        synth.StrategyTask(
                            task_id=recv_b,
                            kind="RECV_B",
                            pp_rank=rank,
                            vp_chunk=chunk,
                            microbatch=microbatch,
                            est_comm_ms=_lookup_time(comm_times, rank, chunk, microbatch),
                            deps=(f"SEND_B:r{rank + 1}:c{chunk}:m{microbatch}",),
                        )
                    )
                tasks.append(
                    synth.StrategyTask(
                        task_id=dgrad,
                        kind="B_DGRAD",
                        pp_rank=rank,
                        vp_chunk=chunk,
                        microbatch=microbatch,
                        est_compute_ms=_lookup_time(dgrad_times, rank, chunk, microbatch),
                        est_memory_mb=memory_mb / max(args.pipeline_parallel_size, 1),
                        deps=(forward, recv_b) if rank < args.pipeline_parallel_size - 1 else (forward,),
                    )
                )
                tasks.append(
                    synth.StrategyTask(
                        task_id=wgrad,
                        kind="B_WGRAD",
                        pp_rank=rank,
                        vp_chunk=chunk,
                        microbatch=microbatch,
                        est_compute_ms=_lookup_time(wgrad_times, rank, chunk, microbatch),
                        deps=(dgrad,),
                    )
                )
                if rank > 0:
                    tasks.append(
                        synth.StrategyTask(
                            task_id=send_b,
                            kind="SEND_B",
                            pp_rank=rank,
                            vp_chunk=chunk,
                            microbatch=microbatch,
                            est_comm_ms=_lookup_time(comm_times, rank, chunk, microbatch),
                            deps=(dgrad,),
                        )
                    )
    return tuple(tasks)


def _task_duration_ms(task: Any) -> float:
    return max(0.0, float(task.est_compute_ms) + float(task.est_comm_ms))


def _task_dag_critical_path_ms(tasks: Tuple[Any, ...]) -> float:
    """Compute the longest dependency path over the strategy task IR.

    This is the static counterpart to trace-level critical-path span. It gives
    the search report a schedule-theoretic signal that is independent from
    timestamp noise and can be recomputed for every legal rewrite candidate.
    """

    task_by_id = {task.task_id: task for task in tasks}
    memo: Dict[str, float] = {}
    visiting: set[str] = set()

    def visit(task_id: str) -> float:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            raise ValueError(f"cycle detected in strategy task DAG at {task_id}")
        visiting.add(task_id)
        task = task_by_id[task_id]
        dep_path = 0.0
        for dep in task.deps:
            if dep in task_by_id:
                dep_path = max(dep_path, visit(dep))
        visiting.remove(task_id)
        total = dep_path + _task_duration_ms(task)
        memo[task_id] = total
        return total

    return max((visit(task.task_id) for task in tasks), default=0.0)


def _estimate_step_time(
    events: List[dict],
    group_size: int,
    policy: str,
    vpp_size: int,
    baseline_vpp_size: int,
    layout: str | None,
    memory_actions: Tuple[Tuple[int, str], ...] = (),
    cross_node_boundaries: Tuple[int, ...] = (),
    topology_objective: float = 0.0,
) -> float:
    rank_totals: Dict[int, float] = {}
    p2p_wait = 0.0
    sync_wait = 0.0
    for event in events:
        rank = int(event.get("pp_rank", 0))
        elapsed = float(event.get("elapsed_ms", 0.0))
        rank_totals[rank] = rank_totals.get(rank, 0.0) + elapsed
        name = str(event.get("name", ""))
        if name.startswith("p2p_"):
            p2p_wait += float(event.get("wait_ms", elapsed))
        if "sync" in name:
            sync_wait += elapsed
    max_rank_time = max(rank_totals.values()) if rank_totals else 1.0
    # A conservative model: front-loaded can reduce warmup skew but may not help
    # steady state. Smaller group sizes expose more interleaving but can increase
    # communication pressure.
    if policy == "front-loaded":
        policy_factor = 0.97
    elif policy == "seam-staggered":
        policy_factor = 0.94
    else:
        policy_factor = 1.0
    group_penalty = 1.0 + max(0, 2 - group_size) * 0.03
    if vpp_size > baseline_vpp_size:
        vpp_factor = 1.0 - min(0.08, 0.025 * (vpp_size - baseline_vpp_size))
    elif vpp_size < baseline_vpp_size:
        vpp_factor = 1.0 + min(0.08, 0.025 * (baseline_vpp_size - vpp_size))
    else:
        vpp_factor = 1.0
    layout_factor = 0.96 if layout else 1.0
    retain_count = sum(1 for _stage_idx, action in memory_actions if action == "retain")
    recompute_count = sum(1 for _stage_idx, action in memory_actions if action == "recompute")
    offload_count = sum(1 for _stage_idx, action in memory_actions if action == "offload")
    seam_factor = 1.0 - min(0.03, 0.01 * len(cross_node_boundaries) + 0.01 * retain_count)
    memory_action_factor = 1.0 + 0.008 * recompute_count + 0.012 * offload_count
    topology_factor = 1.0
    if layout and topology_objective > 0.0:
        topology_factor -= min(0.04, 4.0 / max(topology_objective, 100.0))
    unchanged_penalty = 1.02 if vpp_size == baseline_vpp_size and layout is None else 1.0
    return (
        max_rank_time
        * policy_factor
        * group_penalty
        * vpp_factor
        * layout_factor
        * seam_factor
        * memory_action_factor
        * topology_factor
        * unchanged_penalty
        + 0.35 * p2p_wait
        + 0.1 * sync_wait
    )


def _build_candidate(
    synth,
    twincut,
    spec: CandidateSpec,
    args,
    rewrites: List[dict],
    events: List[dict],
    budget: BcpBudget,
) -> Any:
    legal_actions = {
        "change_schedule_order",
        "move_layer_boundary",
        "swap_stage_placement",
        "change_microbatch_group_size",
        "split_wgrad",
        "change_checkpoint_window",
        "enable_ready_set_dispatch",
    }
    candidate = synth.build_strategy_schedule_table(
        spec.policy,
        synth.StrategyConstraints(
            num_microbatches=args.num_microbatches,
            num_model_chunks=spec.vpp_size,
            microbatch_group_size=spec.group_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        ),
    )
    layout = spec.layout
    if spec.segment_decoder_counts:
        layout = twincut.build_pipeline_layout(spec.segment_decoder_counts)
    step_time = _estimate_step_time(
        events,
        spec.group_size,
        candidate.name,
        spec.vpp_size,
        args.num_model_chunks,
        layout,
        spec.memory_actions,
        spec.cross_node_boundaries,
        spec.topology_objective,
    )
    bcp = _bcp_stats(
        events,
        group_size=spec.group_size,
        policy=candidate.name,
        vpp_size=spec.vpp_size,
        baseline_vpp_size=args.num_model_chunks,
        layout=layout,
        runtime=args.runtime,
        budget=budget,
        memory_actions=spec.memory_actions,
        cross_node_boundaries=spec.cross_node_boundaries,
    )
    tasks = _build_task_dag(synth, events, args, spec.vpp_size)
    task_dag_critical_path = _task_dag_critical_path_ms(tasks)
    estimated_peak_memory_mb = _estimate_memory(events)
    topology_breakdown: Dict[str, float] = {}
    if spec.segment_decoder_counts and spec.node_assignment:
        layer_time_ms, activation_mb_per_layer = _estimate_layer_costs(
            events,
            args.num_layers or sum(spec.segment_decoder_counts),
            args.pipeline_parallel_size,
        )
        topology_breakdown = twincut.estimate_cost(
            spec.segment_decoder_counts,
            spec.node_assignment,
            layer_time_ms=layer_time_ms,
            activation_mb_per_layer=activation_mb_per_layer,
            exposed_wait_ms=bcp.exposed_p2p_wait_ms,
            p2p_credit_pressure=bcp.p2p_credit_pressure,
        )
    runtime_policy = {
        "runtime": args.runtime,
        "allow_out_of_order_p2p": False,
        "bubble_fill": args.runtime == "bcp-ready",
    }
    plan = synth.strategy_candidate_to_plan(
        candidate,
        microbatch_group_size=spec.group_size,
        runtime_policy=runtime_policy,
        rewrites=tuple(
            synth.StrategyRewrite(
                action=item.get("action", "change_schedule_order")
                if item.get("action") in legal_actions
                else "change_schedule_order",
                target=item.get("target") or {},
                reason=item.get("reason", ""),
            )
            for item in rewrites
            if not item.get("requires_runtime_support", False)
        ),
        metadata={
            "estimated_step_time_ms": step_time,
            "bcp_score": bcp.score,
            "bcp_critical_path_ms": bcp.critical_path_ms,
            "bcp_exposed_p2p_wait_ms": bcp.exposed_p2p_wait_ms,
            "bcp_activation_peak_mb": bcp.activation_peak_mb,
            "bcp_fb_delay_steps": bcp.fb_delay_steps,
            "bcp_chunk_skew": bcp.chunk_skew,
            "bcp_p2p_credit_pressure": bcp.p2p_credit_pressure,
            "bcp_static_task_dag_critical_path_ms": task_dag_critical_path,
            "estimated_throughput_score": 1.0 / max(step_time, 1e-6),
            "estimated_peak_memory_mb": estimated_peak_memory_mb,
            "source": "search_pipeline_strategy.py",
            "objective": args.objective,
            "layout_kind": "custom" if layout else "default",
            "baseline_num_virtual_stages_per_pipeline_rank": args.num_model_chunks,
            "twincut_name": spec.twincut_name or None,
            "twincut_topology_objective": spec.topology_objective,
            "twincut_cost_breakdown": topology_breakdown,
            "num_decoder_layers": args.num_layers,
        },
    )
    plan = plan.__class__(
        name=(
            f"{plan.name}-vp{spec.vpp_size}-g{spec.group_size}"
            + ("-layout" if layout else "")
            + (f"-{spec.twincut_name}" if spec.twincut_name else "")
        ),
        schedule_table=plan.schedule_table,
        pipeline_layout=layout,
        num_virtual_stages_per_pipeline_rank=spec.vpp_size,
        placement=spec.placement,
        segment_boundaries=spec.segment_boundaries,
        cross_node_boundaries=spec.cross_node_boundaries,
        node_assignment=spec.node_assignment,
        vpp_packing=spec.vpp_packing,
        memory_actions=spec.memory_actions,
        microbatch_group_size=plan.microbatch_group_size,
        checkpoint_policy=plan.checkpoint_policy,
        wgrad_policy=plan.wgrad_policy,
        runtime_policy=plan.runtime_policy,
        rewrites=plan.rewrites,
        tasks=tasks,
        metadata=plan.metadata,
    )
    synth.StrategyVerifier(
        synth.StrategyConstraints(
            num_microbatches=args.num_microbatches,
            num_model_chunks=spec.vpp_size,
            microbatch_group_size=spec.group_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        ),
        memory_budget_mb=args.memory_budget_mb,
    ).verify(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--proposal-json", default=None)
    parser.add_argument("--num-microbatches", type=int, required=True)
    parser.add_argument("--num-model-chunks", type=int, required=True)
    parser.add_argument("--microbatch-group-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--runtime", choices=["fixed", "ready-set", "bcp-ready"], default="fixed")
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    parser.add_argument("--candidate-budget", type=int, default=16)
    parser.add_argument(
        "--objective",
        choices=["legacy", "bcp"],
        default="bcp",
        help="Rank candidates by legacy step-time estimate or bounded critical-path score.",
    )
    parser.add_argument(
        "--bcp-activation-budget-mb",
        type=float,
        default=None,
        help="Soft activation peak budget for BCP ranking.",
    )
    parser.add_argument(
        "--bcp-p2p-credit-budget",
        type=int,
        default=None,
        help="Soft maximum outstanding P2P request budget for BCP ranking.",
    )
    parser.add_argument(
        "--bcp-fb-delay-budget",
        type=int,
        default=None,
        help="Soft forward/backward event-distance budget for BCP ranking.",
    )
    args = parser.parse_args()
    bcp_budget = BcpBudget(
        activation_peak_mb=args.bcp_activation_budget_mb,
        p2p_credit=args.bcp_p2p_credit_budget,
        fb_delay_steps=args.bcp_fb_delay_budget,
    )

    synth = _load_module(
        "strategy_synthesizer",
        "megatron/core/pipeline_parallel/strategy_synthesizer.py",
    )
    twincut = _load_module(
        "twincut_partition",
        "megatron/core/pipeline_parallel/twincut_partition.py",
    )
    agent = _load_module("pipeline_strategy_agent", "tools/pipeline_strategy_agent.py")

    events = agent.load_events(args.trace)
    if args.proposal_json:
        proposal_doc = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
        bottlenecks = [
            agent.Bottleneck(
                kind=item["kind"],
                score=float(item["score"]),
                evidence=item["evidence"],
                affected_tasks=item.get("affected_tasks"),
            )
            for item in proposal_doc.get("bottlenecks", [])
        ]
        proposals = [
            agent.RewriteProposal(
                action=item["action"],
                reason=item.get("reason", ""),
                expected_effect=item.get("expected_effect", ""),
                policy=item.get("policy", "default"),
                target=item.get("target") or {},
                requires_runtime_support=bool(item.get("requires_runtime_support", False)),
            )
            for item in proposal_doc.get("proposals", [])
        ]
    else:
        bottlenecks = agent.attribute_bottlenecks(events)
        proposals = agent.propose_rewrites(bottlenecks)

    diagnostics = _trace_diagnostics(events, args.num_model_chunks)
    candidate_specs = _build_candidate_specs(args, proposals, diagnostics, events, twincut)

    accepted = []
    rejected = []
    proposal_payload = [asdict(item) for item in proposals]
    for spec in candidate_specs:
        try:
            plan = _build_candidate(
                synth,
                twincut,
                spec,
                args,
                proposal_payload,
                events,
                bcp_budget,
            )
            metadata = dict(plan.metadata)
            metadata.update(
                {
                    "rewrite": spec.rewrite,
                    "rewrite_rationale": spec.rationale,
                    "candidate_quick_score": spec.quick_score,
                }
            )
            plan = plan.__class__(
                name=plan.name,
                schedule_table=plan.schedule_table,
                pipeline_layout=plan.pipeline_layout,
                num_virtual_stages_per_pipeline_rank=plan.num_virtual_stages_per_pipeline_rank,
                placement=plan.placement,
                segment_boundaries=getattr(plan, "segment_boundaries", ()),
                cross_node_boundaries=getattr(plan, "cross_node_boundaries", ()),
                node_assignment=getattr(plan, "node_assignment", ()),
                vpp_packing=getattr(plan, "vpp_packing", ()),
                memory_actions=getattr(plan, "memory_actions", ()),
                microbatch_group_size=plan.microbatch_group_size,
                checkpoint_policy=plan.checkpoint_policy,
                wgrad_policy=plan.wgrad_policy,
                runtime_policy=plan.runtime_policy,
                rewrites=plan.rewrites,
                tasks=plan.tasks,
                metadata=metadata,
            )
            accepted.append(plan)
        except Exception as exc:  # noqa: BLE001 - report verifier/search failures.
            rejected.append(
                {
                    "policy": spec.policy,
                    "group_size": spec.group_size,
                    "vpp_size": spec.vpp_size,
                    "layout": spec.layout,
                    "placement": list(spec.placement),
                    "rewrite": spec.rewrite,
                    "rationale": spec.rationale,
                    "quick_score": spec.quick_score,
                    "reason": str(exc),
                }
            )

    if not accepted:
        raise RuntimeError(f"no valid strategy candidates; rejected={rejected}")
    if args.objective == "legacy":
        best = min(accepted, key=lambda plan: plan.metadata["estimated_step_time_ms"])
    else:
        best = min(accepted, key=lambda plan: plan.metadata["bcp_score"])
    synth.emit_strategy_plan(best, args.output)

    report = {
        "bottlenecks": [asdict(item) for item in bottlenecks],
        "proposals": proposal_payload,
        "objective": args.objective,
        "bcp_budget": asdict(bcp_budget),
        "bcp_diagnostics": diagnostics,
        "rewrite_algebra": BCP_REWRITE_ALGEBRA,
        "candidate_specs": [asdict(item) for item in candidate_specs],
        "accepted": [synth.strategy_plan_to_dict(item) for item in accepted],
        "rejected": rejected,
        "best": synth.strategy_plan_to_dict(best),
    }
    report_path = Path(args.report) if args.report else Path(args.output).with_name("candidate_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
