"""Topology-aware structured partition helpers for dual-node PP/VPP search.

This module keeps the TwinCut search space deterministic and lightweight.
It does not execute schedules directly; instead it synthesizes structured
partition candidates that can be compiled into Megatron pipeline layouts and
verified by the strategy synthesizer.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


MemoryAction = Tuple[int, str]


@dataclass(frozen=True)
class TwinCutSpec:
    """Structured partition candidate for PP/VPP strategy search."""

    name: str
    vpp_size: int
    segment_decoder_counts: Tuple[int, ...]
    segment_boundaries: Tuple[int, ...]
    node_assignment: Tuple[int, ...]
    cross_node_boundaries: Tuple[int, ...]
    vpp_packing: Tuple[Tuple[int, ...], ...]
    memory_actions: Tuple[MemoryAction, ...]
    objective: float
    cost_breakdown: Dict[str, float]


@dataclass(frozen=True)
class TwinCutSearchConfig:
    """Bounded search controls for structured partition synthesis."""

    max_expansions: int = 128
    max_candidates: int = 8
    max_depth: int = 6
    branch_width: int = 12
    seam_bias_weight: float = 1.5


@dataclass(frozen=True)
class TwinCutSearchState:
    """One partial structured partition candidate in the search tree."""

    counts: Tuple[int, ...]
    depth: int
    lower_bound: float
    score: float
    move_history: Tuple[str, ...] = ()


def _balanced_stage_counts(num_layers: int, total_stages: int) -> List[int]:
    if total_stages <= 0:
        raise ValueError("total_stages must be positive")
    if num_layers < total_stages:
        raise ValueError(
            f"num_layers={num_layers} is too small for total_stages={total_stages}; "
            "each stage must own at least one decoder layer"
        )
    base = num_layers // total_stages
    rem = num_layers % total_stages
    return [base + (1 if stage < rem else 0) for stage in range(total_stages)]


def _shift_one(counts: List[int], src: int, dst: int) -> bool:
    if src < 0 or src >= len(counts) or dst < 0 or dst >= len(counts):
        return False
    if counts[src] <= 1:
        return False
    counts[src] -= 1
    counts[dst] += 1
    return True


def _seam_stage_indices(pipeline_parallel_size: int, vpp_size: int) -> List[Tuple[int, int]]:
    if pipeline_parallel_size < 2:
        return []
    left_rank = pipeline_parallel_size // 2 - 1
    right_rank = pipeline_parallel_size // 2
    return [
        (
            vp_rank * pipeline_parallel_size + left_rank,
            vp_rank * pipeline_parallel_size + right_rank,
        )
        for vp_rank in range(vpp_size)
    ]


def _apply_seam_relief(
    counts: List[int],
    pipeline_parallel_size: int,
    vpp_size: int,
    seam_relief: int,
) -> List[int]:
    counts = list(counts)
    if seam_relief <= 0:
        return counts
    for _ in range(seam_relief):
        for left_idx, right_idx in _seam_stage_indices(pipeline_parallel_size, vpp_size):
            _shift_one(counts, left_idx, left_idx - 1)
            _shift_one(counts, right_idx, right_idx + 1)
    return counts


def _apply_tail_rebalance(counts: List[int], tail_relief: int) -> List[int]:
    counts = list(counts)
    if tail_relief <= 0 or len(counts) < 4:
        return counts
    for _ in range(tail_relief):
        if not _shift_one(counts, len(counts) - 2, len(counts) - 1):
            break
    return counts


def build_node_assignment(
    pipeline_parallel_size: int,
    vpp_size: int,
    num_nodes: int = 2,
) -> Tuple[int, ...]:
    if pipeline_parallel_size <= 0:
        raise ValueError("pipeline_parallel_size must be positive")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    ranks_per_node = max(1, pipeline_parallel_size // num_nodes)
    assignment: List[int] = []
    for vp_rank in range(vpp_size):
        del vp_rank
        for pp_rank in range(pipeline_parallel_size):
            node = min(pp_rank // ranks_per_node, num_nodes - 1)
            assignment.append(node)
    return tuple(assignment)


def infer_cross_node_boundaries(node_assignment: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(
        stage_idx
        for stage_idx in range(1, len(node_assignment))
        if node_assignment[stage_idx] != node_assignment[stage_idx - 1]
    )


def segment_boundaries_from_counts(segment_decoder_counts: Tuple[int, ...]) -> Tuple[int, ...]:
    boundaries = [0]
    for count in segment_decoder_counts:
        if count <= 0:
            raise ValueError(f"segment count must be positive, got {count}")
        boundaries.append(boundaries[-1] + int(count))
    return tuple(boundaries)


def build_vpp_packing(
    segment_decoder_counts: Tuple[int, ...],
    pipeline_parallel_size: int,
    vpp_size: int,
) -> Tuple[Tuple[int, ...], ...]:
    total_stages = pipeline_parallel_size * max(vpp_size, 1)
    if len(segment_decoder_counts) != total_stages:
        raise ValueError(
            f"segment count length {len(segment_decoder_counts)} does not match "
            f"pipeline_parallel_size*vpp_size={total_stages}"
        )
    packing: List[List[int]] = []
    for pp_rank in range(pipeline_parallel_size):
        packing.append(
            [
                int(segment_decoder_counts[vp_rank * pipeline_parallel_size + pp_rank])
                for vp_rank in range(vpp_size)
            ]
        )
    return tuple(tuple(row) for row in packing)


def build_pipeline_layout(segment_decoder_counts: Tuple[int, ...]) -> str:
    total_stages = len(segment_decoder_counts)
    if total_stages <= 0:
        raise ValueError("segment_decoder_counts must not be empty")
    segments: List[str] = []
    for stage_idx, decoder_layers in enumerate(segment_decoder_counts):
        tokens = ["t"] * int(decoder_layers)
        if stage_idx == 0:
            tokens = ["E", *tokens]
        if stage_idx == total_stages - 1:
            tokens = [*tokens, "L"]
        segments.append("".join(tokens))
    return "|".join(segments)


def build_memory_actions(
    segment_decoder_counts: Tuple[int, ...],
    cross_node_boundaries: Tuple[int, ...],
) -> Tuple[MemoryAction, ...]:
    if not segment_decoder_counts:
        return ()
    counts_with_idx = sorted(
        enumerate(segment_decoder_counts),
        key=lambda item: item[1],
        reverse=True,
    )
    actions: List[MemoryAction] = []
    heavy_stage_budget = min(2, len(counts_with_idx))
    for stage_idx, _count in counts_with_idx[:heavy_stage_budget]:
        actions.append((int(stage_idx), "recompute"))
    for boundary in cross_node_boundaries:
        left_stage = boundary - 1
        right_stage = boundary
        if left_stage >= 0:
            actions.append((left_stage, "retain"))
        if right_stage < len(segment_decoder_counts):
            actions.append((right_stage, "retain"))
    deduped: List[MemoryAction] = []
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        deduped.append(action)
    return tuple(deduped)


def _normalize_memory_actions(actions: Tuple[MemoryAction, ...]) -> Tuple[MemoryAction, ...]:
    priority = {"retain": 0, "checkpoint": 1, "recompute": 2, "offload": 3, "none": 4}
    chosen: Dict[int, str] = {}
    for stage_idx, action in actions:
        current = chosen.get(stage_idx)
        if current is None or priority[action] < priority[current]:
            chosen[stage_idx] = action
    return tuple((stage_idx, chosen[stage_idx]) for stage_idx in sorted(chosen))


def optimize_memory_actions(
    segment_decoder_counts: Tuple[int, ...],
    cross_node_boundaries: Tuple[int, ...],
    *,
    activation_mb_per_layer: float,
    memory_budget_mb: float | None = None,
) -> Tuple[MemoryAction, ...]:
    """Choose segment-local memory actions with a tiny deterministic controller.

    This is intentionally simple but no longer a pure annotation pass. The
    searcher selects actions by comparing per-stage activation pressure against
    the current budget and always protecting cross-node seam stages first.
    """

    if not segment_decoder_counts:
        return ()
    actions: List[MemoryAction] = []
    stage_memory = [count * activation_mb_per_layer for count in segment_decoder_counts]
    average_memory = sum(stage_memory) / max(len(stage_memory), 1)
    seam_stages = set()
    for boundary in cross_node_boundaries:
        seam_stages.add(boundary - 1)
        seam_stages.add(boundary)
    for stage_idx in sorted(stage for stage in seam_stages if 0 <= stage < len(stage_memory)):
        if stage_memory[stage_idx] >= 0.75 * max(average_memory, 1e-6):
            actions.append((stage_idx, "retain"))
    for stage_idx, memory_mb in enumerate(stage_memory):
        if memory_budget_mb is not None and memory_mb > memory_budget_mb:
            actions.append((stage_idx, "offload"))
        elif memory_mb > 1.15 * max(average_memory, 1e-6):
            actions.append((stage_idx, "recompute"))
        elif memory_mb < 0.7 * max(average_memory, 1e-6) and stage_idx not in seam_stages:
            actions.append((stage_idx, "none"))
    return _normalize_memory_actions(tuple(actions))


def estimate_cost(
    segment_decoder_counts: Tuple[int, ...],
    node_assignment: Tuple[int, ...],
    *,
    layer_time_ms: float,
    activation_mb_per_layer: float,
    exposed_wait_ms: float,
    p2p_credit_pressure: float,
) -> Dict[str, float]:
    if not segment_decoder_counts:
        return {
            "step_time_ms": 0.0,
            "imbalance_penalty": 0.0,
            "cross_node_penalty": 0.0,
            "memory_penalty": 0.0,
            "objective": 0.0,
        }

    average_layers = sum(segment_decoder_counts) / len(segment_decoder_counts)
    max_layers = max(segment_decoder_counts)
    imbalance_ratio = max_layers / max(average_layers, 1e-6) - 1.0
    step_time_ms = max_layers * layer_time_ms
    cross_edges = len(infer_cross_node_boundaries(node_assignment))
    cross_node_penalty = cross_edges * (0.35 * exposed_wait_ms + 0.5 * layer_time_ms)
    memory_peak_mb = max_layers * activation_mb_per_layer
    memory_penalty = 0.03 * memory_peak_mb
    credit_penalty = 2.0 * max(0.0, p2p_credit_pressure)
    imbalance_penalty = 10.0 * max(0.0, imbalance_ratio)
    objective = step_time_ms + cross_node_penalty + memory_penalty + credit_penalty + imbalance_penalty
    return {
        "step_time_ms": step_time_ms,
        "imbalance_penalty": imbalance_penalty,
        "cross_node_penalty": cross_node_penalty,
        "memory_penalty": memory_penalty,
        "credit_penalty": credit_penalty,
        "objective": objective,
    }


def lower_bound_cost(
    segment_decoder_counts: Tuple[int, ...],
    node_assignment: Tuple[int, ...],
    *,
    layer_time_ms: float,
    activation_mb_per_layer: float,
    exposed_wait_ms: float,
    p2p_credit_pressure: float,
) -> Dict[str, float]:
    """A cheap optimistic bound used for branch-and-bound pruning."""

    if not segment_decoder_counts:
        return {
            "step_time_ms": 0.0,
            "cross_node_penalty": 0.0,
            "memory_penalty": 0.0,
            "credit_penalty": 0.0,
            "objective": 0.0,
        }
    total_layers = sum(segment_decoder_counts)
    stage_count = len(segment_decoder_counts)
    perfect_balanced_peak = math.ceil(total_layers / max(stage_count, 1))
    step_time_ms = perfect_balanced_peak * layer_time_ms
    cross_edges = len(infer_cross_node_boundaries(node_assignment))
    cross_node_penalty = cross_edges * max(0.1 * exposed_wait_ms, 0.25 * layer_time_ms)
    memory_penalty = 0.03 * perfect_balanced_peak * activation_mb_per_layer
    credit_penalty = 1.5 * max(0.0, p2p_credit_pressure)
    objective = step_time_ms + cross_node_penalty + memory_penalty + credit_penalty
    return {
        "step_time_ms": step_time_ms,
        "cross_node_penalty": cross_node_penalty,
        "memory_penalty": memory_penalty,
        "credit_penalty": credit_penalty,
        "objective": objective,
    }


def _state_signature(counts: Tuple[int, ...]) -> Tuple[int, ...]:
    return counts


def _adjacent_moves(
    counts: Tuple[int, ...],
    pipeline_parallel_size: int,
    vpp_size: int,
    seam_bias_weight: float,
) -> List[Tuple[float, Tuple[int, ...], str]]:
    """Generate local legal layer-boundary moves around imbalance hotspots."""

    average = sum(counts) / max(len(counts), 1)
    seam_pairs = set(_seam_stage_indices(pipeline_parallel_size, vpp_size))
    candidates: List[Tuple[float, Tuple[int, ...], str]] = []
    for idx in range(len(counts) - 1):
        left = counts[idx]
        right = counts[idx + 1]
        if left > 1:
            moved = list(counts)
            moved[idx] -= 1
            moved[idx + 1] += 1
            seam_bonus = seam_bias_weight if (idx, idx + 1) in seam_pairs else 1.0
            skew_before = abs(left - average) + abs(right - average)
            skew_after = abs(moved[idx] - average) + abs(moved[idx + 1] - average)
            gain = seam_bonus * (skew_before - skew_after)
            candidates.append((gain, tuple(moved), f"{idx}->{idx + 1}"))
        if right > 1:
            moved = list(counts)
            moved[idx] += 1
            moved[idx + 1] -= 1
            seam_bonus = seam_bias_weight if (idx, idx + 1) in seam_pairs else 1.0
            skew_before = abs(left - average) + abs(right - average)
            skew_after = abs(moved[idx] - average) + abs(moved[idx + 1] - average)
            gain = seam_bonus * (skew_before - skew_after)
            candidates.append((gain, tuple(moved), f"{idx + 1}->{idx}"))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def solve_twincut_specs(
    *,
    num_layers: int,
    pipeline_parallel_size: int,
    vpp_size: int,
    layer_time_ms: float,
    activation_mb_per_layer: float,
    exposed_wait_ms: float,
    p2p_credit_pressure: float,
    memory_budget_mb: float | None = None,
    search_config: TwinCutSearchConfig | None = None,
) -> List[TwinCutSpec]:
    """Search structured partition candidates with bounded branch-and-bound."""

    config = search_config or TwinCutSearchConfig()
    total_stages = pipeline_parallel_size * max(vpp_size, 1)
    balanced = tuple(_balanced_stage_counts(num_layers, total_stages))
    node_assignment = build_node_assignment(pipeline_parallel_size, vpp_size, num_nodes=2)

    root_lower_bound = lower_bound_cost(
        balanced,
        node_assignment,
        layer_time_ms=layer_time_ms,
        activation_mb_per_layer=activation_mb_per_layer,
        exposed_wait_ms=exposed_wait_ms,
        p2p_credit_pressure=p2p_credit_pressure,
    )["objective"]
    root_score = estimate_cost(
        balanced,
        node_assignment,
        layer_time_ms=layer_time_ms,
        activation_mb_per_layer=activation_mb_per_layer,
        exposed_wait_ms=exposed_wait_ms,
        p2p_credit_pressure=p2p_credit_pressure,
    )["objective"]
    root = TwinCutSearchState(
        counts=balanced,
        depth=0,
        lower_bound=root_lower_bound,
        score=root_score,
        move_history=(),
    )
    frontier: List[Tuple[float, float, int, int, TwinCutSearchState]] = []
    counter = 0
    heapq.heappush(frontier, (root.lower_bound, root.score, 0, counter, root))
    visited = {_state_signature(root.counts)}
    expansions = 0
    accepted_states: List[TwinCutSearchState] = []
    incumbent = root.score

    while frontier and expansions < config.max_expansions:
        _bound, _score, _neg_depth, _counter, state = heapq.heappop(frontier)
        accepted_states.append(state)
        incumbent = min(incumbent, state.score)
        if state.depth >= config.max_depth:
            continue

        expansions += 1
        moves = _adjacent_moves(
            state.counts,
            pipeline_parallel_size,
            vpp_size,
            config.seam_bias_weight,
        )[: max(config.branch_width, 1)]
        for _gain, moved_counts, move_label in moves:
            signature = _state_signature(moved_counts)
            if signature in visited:
                continue
            visited.add(signature)
            lower_bound = lower_bound_cost(
                moved_counts,
                node_assignment,
                layer_time_ms=layer_time_ms,
                activation_mb_per_layer=activation_mb_per_layer,
                exposed_wait_ms=exposed_wait_ms,
                p2p_credit_pressure=p2p_credit_pressure,
            )["objective"]
            if lower_bound > incumbent * 1.08:
                continue
            score = estimate_cost(
                moved_counts,
                node_assignment,
                layer_time_ms=layer_time_ms,
                activation_mb_per_layer=activation_mb_per_layer,
                exposed_wait_ms=exposed_wait_ms,
                p2p_credit_pressure=p2p_credit_pressure,
            )["objective"]
            child = TwinCutSearchState(
                counts=moved_counts,
                depth=state.depth + 1,
                lower_bound=lower_bound,
                score=score,
                move_history=state.move_history + (move_label,),
            )
            counter += 1
            heapq.heappush(
                frontier,
                (child.lower_bound, child.score, -child.depth, counter, child),
            )

    scored_states = sorted(
        accepted_states,
        key=lambda item: (item.score, item.lower_bound, item.depth),
    )[: max(config.max_candidates, 1)]

    specs: List[TwinCutSpec] = []
    for rank, state in enumerate(scored_states):
        counts = state.counts
        cross_node_boundaries = infer_cross_node_boundaries(node_assignment)
        cost = estimate_cost(
            counts,
            node_assignment,
            layer_time_ms=layer_time_ms,
            activation_mb_per_layer=activation_mb_per_layer,
            exposed_wait_ms=exposed_wait_ms,
            p2p_credit_pressure=p2p_credit_pressure,
        )
        cost.update(
            {
                "lower_bound": state.lower_bound,
                "search_depth": float(state.depth),
                "num_moves": float(len(state.move_history)),
            }
        )
        memory_actions = optimize_memory_actions(
            counts,
            cross_node_boundaries,
            activation_mb_per_layer=activation_mb_per_layer,
            memory_budget_mb=memory_budget_mb,
        )
        specs.append(
            TwinCutSpec(
                name="solver-root" if rank == 0 and not state.move_history else f"solver-{rank}",
                vpp_size=vpp_size,
                segment_decoder_counts=counts,
                segment_boundaries=segment_boundaries_from_counts(counts),
                node_assignment=node_assignment,
                cross_node_boundaries=cross_node_boundaries,
                vpp_packing=build_vpp_packing(counts, pipeline_parallel_size, vpp_size),
                memory_actions=memory_actions,
                objective=cost["objective"],
                cost_breakdown=cost,
            )
        )
    return specs


def propose_twincut_specs(
    *,
    num_layers: int,
    pipeline_parallel_size: int,
    vpp_size: int,
    layer_time_ms: float,
    activation_mb_per_layer: float,
    exposed_wait_ms: float,
    p2p_credit_pressure: float,
    memory_budget_mb: float | None = None,
) -> List[TwinCutSpec]:
    specs = solve_twincut_specs(
        num_layers=num_layers,
        pipeline_parallel_size=pipeline_parallel_size,
        vpp_size=vpp_size,
        layer_time_ms=layer_time_ms,
        activation_mb_per_layer=activation_mb_per_layer,
        exposed_wait_ms=exposed_wait_ms,
        p2p_credit_pressure=p2p_credit_pressure,
        memory_budget_mb=memory_budget_mb,
    )

    # Keep a tiny heuristic backstop so the search space never collapses to one
    # branch when trace estimates are noisy.
    total_stages = pipeline_parallel_size * max(vpp_size, 1)
    balanced = tuple(_balanced_stage_counts(num_layers, total_stages))
    node_assignment = build_node_assignment(pipeline_parallel_size, vpp_size, num_nodes=2)
    seam_relief = tuple(_apply_seam_relief(list(balanced), pipeline_parallel_size, vpp_size, 1))
    fallback_candidates = [("balanced", balanced)]
    if seam_relief != balanced:
        fallback_candidates.append(("seam-relief-heuristic", seam_relief))
    for name, counts in fallback_candidates:
        cross_node_boundaries = infer_cross_node_boundaries(node_assignment)
        cost = estimate_cost(
            counts,
            node_assignment,
            layer_time_ms=layer_time_ms,
            activation_mb_per_layer=activation_mb_per_layer,
            exposed_wait_ms=exposed_wait_ms,
            p2p_credit_pressure=p2p_credit_pressure,
        )
        specs.append(
            TwinCutSpec(
                name=name,
                vpp_size=vpp_size,
                segment_decoder_counts=counts,
                segment_boundaries=segment_boundaries_from_counts(counts),
                node_assignment=node_assignment,
                cross_node_boundaries=cross_node_boundaries,
                vpp_packing=build_vpp_packing(counts, pipeline_parallel_size, vpp_size),
                memory_actions=optimize_memory_actions(
                    counts,
                    cross_node_boundaries,
                    activation_mb_per_layer=activation_mb_per_layer,
                    memory_budget_mb=memory_budget_mb,
                ),
                objective=cost["objective"],
                cost_breakdown=cost,
            )
        )

    deduped: List[TwinCutSpec] = []
    seen = set()
    for spec in sorted(specs, key=lambda item: item.objective):
        key = (spec.segment_decoder_counts, spec.vpp_packing, spec.memory_actions)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped
