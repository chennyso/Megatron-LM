# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""PP/VPP strategy synthesis primitives.

This module is intentionally deterministic. LLM agents and offline searchers can
propose strategy rewrites through this API, but correctness is guarded here by a
small IR plus verifier before the pipeline schedule consumes a candidate.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


ScheduleEntry = Tuple[int, int]
TaskKind = Literal[
    "F",
    "B_DGRAD",
    "B_WGRAD",
    "SEND_F",
    "RECV_F",
    "SEND_B",
    "RECV_B",
    "PARAM_SYNC",
    "GRAD_SYNC",
    "RECOMPUTE",
]
RewriteAction = Literal[
    "change_schedule_order",
    "move_layer_boundary",
    "swap_stage_placement",
    "change_microbatch_group_size",
    "split_wgrad",
    "change_checkpoint_window",
    "enable_ready_set_dispatch",
]


@dataclass(frozen=True)
class PipelineTask:
    """A schedule-table task identified by microbatch and virtual chunk."""

    microbatch_id: int
    model_chunk_id: int


@dataclass(frozen=True)
class StrategyTask:
    """Fine-grained task IR used by offline search and verification."""

    task_id: str
    kind: TaskKind
    pp_rank: int
    vp_chunk: int
    microbatch: int
    est_compute_ms: float = 0.0
    est_comm_ms: float = 0.0
    est_memory_mb: float = 0.0
    deps: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyRewrite:
    """Structured strategy rewrite emitted by a heuristic or LLM proposer."""

    action: RewriteAction
    target: Dict[str, Any]
    reason: str


@dataclass(frozen=True)
class StrategyPlan:
    """Executable PP/VPP strategy plan."""

    name: str
    schedule_table: Tuple[ScheduleEntry, ...]
    pipeline_layout: Optional[str] = None
    num_virtual_stages_per_pipeline_rank: Optional[int] = None
    placement: Tuple[int, ...] = ()
    segment_boundaries: Tuple[int, ...] = ()
    cross_node_boundaries: Tuple[int, ...] = ()
    node_assignment: Tuple[int, ...] = ()
    vpp_packing: Tuple[Tuple[int, ...], ...] = ()
    memory_actions: Tuple[Tuple[int, str], ...] = ()
    microbatch_group_size: int = 0
    checkpoint_policy: Dict[str, Any] = field(default_factory=dict)
    wgrad_policy: Dict[str, Any] = field(default_factory=dict)
    runtime_policy: Dict[str, Any] = field(default_factory=dict)
    rewrites: Tuple[StrategyRewrite, ...] = ()
    tasks: Tuple[StrategyTask, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyTraceEvent:
    """Runtime signal used by external agents/searchers to rewrite strategies."""

    name: str
    pp_rank: int
    microbatch_id: Optional[int]
    model_chunk_id: Optional[int]
    elapsed_ms: float
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    ready_ts: Optional[float] = None
    wait_ms: float = 0.0
    memory_mb: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyTrace:
    """Low-overhead trace buffer for one rank."""

    enabled: bool = False
    pp_rank: int = 0
    events: List[StrategyTraceEvent] = field(default_factory=list)
    flush_path: Optional[str] = None

    def record(
        self,
        name: str,
        elapsed_ms: float,
        microbatch_id: Optional[int] = None,
        model_chunk_id: Optional[int] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        ready_ts: Optional[float] = None,
        wait_ms: float = 0.0,
        memory_mb: Optional[float] = None,
        **metadata,
    ) -> None:
        if not self.enabled:
            return
        self.events.append(
            StrategyTraceEvent(
                name=name,
                pp_rank=self.pp_rank,
                microbatch_id=microbatch_id,
                model_chunk_id=model_chunk_id,
                elapsed_ms=elapsed_ms,
                start_ts=start_ts,
                end_ts=end_ts,
                ready_ts=ready_ts,
                wait_ms=wait_ms,
                memory_mb=memory_mb,
                metadata=metadata,
            )
        )

    def flush(self) -> None:
        if not self.enabled or not self.flush_path:
            return
        path = Path(self.flush_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(event) for event in self.events]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class StrategyConstraints:
    """Legality constraints for a VPP schedule table."""

    num_microbatches: int
    num_model_chunks: int
    microbatch_group_size: int
    pipeline_parallel_size: int
    allow_chunk_reorder: bool = True


@dataclass(frozen=True)
class StrategyCandidate:
    """A complete local schedule-table candidate for one VPP execution."""

    name: str
    schedule_table: Tuple[ScheduleEntry, ...]
    rewrites: Tuple[str, ...] = ()


def _rewrite_from_dict(payload: Dict[str, Any]) -> StrategyRewrite:
    return StrategyRewrite(
        action=payload["action"],
        target=dict(payload.get("target", {})),
        reason=str(payload.get("reason", "")),
    )


def _task_from_dict(payload: Dict[str, Any]) -> StrategyTask:
    return StrategyTask(
        task_id=str(payload["task_id"]),
        kind=payload["kind"],
        pp_rank=int(payload["pp_rank"]),
        vp_chunk=int(payload["vp_chunk"]),
        microbatch=int(payload["microbatch"]),
        est_compute_ms=float(payload.get("est_compute_ms", 0.0)),
        est_comm_ms=float(payload.get("est_comm_ms", 0.0)),
        est_memory_mb=float(payload.get("est_memory_mb", 0.0)),
        deps=tuple(str(dep) for dep in payload.get("deps", ())),
    )


def strategy_plan_from_dict(payload: Dict[str, Any]) -> StrategyPlan:
    """Build a StrategyPlan from a JSON-compatible dictionary."""

    return StrategyPlan(
        name=str(payload.get("name", "unnamed")),
        schedule_table=tuple((int(entry[0]), int(entry[1])) for entry in payload["schedule_table"]),
        pipeline_layout=payload.get("pipeline_layout"),
        num_virtual_stages_per_pipeline_rank=(
            int(payload["num_virtual_stages_per_pipeline_rank"])
            if payload.get("num_virtual_stages_per_pipeline_rank") is not None
            else None
        ),
        placement=tuple(int(item) for item in payload.get("placement", ())),
        segment_boundaries=tuple(int(item) for item in payload.get("segment_boundaries", ())),
        cross_node_boundaries=tuple(int(item) for item in payload.get("cross_node_boundaries", ())),
        node_assignment=tuple(int(item) for item in payload.get("node_assignment", ())),
        vpp_packing=tuple(
            tuple(int(count) for count in row) for row in payload.get("vpp_packing", ())
        ),
        memory_actions=tuple(
            (int(item[0]), str(item[1])) for item in payload.get("memory_actions", ())
        ),
        microbatch_group_size=int(payload.get("microbatch_group_size", 0)),
        checkpoint_policy=dict(payload.get("checkpoint_policy", {})),
        wgrad_policy=dict(payload.get("wgrad_policy", {})),
        runtime_policy=dict(payload.get("runtime_policy", {})),
        rewrites=tuple(_rewrite_from_dict(item) for item in payload.get("rewrites", ())),
        tasks=tuple(_task_from_dict(item) for item in payload.get("tasks", ())),
        metadata=dict(payload.get("metadata", {})),
    )


def load_strategy_plan(path: str | Path) -> StrategyPlan:
    """Load a StrategyPlan from JSON without verifying it."""

    return strategy_plan_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def strategy_plan_to_dict(plan: StrategyPlan) -> Dict[str, Any]:
    """Serialize a StrategyPlan to a JSON-compatible dictionary."""

    return asdict(plan)


def emit_strategy_plan(plan: StrategyPlan, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strategy_plan_to_dict(plan), indent=2), encoding="utf-8")


def strategy_candidate_to_plan(
    candidate: StrategyCandidate,
    microbatch_group_size: int,
    runtime_policy: Optional[Dict[str, Any]] = None,
    rewrites: Tuple[StrategyRewrite, ...] = (),
    metadata: Optional[Dict[str, Any]] = None,
) -> StrategyPlan:
    return StrategyPlan(
        name=candidate.name,
        schedule_table=candidate.schedule_table,
        num_virtual_stages_per_pipeline_rank=None,
        microbatch_group_size=microbatch_group_size,
        runtime_policy=runtime_policy or {"runtime": "fixed"},
        rewrites=rewrites,
        metadata=metadata or {},
    )


class StrategyVerifier:
    """Deterministic legality checks for generated schedule tables."""

    def __init__(self, constraints: StrategyConstraints, memory_budget_mb: Optional[float] = None):
        self.constraints = constraints
        self.memory_budget_mb = memory_budget_mb

    def verify(self, candidate: StrategyCandidate | StrategyPlan) -> None:
        c = self.constraints
        expected_len = c.num_microbatches * c.num_model_chunks
        if len(candidate.schedule_table) != expected_len:
            raise ValueError(
                f"strategy {candidate.name} has {len(candidate.schedule_table)} tasks; "
                f"expected {expected_len}"
            )

        seen = set()
        for microbatch_id, model_chunk_id in candidate.schedule_table:
            if not 0 <= microbatch_id < c.num_microbatches:
                raise ValueError(f"invalid microbatch id {microbatch_id}")
            if not 0 <= model_chunk_id < c.num_model_chunks:
                raise ValueError(f"invalid model chunk id {model_chunk_id}")
            key = (microbatch_id, model_chunk_id)
            if key in seen:
                raise ValueError(f"duplicate schedule task {key}")
            seen.add(key)

        for model_chunk_id in range(c.num_model_chunks):
            chunk_mbs = [
                microbatch_id
                for microbatch_id, chunk in candidate.schedule_table
                if chunk == model_chunk_id
            ]
            if chunk_mbs != sorted(chunk_mbs):
                raise ValueError(
                    f"strategy {candidate.name} reorders microbatches inside chunk "
                    f"{model_chunk_id}, which is not supported by the current Megatron backend"
                )

        for microbatch_id in range(c.num_microbatches):
            mb_chunks = [
                model_chunk_id
                for microbatch, model_chunk_id in candidate.schedule_table
                if microbatch == microbatch_id
            ]
            if mb_chunks != sorted(mb_chunks):
                raise ValueError(
                    f"strategy {candidate.name} reorders chunks inside microbatch "
                    f"{microbatch_id}, which is not supported without tagged P2P"
                )

        if isinstance(candidate, StrategyPlan):
            self._verify_tasks(candidate)
            self._verify_runtime_policy(candidate)
            self._verify_parallel_shape(candidate)
            self._verify_structured_partition(candidate)
            self._verify_backend_schedule_table(candidate)
            self._verify_memory(candidate)

    def _verify_tasks(self, plan: StrategyPlan) -> None:
        if not plan.tasks:
            return
        task_by_id = {}
        for task in plan.tasks:
            if task.task_id in task_by_id:
                raise ValueError(f"duplicate strategy task id {task.task_id}")
            task_by_id[task.task_id] = task
            if not 0 <= task.microbatch < self.constraints.num_microbatches:
                raise ValueError(f"invalid task microbatch {task.microbatch} in {task.task_id}")
            if not 0 <= task.vp_chunk < self.constraints.num_model_chunks:
                raise ValueError(f"invalid task vp chunk {task.vp_chunk} in {task.task_id}")
            if task.pp_rank < 0:
                raise ValueError(f"invalid task pp rank {task.pp_rank} in {task.task_id}")

        for task in plan.tasks:
            for dep in task.deps:
                if dep not in task_by_id:
                    raise ValueError(f"task {task.task_id} depends on unknown task {dep}")
            if task.kind == "B_WGRAD":
                matching_dgrad = f"B_DGRAD:r{task.pp_rank}:c{task.vp_chunk}:m{task.microbatch}"
                if matching_dgrad in task_by_id and matching_dgrad not in task.deps:
                    raise ValueError(
                        f"WGrad task {task.task_id} must depend on matching DGrad task "
                        f"{matching_dgrad}"
                    )

        expected_fb = {
            (rank, chunk, microbatch)
            for rank in range(self.constraints.pipeline_parallel_size)
            for chunk in range(self.constraints.num_model_chunks)
            for microbatch in range(self.constraints.num_microbatches)
        }
        for kind in ("F", "B_DGRAD"):
            actual = {
                (task.pp_rank, task.vp_chunk, task.microbatch)
                for task in plan.tasks
                if task.kind == kind
            }
            missing = expected_fb - actual
            if missing:
                sample = sorted(missing)[0]
                raise ValueError(f"missing {kind} task for rank/chunk/microbatch {sample}")

        self._verify_pipeline_message_tasks(task_by_id)
        self._verify_compute_task_deps(task_by_id)

    def _verify_pipeline_message_tasks(self, task_by_id: Dict[str, StrategyTask]) -> None:
        pp_size = self.constraints.pipeline_parallel_size
        for task in task_by_id.values():
            rank = task.pp_rank
            chunk = task.vp_chunk
            microbatch = task.microbatch
            if task.kind == "SEND_F" and rank < pp_size - 1:
                match = f"RECV_F:r{rank + 1}:c{chunk}:m{microbatch}"
                if match not in task_by_id:
                    raise ValueError(f"forward send {task.task_id} has no matching recv {match}")
            elif task.kind == "RECV_F" and rank > 0:
                match = f"SEND_F:r{rank - 1}:c{chunk}:m{microbatch}"
                if match not in task_by_id:
                    raise ValueError(f"forward recv {task.task_id} has no matching send {match}")
            elif task.kind == "SEND_B" and rank > 0:
                match = f"RECV_B:r{rank - 1}:c{chunk}:m{microbatch}"
                if match not in task_by_id:
                    raise ValueError(f"backward send {task.task_id} has no matching recv {match}")
            elif task.kind == "RECV_B" and rank < pp_size - 1:
                match = f"SEND_B:r{rank + 1}:c{chunk}:m{microbatch}"
                if match not in task_by_id:
                    raise ValueError(f"backward recv {task.task_id} has no matching send {match}")

    def _verify_compute_task_deps(self, task_by_id: Dict[str, StrategyTask]) -> None:
        pp_size = self.constraints.pipeline_parallel_size
        deps_by_id = {task.task_id: set(task.deps) for task in task_by_id.values()}
        for task in task_by_id.values():
            rank = task.pp_rank
            chunk = task.vp_chunk
            microbatch = task.microbatch
            if task.kind == "F" and rank > 0:
                recv = f"RECV_F:r{rank}:c{chunk}:m{microbatch}"
                if recv in task_by_id and recv not in deps_by_id[task.task_id]:
                    raise ValueError(f"forward task {task.task_id} must depend on {recv}")
            elif task.kind == "SEND_F":
                forward = f"F:r{rank}:c{chunk}:m{microbatch}"
                if forward in task_by_id and forward not in deps_by_id[task.task_id]:
                    raise ValueError(f"forward send {task.task_id} must depend on {forward}")
            elif task.kind == "B_DGRAD":
                forward = f"F:r{rank}:c{chunk}:m{microbatch}"
                if forward in task_by_id and forward not in deps_by_id[task.task_id]:
                    raise ValueError(f"DGrad task {task.task_id} must depend on {forward}")
                if rank < pp_size - 1:
                    recv = f"RECV_B:r{rank}:c{chunk}:m{microbatch}"
                    if recv in task_by_id and recv not in deps_by_id[task.task_id]:
                        raise ValueError(f"DGrad task {task.task_id} must depend on {recv}")
            elif task.kind == "SEND_B":
                dgrad = f"B_DGRAD:r{rank}:c{chunk}:m{microbatch}"
                if dgrad in task_by_id and dgrad not in deps_by_id[task.task_id]:
                    raise ValueError(f"backward send {task.task_id} must depend on {dgrad}")

    def _verify_runtime_policy(self, plan: StrategyPlan) -> None:
        runtime = plan.runtime_policy or {}
        runtime_name = runtime.get("runtime", runtime.get("mode", "fixed"))
        if runtime_name not in {"fixed", "ready-set", "bcp-ready"}:
            raise ValueError(f"unsupported strategy runtime {runtime_name}")
        if runtime_name in {"ready-set", "bcp-ready"} and runtime.get("allow_out_of_order_p2p", False):
            raise ValueError(
                "out-of-order P2P requires tagged P2P runtime; current backend only supports "
                "conservative ready-set or bcp-ready dispatch"
            )

    def _verify_parallel_shape(self, plan: StrategyPlan) -> None:
        plan_vp_size = plan.num_virtual_stages_per_pipeline_rank
        if plan_vp_size is not None and plan_vp_size < 1:
            raise ValueError("num_virtual_stages_per_pipeline_rank must be >= 1 when provided")

        if plan_vp_size is not None and plan_vp_size != self.constraints.num_model_chunks:
            raise ValueError(
                "plan num_virtual_stages_per_pipeline_rank "
                f"{plan_vp_size} does not match verifier num_model_chunks "
                f"{self.constraints.num_model_chunks}"
            )

        if plan.pipeline_layout:
            num_stages = len(plan.pipeline_layout.split("|"))
            if num_stages % self.constraints.pipeline_parallel_size != 0:
                raise ValueError(
                    f"pipeline_layout has {num_stages} stages, which is not divisible by "
                    f"pipeline_parallel_size {self.constraints.pipeline_parallel_size}"
                )
            layout_vp_size = num_stages // self.constraints.pipeline_parallel_size
            if layout_vp_size != self.constraints.num_model_chunks:
                raise ValueError(
                    f"pipeline_layout implies VP={layout_vp_size}, but verifier "
                    f"num_model_chunks={self.constraints.num_model_chunks}"
                )
            if plan_vp_size is not None and plan_vp_size != layout_vp_size:
                raise ValueError(
                    f"plan VP={plan_vp_size} conflicts with pipeline_layout VP={layout_vp_size}"
                )

    def _verify_structured_partition(self, plan: StrategyPlan) -> None:
        stage_count = self.constraints.pipeline_parallel_size * self.constraints.num_model_chunks
        if plan.segment_boundaries:
            boundaries = plan.segment_boundaries
            if len(boundaries) != stage_count + 1:
                raise ValueError(
                    f"segment_boundaries length {len(boundaries)} must equal stage_count+1 "
                    f"({stage_count + 1})"
                )
            if boundaries[0] != 0:
                raise ValueError("segment_boundaries must start at 0")
            for left, right in zip(boundaries, boundaries[1:]):
                if right <= left:
                    raise ValueError(
                        f"segment_boundaries must be strictly increasing, got {left}, {right}"
                    )

        if plan.node_assignment:
            if len(plan.node_assignment) != stage_count:
                raise ValueError(
                    f"node_assignment length {len(plan.node_assignment)} must match stage_count "
                    f"{stage_count}"
                )
            if any(node < 0 for node in plan.node_assignment):
                raise ValueError("node_assignment values must be non-negative")

        if plan.cross_node_boundaries:
            for boundary in plan.cross_node_boundaries:
                if not 0 < boundary < stage_count:
                    raise ValueError(
                        f"cross_node boundary {boundary} must be within (0, {stage_count})"
                    )
            if plan.node_assignment:
                expected = tuple(
                    idx
                    for idx in range(1, len(plan.node_assignment))
                    if plan.node_assignment[idx] != plan.node_assignment[idx - 1]
                )
                if tuple(plan.cross_node_boundaries) != expected:
                    raise ValueError(
                        "cross_node_boundaries do not match node_assignment transitions: "
                        f"{plan.cross_node_boundaries} vs {expected}"
                    )

        if plan.vpp_packing:
            if len(plan.vpp_packing) != self.constraints.pipeline_parallel_size:
                raise ValueError(
                    f"vpp_packing row count {len(plan.vpp_packing)} must match pipeline size "
                    f"{self.constraints.pipeline_parallel_size}"
                )
            for row in plan.vpp_packing:
                if len(row) != self.constraints.num_model_chunks:
                    raise ValueError(
                        "each vpp_packing row must have "
                        f"{self.constraints.num_model_chunks} entries"
                    )
                if any(count <= 0 for count in row):
                    raise ValueError(f"vpp_packing counts must be positive, got {row}")

        if plan.segment_boundaries and plan.vpp_packing:
            boundary_counts = tuple(
                right - left for left, right in zip(plan.segment_boundaries, plan.segment_boundaries[1:])
            )
            packing_counts = tuple(count for row in plan.vpp_packing for count in row)
            if packing_counts != boundary_counts:
                raise ValueError(
                    "vpp_packing does not match segment boundaries: "
                    f"{packing_counts} vs {boundary_counts}"
                )

        if plan.pipeline_layout and plan.segment_boundaries:
            layout_segments = plan.pipeline_layout.split("|")
            if len(layout_segments) != stage_count:
                raise ValueError(
                    f"pipeline_layout stage count {len(layout_segments)} must match stage_count "
                    f"{stage_count}"
                )
            layout_counts = tuple(segment.count("t") for segment in layout_segments)
            boundary_counts = tuple(
                right - left for left, right in zip(plan.segment_boundaries, plan.segment_boundaries[1:])
            )
            if layout_counts != boundary_counts:
                raise ValueError(
                    "pipeline_layout decoder counts do not match segment boundaries: "
                    f"{layout_counts} vs {boundary_counts}"
                )

        if plan.memory_actions:
            legal_actions = {"retain", "recompute", "offload", "checkpoint", "none"}
            for stage_idx, action in plan.memory_actions:
                if not 0 <= stage_idx < stage_count:
                    raise ValueError(
                        f"memory action stage index {stage_idx} must be within [0, {stage_count})"
                    )
                if action not in legal_actions:
                    raise ValueError(f"unsupported memory action {action}")

    def _verify_backend_schedule_table(self, plan: StrategyPlan) -> None:
        """Reject tables the current Megatron VPP loop cannot execute safely.

        The fixed interleaved backend maintains per-chunk input/output tensor queues
        and derives release counts from the default grouped schedule. Dependency-safe
        schedule tables can still break those queue invariants. Until a tagged P2P /
        table-driven executor owns tensor lifetimes directly, executable plans must
        preserve the default table shape.
        """

        runtime = plan.runtime_policy or {}
        runtime_name = runtime.get("runtime", runtime.get("mode", "fixed"))
        if runtime_name != "fixed":
            return
        expected = tuple(
            _default_schedule_table(
                self.constraints.num_microbatches,
                self.constraints.num_model_chunks,
                self.constraints.microbatch_group_size,
            )
        )
        if tuple(plan.schedule_table) != expected:
            raise ValueError(
                f"strategy {plan.name} changes the schedule table, but fixed Megatron VPP "
                "currently requires the default table to preserve tensor queue lifetimes"
            )

    def _verify_memory(self, plan: StrategyPlan) -> None:
        if self.memory_budget_mb is None:
            return
        task_memory = sum(max(0.0, task.est_memory_mb) for task in plan.tasks)
        plan_memory = float(plan.metadata.get("estimated_peak_memory_mb", task_memory))
        if plan_memory > self.memory_budget_mb:
            raise ValueError(
                f"strategy {plan.name} estimated memory {plan_memory:.1f} MB exceeds "
                f"budget {self.memory_budget_mb:.1f} MB"
            )
        if plan.pipeline_layout is not None and not isinstance(plan.pipeline_layout, str):
            raise ValueError("pipeline_layout must be a string when provided")
        if plan.placement and len(plan.placement) != self.constraints.pipeline_parallel_size:
            raise ValueError(
                f"placement length {len(plan.placement)} does not match pipeline size "
                f"{self.constraints.pipeline_parallel_size}"
            )


def _default_schedule_table(
    num_microbatches: int, num_model_chunks: int, microbatch_group_size: int
) -> List[ScheduleEntry]:
    schedule_table: List[ScheduleEntry] = []
    for min_microbatch_id_in_group in range(0, num_microbatches, microbatch_group_size):
        if min_microbatch_id_in_group + microbatch_group_size >= num_microbatches:
            microbatch_range = range(min_microbatch_id_in_group, num_microbatches)
        else:
            microbatch_range = range(
                min_microbatch_id_in_group, min_microbatch_id_in_group + microbatch_group_size
            )
        schedule_table.extend(
            (microbatch_id, model_chunk_id)
            for model_chunk_id in range(num_model_chunks)
            for microbatch_id in microbatch_range
        )
    return schedule_table


def _grouped_schedule_table(
    num_microbatches: int,
    num_model_chunks: int,
    group_sizes: List[int],
) -> List[ScheduleEntry]:
    """Compile a monotonic microbatch partition into a legal grouped schedule."""

    schedule_table: List[ScheduleEntry] = []
    cursor = 0
    for group_size in group_sizes:
        if group_size <= 0:
            raise ValueError(f"group sizes must be positive, got {group_size}")
        if cursor >= num_microbatches:
            break
        end = min(num_microbatches, cursor + group_size)
        schedule_table.extend(
            (microbatch_id, model_chunk_id)
            for model_chunk_id in range(num_model_chunks)
            for microbatch_id in range(cursor, end)
        )
        cursor = end
    if cursor != num_microbatches:
        raise ValueError(
            f"group sizes {group_sizes} cover {cursor} microbatches; expected {num_microbatches}"
        )
    return schedule_table


def _front_loaded_schedule_table(
    num_microbatches: int, num_model_chunks: int, microbatch_group_size: int
) -> List[ScheduleEntry]:
    """A dependency-safe rewrite that shrinks the first scheduling group.

    VPP chunks must execute in increasing chunk order for each microbatch in the
    current Megatron backend. This rewrite keeps that invariant and only changes
    group boundaries, which is enough for an agent/searcher to test whether a
    shorter initial wave reduces warmup/cooldown skew without tagged P2P.
    """

    schedule_table: List[ScheduleEntry] = []
    first_group = max(1, min(microbatch_group_size, num_microbatches) // 2)
    group_starts = [0]
    if first_group < num_microbatches:
        group_starts.extend(range(first_group, num_microbatches, microbatch_group_size))

    for group_index, min_microbatch_id_in_group in enumerate(group_starts):
        group_size = first_group if group_index == 0 else microbatch_group_size
        end = min(num_microbatches, min_microbatch_id_in_group + group_size)
        microbatch_range = range(min_microbatch_id_in_group, end)
        schedule_table.extend(
            (microbatch_id, model_chunk_id)
            for model_chunk_id in range(num_model_chunks)
            for microbatch_id in microbatch_range
        )
    return schedule_table


def _seam_staggered_schedule_table(
    num_microbatches: int,
    num_model_chunks: int,
    microbatch_group_size: int,
) -> List[ScheduleEntry]:
    """Shrink warmup/cooldown groups and thicken the steady-state wave.

    This policy keeps both Megatron backend invariants intact:
    1. each chunk still sees monotonically increasing microbatches
    2. each microbatch still traverses chunks in increasing order

    The only freedom we exercise is how microbatches are partitioned into
    grouped chunk sweeps. Relative to `front-loaded`, this policy also shrinks
    the tail wave and pushes those microbatches into the interior groups, which
    better matches dual-node seam-heavy runs where warmup and cooldown both
    expose cross-node bubbles.
    """

    if num_microbatches <= 1 or microbatch_group_size <= 1:
        return _default_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size)

    default_group_count = max(1, (num_microbatches + microbatch_group_size - 1) // microbatch_group_size)
    if default_group_count < 3:
        return _front_loaded_schedule_table(
            num_microbatches,
            num_model_chunks,
            microbatch_group_size,
        )

    head = max(1, min(microbatch_group_size, num_microbatches) // 2)
    tail = max(1, min(head, num_microbatches - head))
    middle_groups = max(1, default_group_count - 2)
    middle_microbatches = num_microbatches - head - tail
    if middle_microbatches < middle_groups:
        return _front_loaded_schedule_table(
            num_microbatches,
            num_model_chunks,
            microbatch_group_size,
        )

    base = middle_microbatches // middle_groups
    rem = middle_microbatches % middle_groups
    group_sizes = [head]
    group_sizes.extend(base + (1 if idx < rem else 0) for idx in range(middle_groups))
    group_sizes.append(tail)
    return _grouped_schedule_table(num_microbatches, num_model_chunks, group_sizes)


def build_strategy_schedule_table(
    policy: str,
    constraints: StrategyConstraints,
) -> StrategyCandidate:
    """Build and verify a VPP schedule-table candidate.

    The policy names are deliberately stable so an LLM/search agent can emit
    them as rewrite actions. More aggressive policies can be added behind the
    same verifier without changing the Megatron schedule loop.
    """

    policy = (policy or "default").lower()
    if policy == "default":
        table = _default_schedule_table(
            constraints.num_microbatches,
            constraints.num_model_chunks,
            constraints.microbatch_group_size,
        )
        candidate = StrategyCandidate("default", tuple(table), ())
    elif policy in {"front-loaded", "front_loaded"}:
        table = _front_loaded_schedule_table(
            constraints.num_microbatches,
            constraints.num_model_chunks,
            constraints.microbatch_group_size,
        )
        candidate = StrategyCandidate("front-loaded", tuple(table), ("shrink_initial_group",))
    elif policy in {"seam-staggered", "seam_staggered", "seam-stagger"}:
        table = _seam_staggered_schedule_table(
            constraints.num_microbatches,
            constraints.num_model_chunks,
            constraints.microbatch_group_size,
        )
        candidate = StrategyCandidate(
            "seam-staggered",
            tuple(table),
            ("shrink_boundary_groups", "expand_steady_state_groups"),
        )
    else:
        raise ValueError(f"unknown pipeline strategy synthesis policy: {policy}")

    StrategyVerifier(constraints).verify(candidate)
    return candidate


def build_schedule_table_from_policy(
    policy: str,
    num_microbatches: int,
    num_model_chunks: int,
    microbatch_group_size: int,
    pipeline_parallel_size: int,
) -> List[ScheduleEntry]:
    """Pure helper for offline validation and agentic search."""

    candidate = build_strategy_schedule_table(
        policy,
        StrategyConstraints(
            num_microbatches=num_microbatches,
            num_model_chunks=num_model_chunks,
            microbatch_group_size=microbatch_group_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
    )
    return list(candidate.schedule_table)


class CudaTimer:
    """CUDA-aware elapsed timer with CPU fallback."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._use_cuda = torch.cuda.is_available()
        self._start_event = None
        self._end_event = None
        self._start_time = 0.0

    def __enter__(self) -> "CudaTimer":
        if self._use_cuda:
            self._start_event = self._torch.cuda.Event(enable_timing=True)
            self._end_event = self._torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._use_cuda:
            self._end_event.record()
            self._end_event.synchronize()
        else:
            self._start_time = (time.perf_counter() - self._start_time) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        if self._use_cuda:
            return float(self._start_event.elapsed_time(self._end_event))
        return float(self._start_time)
