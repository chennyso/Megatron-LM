# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Conservative runtime arbitration for PP/VPP strategy plans.

The initial runtime intentionally preserves Megatron's P2P matching semantics.
It can select an alternate ready task only when the task is marked local-safe by
the strategy plan. Full out-of-order P2P requires tagged send/recv support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class RuntimeTask:
    """Runtime task handle used by conservative ready-set dispatch."""

    task_id: str
    priority: float = 0.0
    local_safe: bool = False
    memory_mb: float = 0.0


@dataclass
class RuntimeMemoryState:
    """Minimal memory state used by ready-set filtering."""

    used_mb: float = 0.0
    budget_mb: Optional[float] = None

    def can_run(self, task: RuntimeTask) -> bool:
        if self.budget_mb is None:
            return True
        return self.used_mb + max(0.0, task.memory_mb) <= self.budget_mb


@dataclass
class ReadySetRuntime:
    """Select tasks from a ready set while preserving fixed schedule semantics by default."""

    mode: str = "fixed"
    tasks: Dict[str, RuntimeTask] = field(default_factory=dict)
    ready: Set[str] = field(default_factory=set)
    completed: Set[str] = field(default_factory=set)

    def register(self, task: RuntimeTask) -> None:
        self.tasks[task.task_id] = task

    def mark_ready(self, task_id: str) -> None:
        if task_id not in self.tasks:
            self.register(RuntimeTask(task_id=task_id))
        if task_id not in self.completed:
            self.ready.add(task_id)

    def complete(self, task_id: str) -> None:
        self.ready.discard(task_id)
        self.completed.add(task_id)

    def pop_next(
        self,
        hint_task_id: str,
        ready_tasks: Optional[Iterable[str]] = None,
        memory_state: Optional[RuntimeMemoryState] = None,
    ) -> str:
        """Return the selected task id.

        In fixed mode, this returns the hint. In ready-set mode, it only bypasses
        the hint if the hint is not ready and a local-safe ready task is legal.
        """

        if self.mode == "fixed":
            return hint_task_id

        ready = set(ready_tasks) if ready_tasks is not None else set(self.ready)
        if hint_task_id in ready:
            return hint_task_id

        memory_state = memory_state or RuntimeMemoryState()
        legal = [
            self.tasks[task_id]
            for task_id in ready
            if self.tasks[task_id].local_safe and memory_state.can_run(self.tasks[task_id])
        ]
        if not legal:
            return hint_task_id
        return max(legal, key=lambda task: task.priority).task_id


@dataclass
class BubbleFillWork:
    """Small unit of local work that can be run before a blocking P2P wait.

    The callable must preserve pipeline message order. This is intended for
    local-safe work such as delayed WGrad chunks, not for arbitrary forward or
    backward reordering.
    """

    name: str
    run: Callable[[], None]
    priority: float = 0.0
    estimated_ms: float = 0.0
    memory_mb: float = 0.0


@dataclass
class BubbleFillResult:
    """Outcome of one pre-wait bubble filling attempt."""

    name: str
    ran: bool
    elapsed_ms: float = 0.0
    reason: str = ""


@dataclass
class BCPReadyRuntime:
    """Critical-path aware local runtime for conservative PP/VPP execution.

    This runtime does not change Megatron's P2P matching order. It only decides
    whether local-safe work should run immediately before a blocking P2P wait.
    That gives the search/runtime stack a real hot-path hook while preserving
    the fixed interleaved schedule as the fallback.
    """

    mode: str = "fixed"
    p2p_credit_budget: Optional[int] = None
    memory_budget_mb: Optional[float] = None
    min_wait_to_fill_ms: float = 0.0
    work: list[BubbleFillWork] = field(default_factory=list)
    outstanding_p2p: int = 0
    fill_attempts: int = 0
    fill_runs: int = 0

    @property
    def enabled(self) -> bool:
        return self.mode in {"bcp-ready", "ready-set"}

    def register_work(self, work: BubbleFillWork) -> None:
        self.work.append(work)
        self.work.sort(key=lambda item: item.priority, reverse=True)

    def mark_p2p_issued(self, count: int = 1) -> None:
        self.outstanding_p2p += max(0, count)

    def mark_p2p_completed(self, count: int = 1) -> None:
        self.outstanding_p2p = max(0, self.outstanding_p2p - max(0, count))

    def _can_run(self, work: BubbleFillWork, expected_wait_ms: Optional[float]) -> tuple[bool, str]:
        if not self.enabled:
            return False, "runtime_disabled"
        if expected_wait_ms is not None and expected_wait_ms < self.min_wait_to_fill_ms:
            return False, "wait_window_too_small"
        if self.p2p_credit_budget is not None and self.outstanding_p2p > self.p2p_credit_budget:
            return False, "p2p_credit_budget_exceeded"
        if self.memory_budget_mb is not None and work.memory_mb > self.memory_budget_mb:
            return False, "memory_budget_exceeded"
        return True, ""

    def pop_fill_work(self, expected_wait_ms: Optional[float] = None) -> BubbleFillWork | None:
        remaining: list[BubbleFillWork] = []
        selected: BubbleFillWork | None = None
        for item in self.work:
            can_run, _reason = self._can_run(item, expected_wait_ms)
            if selected is None and can_run:
                selected = item
            else:
                remaining.append(item)
        self.work = remaining
        return selected

    def run_one_fill(
        self,
        timer_factory: Callable[[], object],
        expected_wait_ms: Optional[float] = None,
    ) -> BubbleFillResult:
        self.fill_attempts += 1
        if not self.work:
            return BubbleFillResult(name="", ran=False, reason="no_work")
        item = self.pop_fill_work(expected_wait_ms)
        if item is None:
            _can_run, reason = self._can_run(self.work[0], expected_wait_ms)
            return BubbleFillResult(name=self.work[0].name, ran=False, reason=reason)

        with timer_factory() as timer:
            item.run()
        self.fill_runs += 1
        return BubbleFillResult(
            name=item.name,
            ran=True,
            elapsed_ms=float(getattr(timer, "elapsed_ms", 0.0)),
        )
