# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Conservative runtime arbitration for PP/VPP strategy plans.

The initial runtime intentionally preserves Megatron's P2P matching semantics.
It can select an alternate ready task only when the task is marked local-safe by
the strategy plan. Full out-of-order P2P requires tagged send/recv support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set


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
