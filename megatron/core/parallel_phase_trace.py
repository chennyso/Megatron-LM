"""Low-overhead trace of cross-parallel communication phases.

The tracer is opt-in and intentionally lives outside any individual parallel
implementation.  It records the operation boundary seen by the Python
distributed API, so PP P2P, TP/SP collectives, EP collectives and ZeRO/FSDP
bucket collectives can be compared in one event schema.  It does not change
collective membership, ordering or tensor values.
"""

from __future__ import annotations

import json
import atexit
import threading
import time
from pathlib import Path
from typing import Any, Callable


_PATCH_LOCK = threading.Lock()
_ACTIVE: "ParallelPhaseTrace | None" = None


def _group_key(group: Any) -> int | None:
    return id(group) if group is not None else None


def _tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _group_size(group: Any) -> int | None:
    try:
        if group is not None:
            return int(group.size())
        import torch.distributed as dist

        return int(dist.get_world_size()) if dist.is_initialized() else 1
    except Exception:
        return None


class _WorkProxy:
    """Preserve the common Work interface while measuring wait completion."""

    def __init__(self, work: Any, trace: "ParallelPhaseTrace", event_id: int):
        self._work = work
        self._trace = trace
        self._event_id = event_id
        self._waited = False

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = self._work.wait(*args, **kwargs)
        if not self._waited:
            self._trace.finish(self._event_id, (time.perf_counter_ns() - start) / 1e6)
            self._waited = True
        return result

    def is_completed(self) -> bool:
        return bool(self._work.is_completed())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


class ParallelPhaseTrace:
    """Process-local event sink and distributed API patch manager."""

    def __init__(
        self, path: str | Path, *, rank: int | None = None, persistent: bool = False
    ):
        self.path = Path(path)
        self.rank = rank
        self.persistent = persistent
        self.context: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self._originals: dict[str, Any] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._group_labels: dict[int, str] = {}

    def register_group(self, group: Any, label: str) -> None:
        key = _group_key(group)
        if key is not None:
            self._group_labels[key] = str(label)

    def register_model_parallel_groups(self) -> None:
        """Register groups after Megatron has finished creating them.

        The process-level tracer is intentionally installed before optimizer
        construction, which is earlier than Megatron's model-parallel group
        initialization in some launch paths.  Group registration therefore
        has to be repeatable and lazy rather than tied to ``install()``.
        """
        try:
            from megatron.core import parallel_state

            for label, getter in (
                ("PP", parallel_state.get_pipeline_model_parallel_group),
                ("TP", parallel_state.get_tensor_model_parallel_group),
                ("CP", parallel_state.get_context_parallel_group),
                ("DP", parallel_state.get_data_parallel_group),
                ("EP", parallel_state.get_expert_model_parallel_group),
            ):
                try:
                    group = getter()
                    if group is not None and _group_size(group) and _group_size(group) > 1:
                        self.register_group(group, label)
                except (AssertionError, RuntimeError):
                    continue
        except (ImportError, RuntimeError):
            return

    def _action_class(self, group: Any, op: str) -> str:
        label = self._group_labels.get(_group_key(group), "UNKNOWN")
        suffix = {
            "all_reduce": "AR",
            "all_gather": "AG",
            "all_gather_into_tensor": "AG",
            "_all_gather_base": "AG",
            "reduce_scatter": "RS",
            "reduce_scatter_tensor": "RS",
            "_reduce_scatter_base": "RS",
            "all_to_all": "A2A",
            "all_to_all_single": "A2A",
        }.get(op, op)
        return f"{label}_{suffix}"

    def set_context(self, **values: Any) -> None:
        # Context updates are incremental: schedule metadata (PP/TP/DP rank,
        # VPP size) must survive per-microbatch phase updates.
        self.context.update({key: value for key, value in values.items() if value is not None})

    def _record(self, event: dict[str, Any]) -> int:
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            event["event_id"] = event_id
            event["rank"] = self.rank
            event["context"] = dict(self.context)
            self.events.append(event)
        return event_id

    def start(self, op: str, payload_bytes: int, group: Any, **metadata: Any) -> int:
        return self._record(
            {
                "name": "collective_issue",
                "op": op,
                "payload_bytes": int(payload_bytes),
                "group_size": _group_size(group),
                "issue_ts_ns": time.time_ns(),
                **metadata,
            }
        )

    def finish(
        self, event_id: int, wait_ms: float | None = None, api_ms: float | None = None
    ) -> None:
        with self._lock:
            for event in reversed(self.events):
                if event.get("event_id") == event_id:
                    if wait_ms is not None:
                        event["wait_ms"] = float(wait_ms)
                    if api_ms is not None:
                        event["api_ms"] = float(api_ms)
                    event["complete_ts_ns"] = time.time_ns()
                    return

    def flush(self) -> None:
        if not self.events:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.events.clear()

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            group = kwargs.get("group")
            if group is None:
                # torch.distributed collective signatures place group after
                # tensor/op or tensor_list/output/input split arguments.
                group_index = {
                    "all_reduce": 2,
                    "all_gather": 2,
                    "all_gather_into_tensor": 2,
                    "_all_gather_base": 2,
                    # ``op`` precedes ``group`` in the reduce-scatter APIs.
                    "reduce_scatter": 3,
                    "reduce_scatter_tensor": 2,
                    "_reduce_scatter_base": 3,
                    "all_to_all": 2,
                    # output/input split lists precede ``group`` here.
                    "all_to_all_single": 4,
                }.get(name)
                if group_index is not None and len(args) > group_index:
                    group = args[group_index]
            payload = _tensor_bytes(args[0] if args else kwargs.get("tensor"))
            if name in {"all_gather", "reduce_scatter"} and len(args) > 1:
                payload = max(payload, _tensor_bytes(args[1]))
            async_op = bool(kwargs.get("async_op", False))
            event_id = self.start(
                name, payload, group, async_op=async_op,
                group_label=self._group_labels.get(_group_key(group), "UNKNOWN"),
                action_class=self._action_class(group, name),
            )
            started_ns = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            api_ms = (time.perf_counter_ns() - started_ns) / 1e6
            if async_op and result is not None and hasattr(result, "wait"):
                self.finish(event_id, api_ms=api_ms)
                return _WorkProxy(result, self, event_id)
            self.finish(event_id, api_ms=api_ms)
            return result

        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped

    def install(self) -> None:
        import torch.distributed as dist

        names = (
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "_all_gather_base",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "_reduce_scatter_base",
            "all_to_all",
            "all_to_all_single",
        )
        with _PATCH_LOCK:
            global _ACTIVE
            if _ACTIVE is not None:
                return
            for name in names:
                fn = getattr(dist, name, None)
                if fn is not None:
                    self._originals[name] = fn
                    setattr(dist, name, self._wrap(name, fn))
            _ACTIVE = self

    def uninstall(self) -> None:
        import torch.distributed as dist

        with _PATCH_LOCK:
            # A process-level tracer must survive the short-lived PP schedule
            # objects so optimizer/ZeRO collectives after forward-backward are
            # recorded in the same event stream.
            if self.persistent:
                self.flush()
                return
            for name, fn in self._originals.items():
                setattr(dist, name, fn)
            self._originals.clear()
            self.flush()
            global _ACTIVE
            if _ACTIVE is self:
                _ACTIVE = None


def install_from_config(config: Any, path: str | Path | None = None) -> ParallelPhaseTrace | None:
    """Install once when ``pipeline_strategy_phase_trace_path`` is configured."""

    if _ACTIVE is not None:
        # A persistent tracer may have been installed before Megatron created
        # its process groups.  Refresh labels on every schedule entry so
        # DP/EP/PP events never fall back to UNKNOWN solely due to timing.
        _ACTIVE.register_model_parallel_groups()
        return _ACTIVE
    trace_path = path or getattr(config, "pipeline_strategy_phase_trace_path", None)
    if not trace_path:
        return None
    rank = None
    try:
        import torch.distributed as dist

        rank = int(dist.get_rank()) if dist.is_initialized() else 0
    except Exception:
        rank = 0
    trace = ParallelPhaseTrace(str(trace_path).format(rank=rank), rank=rank)
    trace.install()
    try:
        trace.register_model_parallel_groups()
    except (ImportError, RuntimeError):
        pass
    return trace


def install_persistent(path: str | Path, *, rank: int | None = None) -> ParallelPhaseTrace:
    """Install a process-level tracer before model/optimizer construction.

    Unlike a schedule-local tracer, this instance remains installed when a PP
    schedule returns, allowing the same trace to contain optimizer and
    distributed-optimizer collectives.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE
    trace = ParallelPhaseTrace(path, rank=rank, persistent=True)
    trace.install()
    # This is a no-op before model-parallel initialization and is refreshed by
    # ``install_from_config`` once the groups exist.
    trace.register_model_parallel_groups()
    atexit.register(trace.uninstall)
    return trace


def get_active() -> ParallelPhaseTrace | None:
    """Return the process-local tracer for iteration-level annotations."""
    return _ACTIVE


__all__ = ["ParallelPhaseTrace", "install_from_config", "install_persistent", "get_active"]
