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
        start_wall = time.time_ns()
        start = time.perf_counter_ns()
        wait_stream = self._trace._current_stream_id()
        completed_before_wait = None
        try:
            completed_before_wait = bool(self._work.is_completed())
        except Exception:
            pass
        result = self._work.wait(*args, **kwargs)
        if not self._waited:
            self._trace.finish(
                self._event_id,
                (time.perf_counter_ns() - start) / 1e6,
                wait_start_ts_ns=start_wall,
                wait_end_ts_ns=time.time_ns(),
                wait_thread_id=threading.get_ident(),
                wait_stream_id=wait_stream,
                completed_before_wait=completed_before_wait,
                wait_count=1,
            )
            self._waited = True
        return result

    def is_completed(self) -> bool:
        return bool(self._work.is_completed())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


class _P2PWorkProxy:
    """Observe deferred P2P request waits without changing request semantics."""

    def __init__(self, work: Any, trace: "ParallelPhaseTrace", event_id: int, token: int):
        self._work = work
        self._trace = trace
        self._event_id = event_id
        self._trace_token = token
        self._waited = False

    def wait(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = self._work.wait(*args, **kwargs)
        self._trace.mark_p2p_request_wait(
            self._trace_token,
            wait_ms=(time.perf_counter_ns() - start) / 1e6,
            double_wait=self._waited,
        )
        self._waited = True
        return result

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
        self._group_tickets: dict[int, int] = {}
        self._p2p_requests: dict[int, int] = {}
        self._p2p_wait_counts: dict[int, int] = {}
        self._next_request_token = 0

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

    def set_collective_label(self, group: Any, label: str) -> None:
        """Explicitly label a non-model-parallel communicator.

        Optimizer and finalize-model-grad paths can use a process group that
        is intentionally distinct from Megatron's canonical DP group.  Those
        paths should register their semantic owner at the call site instead
        of forcing the trace analyzer to infer ownership from world size.
        """
        self.register_group(group, label)

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

    @staticmethod
    def _current_stream_id() -> int | None:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_stream().cuda_stream)
        except Exception:
            pass
        return None

    def _next_group_ticket(self, group: Any) -> int:
        key = _group_key(group)
        with self._lock:
            ticket = self._group_tickets.get(key, 0)
            self._group_tickets[key] = ticket + 1
        return ticket

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
        metadata.setdefault("thread_id", threading.get_ident())
        metadata.setdefault("stream_id", self._current_stream_id())
        metadata.setdefault("group_ticket", self._next_group_ticket(group))
        metadata.setdefault("group_key", _group_key(group))
        metadata.setdefault("issue_monotonic_ns", time.perf_counter_ns())
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
        self, event_id: int, wait_ms: float | None = None, api_ms: float | None = None,
        **metadata: Any,
    ) -> None:
        with self._lock:
            for event in reversed(self.events):
                if event.get("event_id") == event_id:
                    if wait_ms is not None:
                        event["wait_ms"] = float(wait_ms)
                    if api_ms is not None:
                        event["api_ms"] = float(api_ms)
                    event.update(metadata)
                    event["complete_ts_ns"] = time.time_ns()
                    return

    def update(self, event_id: int, **metadata: Any) -> None:
        """Attach late metadata to an already-issued event."""
        with self._lock:
            for event in reversed(self.events):
                if event.get("event_id") == event_id:
                    event.update(metadata)
                    return

    def register_p2p_requests(self, event_id: int, requests: Any) -> Any:
        """Bind returned P2P work objects to their issue event."""
        if isinstance(requests, dict):
            values = list(requests.values())
        elif isinstance(requests, (list, tuple)):
            values = list(requests)
        else:
            values = []
        request_ids = []
        wrapped = []
        for request in values:
            with self._lock:
                request_id = self._next_request_token
                self._next_request_token += 1
            proxy = _P2PWorkProxy(request, self, event_id, request_id)
            self._p2p_requests[request_id] = event_id
            self._p2p_wait_counts[request_id] = 0
            request_ids.append(request_id)
            wrapped.append(proxy)
        self.update(event_id, request_count=len(values), request_ids=request_ids)
        if isinstance(requests, dict):
            return {key: proxy for key, proxy in zip(requests.keys(), wrapped)}
        if isinstance(requests, tuple):
            return tuple(wrapped)
        if isinstance(requests, list):
            return wrapped
        return requests

    def mark_p2p_request_wait(self, request_id: int, *, wait_ms: float, double_wait: bool) -> None:
        event_id = self._p2p_requests.get(request_id)
        if event_id is None:
            return
        count = self._p2p_wait_counts.get(request_id, 0) + 1
        self._p2p_wait_counts[request_id] = count
        self.update(
            event_id,
            p2p_waited_count=sum(
                value for request, value in self._p2p_wait_counts.items()
                if self._p2p_requests.get(request) == event_id
            ),
            p2p_double_wait_count=(
                sum(
                    1 for request, value in self._p2p_wait_counts.items()
                    if self._p2p_requests.get(request) == event_id and value > 1
                )
            ),
            last_p2p_wait_ms=float(wait_ms),
            p2p_wait_thread_id=threading.get_ident(),
            p2p_wait_stream_id=self._current_stream_id(),
            p2p_wait_end_ts_ns=time.time_ns(),
            double_wait=bool(double_wait),
        )

    def p2p_wait_metadata(self, requests: Any) -> dict[str, Any]:
        if isinstance(requests, dict):
            values = list(requests.values())
        elif isinstance(requests, (list, tuple)):
            values = list(requests)
        else:
            values = []
        parent_ids = []
        unknown = 0
        for request in values:
            parent = self._p2p_requests.get(getattr(request, "_trace_token", None))
            if parent is None:
                unknown += 1
            else:
                parent_ids.append(parent)
        return {
            "request_count": len(values),
            "request_parent_event_ids": parent_ids,
            "unknown_request_count": unknown,
        }

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
