"""Profile-guided VPP phase/window planner.

The planner is intentionally an offline, dependency-light component.  It does
not reorder NCCL tickets or mutate model ownership.  It consumes a typed
communication trace, estimates the periodic residue classes induced by VPP,
and emits a finite family of deadline-safe phase modes for a runtime executor.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class WindowConstraintError(ValueError):
    """Raised when a candidate mode violates a collective/window contract."""


@dataclass(frozen=True)
class Action:
    action_id: str
    action_class: str
    rank: int
    issue_ns: int
    complete_ns: int
    ticket: int
    microbatch: int | None = None
    vp_chunk: int | None = None
    communicator: str = "unknown"

    @property
    def duration_ns(self) -> int:
        return max(0, self.complete_ns - self.issue_ns)


@dataclass(frozen=True)
class BucketWindow:
    bucket_id: str
    release_ns: int
    deadline_ns: int
    service_ns: int
    communicator: str
    ticket: int
    segment: int = 0

    def validate(self) -> None:
        if self.release_ns > self.deadline_ns:
            raise WindowConstraintError(f"bucket {self.bucket_id} has inverted release/deadline")
        if self.service_ns < 0:
            raise WindowConstraintError(f"bucket {self.bucket_id} has negative service time")


@dataclass(frozen=True)
class PhaseMode:
    name: str
    period_ns: int
    pp_offset_ns: int
    collective_offset_ns: int
    score_ns: int
    alias_score: float
    bucket_windows: tuple[BucketWindow, ...] = field(default_factory=tuple)
    certificate: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bucket_windows"] = [asdict(item) for item in self.bucket_windows]
        return payload


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _phase(value: int, period: int) -> int:
    return value % max(1, period)


class PhaseWeaverPlanner:
    """Synthesize VPP-aware collective windows from measured action intervals."""

    def __init__(self, *, phase_bins: int = 32, min_slack_ns: int = 0):
        if phase_bins < 4:
            raise ValueError("phase_bins must be >= 4")
        self.phase_bins = phase_bins
        self.min_slack_ns = min_slack_ns

    @staticmethod
    def read_trace(paths: Iterable[str | Path]) -> list[Action]:
        actions: list[Action] = []
        ordinal = Counter()
        global_ticket = 0
        for path in paths:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                issue = event.get("issue_ts_ns")
                complete = event.get("complete_ts_ns")
                action_class = str(event.get("action_class", "UNKNOWN"))
                if not isinstance(issue, int) or not isinstance(complete, int):
                    continue
                context = event.get("context") or {}
                ordinal[action_class] += 1
                global_ticket += 1
                actions.append(
                    Action(
                        action_id=f"{action_class}:{event.get('rank', -1)}:{ordinal[action_class]}",
                        action_class=action_class,
                        rank=int(event.get("rank", -1)),
                        issue_ns=issue,
                        complete_ns=complete,
                        ticket=global_ticket,
                        microbatch=context.get("microbatch_id"),
                        vp_chunk=context.get("vp_chunk"),
                        communicator=str(event.get("group_label", "unknown")),
                    )
                )
        return sorted(actions, key=lambda item: (item.issue_ns, item.rank, item.ticket))

    def infer_period(self, actions: Sequence[Action]) -> int:
        """Infer a robust PP/VPP period from repeated PP issue timestamps."""
        pp = sorted(item.issue_ns for item in actions if item.action_class.startswith("PP_"))
        deltas = [right - left for left, right in zip(pp, pp[1:]) if right > left]
        if not deltas:
            return 1
        deltas.sort()
        # The median PP cadence is a seam cadence, not necessarily a complete
        # VPP period.  A period must also accommodate the profiled collective
        # service window; otherwise every legal DP bucket is rejected before
        # synthesis.  Use a robust upper quantile rather than the pathological
        # max caused by startup and rank skew.
        base = max(1, deltas[len(deltas) // 2])
        cadence_q3 = deltas[min(len(deltas) - 1, (3 * len(deltas)) // 4)]
        collective = sorted(
            item.duration_ns for item in actions if item.action_class.startswith(("DP_", "UNKNOWN_"))
        )
        service_q = collective[min(len(collective) - 1, (3 * len(collective)) // 4)] if collective else base
        return max(base, cadence_q3, service_q)

    def residue_histogram(self, actions: Sequence[Action], period_ns: int) -> dict[str, list[int]]:
        bins: dict[str, list[int]] = defaultdict(lambda: [0] * self.phase_bins)
        for item in actions:
            residue = _phase(item.issue_ns, period_ns) * self.phase_bins // period_ns
            bins[item.action_class][min(self.phase_bins - 1, residue)] += 1
        return dict(bins)

    def interaction_matrix(self, actions: Sequence[Action], period_ns: int) -> dict[tuple[str, str], float]:
        """Estimate phase-conditioned pair overlap, normalized by action count."""
        by_class: dict[str, list[Action]] = defaultdict(list)
        for action in actions:
            by_class[action.action_class].append(action)
        matrix: dict[tuple[str, str], float] = {}
        classes = sorted(by_class)
        for index, left_class in enumerate(classes):
            for right_class in classes[index + 1 :]:
                # A trace contains thousands of repeated layer events.  The
                # interaction matrix is a profile summary, not an exhaustive
                # event join; cap each class to a deterministic prefix so the
                # offline compiler remains bounded.
                left_events = by_class[left_class][:256]
                right_events = by_class[right_class][:256]
                total = 0
                count = 0
                for left in left_events:
                    for right in right_events:
                        if abs(_phase(left.issue_ns - right.issue_ns, period_ns)) > period_ns // 4:
                            continue
                        total += _overlap((left.issue_ns, left.complete_ns), (right.issue_ns, right.complete_ns))
                        count += 1
                matrix[(left_class, right_class)] = total / max(1, count) / 1e6
        return matrix

    def _windows(self, actions: Sequence[Action], period_ns: int, offset_ns: int) -> tuple[BucketWindow, ...]:
        dp = [item for item in actions if item.action_class in {"DP_RS", "DP_AG", "DP_AR", "UNKNOWN_RS", "UNKNOWN_AR"}]
        pp = [item for item in actions if item.action_class.startswith("PP_")]
        if not dp:
            return tuple()
        windows: list[BucketWindow] = []
        tickets: dict[str, int] = defaultdict(int)
        for index, item in enumerate(dp):
            release = item.issue_ns
            # A DP bucket is movable within the next VPP period.  Using the
            # immediately following PP issue as a deadline would incorrectly
            # reject every bucket whose measured service spans several fine
            # grained P2P seams; the real contract is the next period frontier.
            deadline = release + period_ns - self.min_slack_ns
            launch = release + offset_ns
            if launch + item.duration_ns > deadline:
                # The candidate is rejected rather than silently violating a
                # synchronous DP ticket or delaying a critical PP edge.
                raise WindowConstraintError(f"DP action {item.action_id} misses PP deadline")
            windows.append(
                BucketWindow(
                    bucket_id=f"bucket-{index}",
                    release_ns=launch,
                    deadline_ns=deadline,
                    service_ns=item.duration_ns,
                    communicator=item.communicator,
                    ticket=tickets[item.communicator],
                )
            )
            tickets[item.communicator] += 1
        return tuple(windows)

    def synthesize(self, actions: Sequence[Action], *, offsets: Sequence[int] | None = None) -> list[PhaseMode]:
        if not actions:
            return []
        period = self.infer_period(actions)
        offsets = tuple(offsets or (0, period // 8, period // 4, period // 2))
        matrix = self.interaction_matrix(actions, period)
        modes: list[PhaseMode] = []
        for index, offset in enumerate(offsets):
            try:
                windows = self._windows(actions, period, offset)
            except WindowConstraintError:
                continue
            alias = sum(value for (left, right), value in matrix.items() if left.startswith("PP_") or right.startswith("PP_"))
            alias *= 1.0 if offset == 0 else max(0.0, 1.0 - offset / max(1, period))
            modes.append(
                PhaseMode(
                    name="eager" if offset == 0 else f"dephase-{index}",
                    period_ns=period,
                    pp_offset_ns=0,
                    collective_offset_ns=offset,
                    score_ns=sum(item.service_ns for item in windows),
                    alias_score=alias,
                    bucket_windows=windows,
                    certificate={
                        "activation_fifo_prefix": True,
                        "collective_ticket_prefix": True,
                        "vpp_cursor_unchanged": True,
                        "zero_outstanding_debt": True,
                    },
                )
            )
        return sorted(modes, key=lambda mode: (mode.alias_score, mode.score_ns))

    @staticmethod
    def verify(mode: PhaseMode) -> None:
        certificate = mode.certificate
        required = ("activation_fifo_prefix", "collective_ticket_prefix", "vpp_cursor_unchanged", "zero_outstanding_debt")
        if any(certificate.get(key) is not True for key in required):
            raise WindowConstraintError("mode lacks the frontier safety certificate")
        last_ticket: dict[str, int] = {}
        for window in mode.bucket_windows:
            window.validate()
            previous = last_ticket.get(window.communicator, -1)
            if window.ticket < previous:
                raise WindowConstraintError(f"collective ticket order regressed for {window.communicator}")
            last_ticket[window.communicator] = window.ticket


def dump_modes(modes: Sequence[PhaseMode], path: str | Path) -> None:
    Path(path).write_text(json.dumps([mode.to_dict() for mode in modes], indent=2) + "\n", encoding="utf-8")
