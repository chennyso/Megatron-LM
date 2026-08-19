"""Unit tests for PhaseWeaver's offline legality contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PLANNER = Path(__file__).resolve().parents[3] / "megatron/core/phaseweaver/planner.py"
SPEC = importlib.util.spec_from_file_location("phaseweaver_planner_test", PLANNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Action = MODULE.Action
BucketWindow = MODULE.BucketWindow
PhaseMode = MODULE.PhaseMode
PhaseWeaverPlanner = MODULE.PhaseWeaverPlanner
WindowConstraintError = MODULE.WindowConstraintError


def _actions() -> list:
    return [
        Action("pp0", "PP_FWD", 0, 0, 100, 0, microbatch=0, vp_chunk=0, communicator="PP"),
        Action("dp0", "DP_RS", 0, 200, 260, 0, communicator="DP"),
        Action("pp1", "PP_BWD", 0, 1_000, 1_120, 1, microbatch=0, vp_chunk=1, communicator="PP"),
        Action("dp1", "DP_RS", 0, 1_400, 1_470, 1, communicator="DP"),
        Action("pp2", "PP_FWD", 0, 2_000, 2_120, 2, microbatch=1, vp_chunk=0, communicator="PP"),
    ]


def test_synthesized_modes_preserve_frontier_contract() -> None:
    planner = PhaseWeaverPlanner()
    modes = planner.synthesize(_actions(), offsets=(0, 100, 200))
    assert modes
    for mode in modes:
        planner.verify(mode)
        assert mode.certificate["activation_fifo_prefix"] is True
        assert mode.certificate["collective_ticket_prefix"] is True


def test_rejects_ticket_regression() -> None:
    mode = PhaseMode(
        name="invalid",
        period_ns=1_000,
        pp_offset_ns=0,
        collective_offset_ns=0,
        score_ns=0,
        alias_score=0,
        bucket_windows=(
            BucketWindow("a", 0, 100, 10, "DP", 1),
            BucketWindow("b", 10, 100, 10, "DP", 0),
        ),
        certificate={
            "activation_fifo_prefix": True,
            "collective_ticket_prefix": True,
            "vpp_cursor_unchanged": True,
            "zero_outstanding_debt": True,
        },
    )
    with pytest.raises(WindowConstraintError, match="ticket order"):
        PhaseWeaverPlanner.verify(mode)


def test_rejects_window_that_misses_period_deadline() -> None:
    planner = PhaseWeaverPlanner()
    actions = [
        Action("pp", "PP_FWD", 0, 0, 10, 0, communicator="PP"),
        Action("dp", "DP_RS", 0, 20, 1_020, 0, communicator="DP"),
        Action("pp-next", "PP_FWD", 0, 100, 110, 1, communicator="PP"),
    ]
    assert planner.synthesize(actions, offsets=(100,)) == []
