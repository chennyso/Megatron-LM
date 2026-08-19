#!/usr/bin/env python3
"""Synthesize and verify PhaseWeaver modes from phase JSONL traces."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

_PLANNER_PATH = Path(__file__).resolve().parents[0].parent / "megatron" / "core" / "phaseweaver" / "planner.py"
_SPEC = importlib.util.spec_from_file_location("phaseweaver_planner", _PLANNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
PhaseWeaverPlanner = _MODULE.PhaseWeaverPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.trace_glob, recursive=True))
    planner = PhaseWeaverPlanner()
    actions = planner.read_trace(paths)
    modes = planner.synthesize(actions)
    for mode in modes:
        planner.verify(mode)
    payload = {
        "trace_files": len(paths),
        "actions": len(actions),
        "period_ns": planner.infer_period(actions) if actions else None,
        "modes": [mode.to_dict() for mode in modes],
        "claim_boundary": "Offline certified candidates only; no speedup is claimed until held-out execution.",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
