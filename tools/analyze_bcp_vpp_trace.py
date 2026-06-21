#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Analyze PP/VPP traces with the bounded-critical-path objective.

This helper is intentionally read-only: it does not emit a StrategyPlan. Use it
to turn Megatron pipeline strategy traces into the diagnostics needed by the
BCP-VPP search loop and by experiment reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _load_module(name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True, help="Trace JSON files from PP ranks")
    parser.add_argument("--output", required=True, help="Path to write BCP diagnostics JSON")
    parser.add_argument("--num-model-chunks", type=int, required=True)
    parser.add_argument("--baseline-num-model-chunks", type=int, default=None)
    parser.add_argument("--microbatch-group-size", type=int, required=True)
    parser.add_argument("--policy", default="default")
    parser.add_argument("--runtime", choices=["fixed", "ready-set"], default="fixed")
    parser.add_argument("--layout-kind", choices=["default", "custom"], default="default")
    parser.add_argument("--bcp-activation-budget-mb", type=float, default=None)
    parser.add_argument("--bcp-p2p-credit-budget", type=int, default=None)
    parser.add_argument("--bcp-fb-delay-budget", type=int, default=None)
    args = parser.parse_args()

    search = _load_module("search_pipeline_strategy", "tools/search_pipeline_strategy.py")
    agent = _load_module("pipeline_strategy_agent", "tools/pipeline_strategy_agent.py")
    events = agent.load_events(args.trace)
    budget = search.BcpBudget(
        activation_peak_mb=args.bcp_activation_budget_mb,
        p2p_credit=args.bcp_p2p_credit_budget,
        fb_delay_steps=args.bcp_fb_delay_budget,
    )
    baseline_vpp = args.baseline_num_model_chunks or args.num_model_chunks
    stats = search._bcp_stats(
        events,
        group_size=args.microbatch_group_size,
        policy=args.policy,
        vpp_size=args.num_model_chunks,
        baseline_vpp_size=baseline_vpp,
        layout="custom" if args.layout_kind == "custom" else None,
        runtime=args.runtime,
        budget=budget,
    )
    payload = {
        "bcp_stats": asdict(stats),
        "bcp_budget": asdict(budget),
        "diagnostics": search._trace_diagnostics(events, args.num_model_chunks),
        "bottlenecks": [asdict(item) for item in agent.attribute_bottlenecks(events)],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
