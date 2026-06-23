#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate pipeline strategy synthesis policies without launching training."""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path


def _load_strategy_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "megatron" / "core" / "pipeline_parallel" / "strategy_synthesizer.py"
    spec = importlib.util.spec_from_file_location("strategy_synthesizer", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="default",
        choices=["default", "front-loaded", "seam-staggered"],
    )
    parser.add_argument("--plan", default=None, help="Optional StrategyPlan JSON to validate")
    parser.add_argument("--num-microbatches", type=int, required=True)
    parser.add_argument("--num-model-chunks", type=int, required=True)
    parser.add_argument("--microbatch-group-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    synth = _load_strategy_module()
    if args.plan:
        plan = synth.load_strategy_plan(args.plan)
        num_model_chunks = plan.num_virtual_stages_per_pipeline_rank or args.num_model_chunks
        if plan.pipeline_layout:
            num_model_chunks = len(plan.pipeline_layout.split("|")) // args.pipeline_parallel_size
        microbatch_group_size = plan.microbatch_group_size or args.microbatch_group_size
        constraints = synth.StrategyConstraints(
            num_microbatches=args.num_microbatches,
            num_model_chunks=num_model_chunks,
            microbatch_group_size=microbatch_group_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        )
        synth.StrategyVerifier(constraints, args.memory_budget_mb).verify(plan)
        payload = {"valid": True, "plan": synth.strategy_plan_to_dict(plan)}
    else:
        constraints = synth.StrategyConstraints(
            num_microbatches=args.num_microbatches,
            num_model_chunks=args.num_model_chunks,
            microbatch_group_size=args.microbatch_group_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        )
        candidate = synth.build_strategy_schedule_table(args.policy, constraints)
        payload = {
            "valid": True,
            "policy": candidate.name,
            "rewrites": list(candidate.rewrites),
            "schedule_table": list(candidate.schedule_table),
        }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
