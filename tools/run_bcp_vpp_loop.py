#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run the offline BCP-VPP strategy search and validation report loop.

This script expects traces from a short Megatron profiling run. It builds the
same artifacts used by a paper experiment: bottleneck proposals, candidate
search report, best StrategyPlan, BCP diagnostics, optional effective-overlap
analysis, and a manifest that records exact commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List


def _run(cmd: List[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"cmd": cmd}, ensure_ascii=False) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_common_shape_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--num-microbatches",
            str(args.num_microbatches),
            "--num-model-chunks",
            str(args.num_model_chunks),
            "--microbatch-group-size",
            str(args.microbatch_group_size),
            "--pipeline-parallel-size",
            str(args.pipeline_parallel_size),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True, help="Megatron trace JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory for loop artifacts")
    parser.add_argument("--num-microbatches", type=int, required=True)
    parser.add_argument("--num-model-chunks", type=int, required=True)
    parser.add_argument("--microbatch-group-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--candidate-budget", type=int, default=16)
    parser.add_argument("--objective", choices=["legacy", "bcp"], default="bcp")
    parser.add_argument("--runtime", choices=["fixed", "ready-set"], default="fixed")
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    parser.add_argument("--bcp-activation-budget-mb", type=float, default=None)
    parser.add_argument("--bcp-p2p-credit-budget", type=int, default=None)
    parser.add_argument("--bcp-fb-delay-budget", type=int, default=None)
    parser.add_argument(
        "--nsys-sqlite",
        nargs="+",
        default=[],
        help="Optional Nsight Systems SQLite exports for effective-overlap evidence",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="LLM mode is delegated to agentic_pipeline_optimizer.py contract when used separately.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "commands.jsonl"
    proposal_path = out_dir / "agent_proposals.json"
    best_path = out_dir / "best_strategy.json"
    report_path = out_dir / "candidate_report.json"
    bcp_analysis_path = out_dir / "bcp_trace_analysis.json"
    overlap_path = out_dir / "effective_overlap.json"
    manifest_path = out_dir / "manifest.json"

    proposal_cmd = [
        sys.executable,
        str(repo_root / "tools" / "pipeline_strategy_agent.py"),
        "--trace",
        *args.trace,
        "--output",
        str(proposal_path),
    ]
    _run(proposal_cmd, log_path=log_path)

    search_cmd = [
        sys.executable,
        str(repo_root / "tools" / "search_pipeline_strategy.py"),
        "--trace",
        *args.trace,
        "--output",
        str(best_path),
        "--report",
        str(report_path),
        "--proposal-json",
        str(proposal_path),
        "--candidate-budget",
        str(args.candidate_budget),
        "--objective",
        args.objective,
        "--runtime",
        args.runtime,
    ]
    _add_common_shape_args(search_cmd, args)
    if args.num_layers is not None:
        search_cmd.extend(["--num-layers", str(args.num_layers)])
    if args.memory_budget_mb is not None:
        search_cmd.extend(["--memory-budget-mb", str(args.memory_budget_mb)])
    if args.bcp_activation_budget_mb is not None:
        search_cmd.extend(["--bcp-activation-budget-mb", str(args.bcp_activation_budget_mb)])
    if args.bcp_p2p_credit_budget is not None:
        search_cmd.extend(["--bcp-p2p-credit-budget", str(args.bcp_p2p_credit_budget)])
    if args.bcp_fb_delay_budget is not None:
        search_cmd.extend(["--bcp-fb-delay-budget", str(args.bcp_fb_delay_budget)])
    _run(search_cmd, log_path=log_path)

    analysis_cmd = [
        sys.executable,
        str(repo_root / "tools" / "analyze_bcp_vpp_trace.py"),
        "--trace",
        *args.trace,
        "--output",
        str(bcp_analysis_path),
        "--num-model-chunks",
        str(args.num_model_chunks),
        "--microbatch-group-size",
        str(args.microbatch_group_size),
        "--runtime",
        args.runtime,
    ]
    if args.bcp_activation_budget_mb is not None:
        analysis_cmd.extend(["--bcp-activation-budget-mb", str(args.bcp_activation_budget_mb)])
    if args.bcp_p2p_credit_budget is not None:
        analysis_cmd.extend(["--bcp-p2p-credit-budget", str(args.bcp_p2p_credit_budget)])
    if args.bcp_fb_delay_budget is not None:
        analysis_cmd.extend(["--bcp-fb-delay-budget", str(args.bcp_fb_delay_budget)])
    _run(analysis_cmd, log_path=log_path)

    overlap_cmd = [
        sys.executable,
        str(repo_root / "tools" / "analyze_effective_overlap.py"),
        "--trace",
        *args.trace,
        "--output",
        str(overlap_path),
    ]
    if args.nsys_sqlite:
        overlap_cmd.extend(["--nsys-sqlite", *args.nsys_sqlite])
    _run(overlap_cmd, log_path=log_path)

    best = _load_json(best_path)
    report = _load_json(report_path)
    overlap = _load_json(overlap_path)
    manifest = {
        "artifact_version": 1,
        "method": "BCP-VPP",
        "inputs": {
            "trace": args.trace,
            "nsys_sqlite": args.nsys_sqlite,
            "num_microbatches": args.num_microbatches,
            "num_model_chunks": args.num_model_chunks,
            "microbatch_group_size": args.microbatch_group_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "num_layers": args.num_layers,
        },
        "budgets": {
            "memory_budget_mb": args.memory_budget_mb,
            "bcp_activation_budget_mb": args.bcp_activation_budget_mb,
            "bcp_p2p_credit_budget": args.bcp_p2p_credit_budget,
            "bcp_fb_delay_budget": args.bcp_fb_delay_budget,
        },
        "artifacts": {
            "proposals": str(proposal_path),
            "best_strategy": str(best_path),
            "candidate_report": str(report_path),
            "bcp_trace_analysis": str(bcp_analysis_path),
            "effective_overlap": str(overlap_path),
            "commands": str(log_path),
        },
        "best_summary": {
            "name": best.get("name"),
            "pipeline_layout": best.get("pipeline_layout"),
            "num_virtual_stages_per_pipeline_rank": best.get("num_virtual_stages_per_pipeline_rank"),
            "microbatch_group_size": best.get("microbatch_group_size"),
            "metadata": best.get("metadata", {}),
        },
        "search_summary": {
            "objective": report.get("objective"),
            "accepted": len(report.get("accepted", [])),
            "rejected": len(report.get("rejected", [])),
            "rewrite_algebra": report.get("rewrite_algebra", {}),
        },
        "overlap_summary": overlap.get("summary", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "best_strategy": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
