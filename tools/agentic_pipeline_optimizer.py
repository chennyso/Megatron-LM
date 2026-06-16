#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline agentic PP/VPP optimizer.

The LLM path is intentionally not in the training hot path. This CLI currently
uses the deterministic heuristic proposer and emits the same JSON artifacts a
future LLM proposer would consume and produce.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


LLM_SYSTEM_PROMPT = """You are the StrategyProposer agent for AgentPipe.
Return only JSON. Do not return code. Do not propose actions outside the schema.
The verifier and benchmark runner decide correctness and performance.
Allowed actions:
- change_schedule_order
- move_layer_boundary
- swap_stage_placement
- change_microbatch_group_size
- split_wgrad
- change_checkpoint_window
- enable_ready_set_dispatch
Each proposal must include action, reason, expected_effect, policy, target, and requires_runtime_support.
"""


def _chat_completions(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed: HTTP {exc.code}: {body[:500]}") from exc
    return data["choices"][0]["message"]["content"]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _llm_refine_proposals(proposal_path: Path, output_path: Path) -> Dict[str, Any]:
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    api_key = os.environ.get("LLM_API_KEY")
    if not base_url or not model or not api_key:
        raise RuntimeError("LLM mode requires LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY")

    proposal = _load_json(proposal_path)
    trace_summary = {
        "bottlenecks": proposal.get("bottlenecks", [])[:12],
        "heuristic_proposals": proposal.get("proposals", [])[:12],
    }
    user_prompt = {
        "task": "Refine PP/VPP strategy rewrite proposals from the trace summary.",
        "constraints": [
            "LLM proposals are candidates only.",
            "Do not propose out-of-order P2P unless requires_runtime_support is true.",
            "Prefer rewrites that can pass a deterministic Megatron verifier.",
            "Every reason must cite evidence from bottlenecks.",
        ],
        "input": trace_summary,
        "output_schema": {
            "hypothesis": "string",
            "proposals": [
                {
                    "action": "allowed action string",
                    "reason": "trace-backed reason",
                    "expected_effect": "expected trace/performance change",
                    "policy": "default or front-loaded",
                    "target": "object",
                    "requires_runtime_support": "boolean",
                }
            ],
            "risk": "string",
        },
    }
    content = _chat_completions(
        base_url,
        api_key,
        model,
        [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )
    refined = json.loads(content)
    merged = {
        "bottlenecks": proposal.get("bottlenecks", []),
        "proposals": refined.get("proposals", proposal.get("proposals", [])),
        "llm": {
            "base_url": base_url,
            "model": model,
            "hypothesis": refined.get("hypothesis", ""),
            "risk": refined.get("risk", ""),
        },
        "heuristic_proposals": proposal.get("proposals", []),
    }
    _write_json(output_path, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-microbatches", type=int, required=True)
    parser.add_argument("--num-model-chunks", type=int, required=True)
    parser.add_argument("--microbatch-group-size", type=int, required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--agent-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--candidate-budget", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = out_dir / "agent_proposals.json"
    best_path = out_dir / "best_strategy.json"
    report_path = out_dir / "candidate_report.json"
    agent_log_path = out_dir / "agent_log.jsonl"

    repo_root = Path(__file__).resolve().parents[1]
    proposal_cmd = [
        sys.executable,
        str(repo_root / "tools" / "pipeline_strategy_agent.py"),
        "--trace",
        *args.trace,
        "--output",
        str(proposal_path),
    ]
    search_cmd_base = [
        sys.executable,
        str(repo_root / "tools" / "search_pipeline_strategy.py"),
        "--trace",
        *args.trace,
        "--output",
        str(best_path),
        "--report",
        str(report_path),
        "--num-microbatches",
        str(args.num_microbatches),
        "--num-model-chunks",
        str(args.num_model_chunks),
        "--microbatch-group-size",
        str(args.microbatch_group_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--candidate-budget",
        str(args.candidate_budget),
    ]

    subprocess.run(proposal_cmd, check=True)
    if args.agent_mode == "llm":
        llm_proposal_path = out_dir / "llm_agent_proposals.json"
        _llm_refine_proposals(proposal_path, llm_proposal_path)
        proposal_path = llm_proposal_path

    if args.num_layers is not None:
        search_cmd_base.extend(["--num-layers", str(args.num_layers)])

    search_cmd = [*search_cmd_base, "--proposal-json", str(proposal_path)]
    subprocess.run(search_cmd, check=True)

    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with agent_log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "agent_mode": args.agent_mode,
                    "llm_base_url": os.environ.get("LLM_BASE_URL") if args.agent_mode == "llm" else None,
                    "llm_model": os.environ.get("LLM_MODEL") if args.agent_mode == "llm" else None,
                    "proposal_path": str(proposal_path),
                    "best_strategy_path": str(best_path),
                    "num_bottlenecks": len(proposal.get("bottlenecks", [])),
                    "num_candidates": len(report.get("accepted", [])),
                    "num_rejected": len(report.get("rejected", [])),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(json.dumps({"best_strategy": str(best_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
