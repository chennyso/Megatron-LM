#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import dump_json, load_yaml, mean_std, maybe_float, read_csv, resolved_experiment, write_csv


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "outputs"
    runs_root = root / "runs"
    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_key = {}
    for run_dir in sorted(runs_root.glob("*/*/*/*")):
        parts = run_dir.parts
        experiment = parts[-4]
        seq_len = int(parts[-3].replace("seq", ""))
        repeat = int(parts[-2].replace("repeat", ""))
        overlap_mode = parts[-1]
        step = {}
        if (run_dir / "step_summary.json").exists():
            step = json.loads((run_dir / "step_summary.json").read_text(encoding="utf-8"))
        nsys_rows = read_csv(run_dir / "nsys_overlap_summary.csv")
        boundary_rows = read_csv(run_dir / "nsys_boundary_comm.csv")
        cfg = {}
        if (run_dir / "config_resolved.yaml").exists():
            cfg = load_yaml(run_dir / "config_resolved.yaml")
        exp = cfg.get("experiment", {})
        row = {
            "experiment": experiment,
            "model": cfg.get("model", {}).get("model_name"),
            "seq_len": seq_len,
            "pp": exp.get("pp"),
            "tp": exp.get("tp"),
            "dp": exp.get("dp"),
            "vpp": exp.get("vpp"),
            "layers_per_virtual_stage": exp.get("layers_per_virtual_stage"),
            "overlap_mode": overlap_mode,
            "repeat": repeat,
            "step_time_ms_mean": step.get("step_time_ms_mean"),
            "step_time_ms_std": step.get("step_time_ms_std"),
            "tokens_per_second_mean": step.get("tokens_per_second_mean"),
            "tokens_per_second_std": step.get("tokens_per_second_std"),
            "peak_memory_gb": None,
            "bubble_ratio": None,
            "p2p_total_ms": None,
            "p2p_exposed_ms": None,
            "p2p_overlap_ratio": nsys_rows[0].get("p2p_overlap_ratio") if nsys_rows else None,
            "nccl_total_ms": None,
            "nccl_exposed_ms": nsys_rows[0].get("exposed_nccl_ms") if nsys_rows else None,
            "stage_imbalance": None,
            "cross_node_boundary_exposed_ms": boundary_rows[0].get("total_send_recv_time") if boundary_rows else None,
            "p2p_message_count": boundary_rows[0].get("send_count") if boundary_rows else None,
            "avg_p2p_message_size_mb": boundary_rows[0].get("avg_msg_size") if boundary_rows else None,
            "run_dir": str(run_dir),
        }
        all_rows.append(row)
        key = (experiment, seq_len, overlap_mode)
        by_key.setdefault(key, []).append(row)

    agg_rows = []
    for key, rows in sorted(by_key.items()):
        first = rows[0]
        step_mean, step_std = mean_std([maybe_float(x["step_time_ms_mean"]) for x in rows])
        tok_mean, tok_std = mean_std([maybe_float(x["tokens_per_second_mean"]) for x in rows])
        agg_rows.append({
            "experiment": first["experiment"],
            "model": first["model"],
            "seq_len": first["seq_len"],
            "pp": first["pp"],
            "tp": first["tp"],
            "dp": first["dp"],
            "vpp": first["vpp"],
            "layers_per_virtual_stage": first["layers_per_virtual_stage"],
            "overlap_mode": first["overlap_mode"],
            "repeat_count": len(rows),
            "step_time_ms_mean": step_mean,
            "step_time_ms_std": step_std,
            "tokens_per_second_mean": tok_mean,
            "tokens_per_second_std": tok_std,
            "peak_memory_gb": None,
            "bubble_ratio": None,
            "p2p_total_ms": None,
            "p2p_exposed_ms": None,
            "p2p_overlap_ratio": None,
            "nccl_total_ms": None,
            "nccl_exposed_ms": None,
            "stage_imbalance": None,
            "cross_node_boundary_exposed_ms": None,
            "p2p_message_count": None,
            "avg_p2p_message_size_mb": None,
        })

    skipped = []
    for yaml_name in ("phase1_qwen14b.yaml", "phase2_qwen32b.yaml"):
        experiment_yaml = Path(__file__).resolve().parents[1] / "configs" / "experiments" / yaml_name
        from _common import feasible_experiments
        res = feasible_experiments(experiment_yaml)
        for item in res["skipped"]:
            skipped.append({
                "source_yaml": yaml_name,
                "experiment": item["name"],
                "skip_reason": item["skip_reason"],
            })

    fastest = min((r for r in agg_rows if maybe_float(r["step_time_ms_mean"]) is not None), key=lambda r: maybe_float(r["step_time_ms_mean"]), default=None)
    slowest = max((r for r in agg_rows if maybe_float(r["step_time_ms_mean"]) is not None), key=lambda r: maybe_float(r["step_time_ms_mean"]), default=None)
    key_findings = {
        "fastest_config": fastest,
        "slowest_config": slowest,
        "notes": [
            "Null metrics indicate unavailable parser inputs and should be treated as unresolved rather than negative evidence.",
            "Qwen3-14B invalid VPP combinations are intentionally skipped when num_layers is not divisible by pp*vpp.",
        ],
    }

    write_csv(summary_root / "all_experiments.csv", all_rows)
    write_csv(summary_root / "by_experiment_mean_std.csv", agg_rows)
    write_csv(summary_root / "skipped_experiments.csv", skipped)
    dump_json(summary_root / "key_findings.json", key_findings)


if __name__ == "__main__":
    main()
