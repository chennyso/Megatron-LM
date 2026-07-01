#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stats_utils import summarize_numeric


METRIC_FIELDS = [
    ("iter_time_ms", "mean"),
    ("tokens_per_second", "mean"),
    ("samples_per_second", "mean"),
    ("grad_norm", "mean"),
    ("learning_rate", "mean"),
    ("throughput_per_gpu_tflops", "mean"),
    ("memory", "peak_allocated_mb", "mean"),
    ("memory", "peak_reserved_mb", "mean"),
    ("dmon", "gpu_util", "mean"),
    ("dmon", "power_w", "mean"),
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def nested_get(payload: dict, path: tuple[str, ...]) -> float | str | int | list | dict | None:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def flatten_summary(summary_path: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    repeat_dir = summary_path.parent
    case_dir = repeat_dir.parent
    case_config_path = repeat_dir / "case_config.json"
    case_config = json.loads(case_config_path.read_text(encoding="utf-8")) if case_config_path.exists() else {}

    row = {
        "case_id": payload["case_id"],
        "phase": payload.get("phase"),
        "paper_model_id": payload.get("paper_model_id"),
        "paper_label": case_config.get("paper_label"),
        "dataset_spec": payload.get("dataset_spec"),
        "repeat_index": payload.get("repeat_index"),
        "case_dir": str(case_dir),
        "repeat_dir": str(repeat_dir),
        "figure_membership": ",".join(payload.get("figure_membership", [])),
        "claim_membership": ",".join(payload.get("claim_membership", [])),
        "oom_or_runtime_error": payload.get("oom_or_runtime_error"),
        "steady_state_step_count": payload.get("steady_state_step_count"),
        "loss_last": nested_get(payload, ("loss", "last")),
        "loss_steady_mean": nested_get(payload, ("loss", "steady_state", "mean")),
        "rank_skew_percent": nested_get(payload, ("memory", "rank_skew_percent")),
        "baseline_case_id": case_config.get("baseline_case_id"),
        "rewrite_type": case_config.get("rewrite_type"),
    }
    for field_path in METRIC_FIELDS:
        if len(field_path) == 2:
            metric_name, stat_name = field_path
            value = nested_get(payload, (metric_name, stat_name))
            row[f"{metric_name}_{stat_name}"] = value
        else:
            parent, metric_name, stat_name = field_path
            value = nested_get(payload, (parent, metric_name, stat_name))
            row[f"{metric_name}_{stat_name}"] = value
    return row


def collect_run_rows(result_root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(result_root.rglob("summary.json")):
        if "hardware" in summary_path.parts:
            continue
        rows.append(flatten_summary(summary_path))
    return rows


def aggregate_case_rows(run_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in run_rows:
        grouped.setdefault(row["case_id"], []).append(row)

    aggregates: list[dict] = []
    for case_id, rows in sorted(grouped.items()):
        first = rows[0]
        aggregate = {
            "case_id": case_id,
            "phase": first["phase"],
            "paper_model_id": first["paper_model_id"],
            "paper_label": first["paper_label"],
            "dataset_spec": first["dataset_spec"],
            "repeat_count": len(rows),
            "oom_or_runtime_error_any": any(bool(row["oom_or_runtime_error"]) for row in rows),
            "figure_membership": first["figure_membership"],
            "claim_membership": first["claim_membership"],
            "baseline_case_id": first["baseline_case_id"],
            "rewrite_type": first["rewrite_type"],
        }
        for metric_key in [
            "iter_time_ms_mean",
            "tokens_per_second_mean",
            "samples_per_second_mean",
            "grad_norm_mean",
            "learning_rate_mean",
            "throughput_per_gpu_tflops_mean",
            "peak_allocated_mb_mean",
            "peak_reserved_mb_mean",
            "gpu_util_mean",
            "power_w_mean",
            "loss_steady_mean",
            "rank_skew_percent",
        ]:
            summary = summarize_numeric(row.get(metric_key) for row in rows)
            for stat_name, stat_value in summary.items():
                aggregate[f"{metric_key}_{stat_name}"] = stat_value
        aggregates.append(aggregate)
    return aggregates


def build_feasibility_matrix(case_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for case in case_rows:
        rows.append(
            {
                "case_id": case["case_id"],
                "paper_label": case["paper_label"],
                "phase": case["phase"],
                "paper_model_id": case["paper_model_id"],
                "dataset_spec": case["dataset_spec"],
                "repeat_count": case["repeat_count"],
                "status": "error" if case["oom_or_runtime_error_any"] else "ok",
            }
        )
    return rows


def build_rewrite_pairs(case_rows: list[dict]) -> list[dict]:
    by_case = {row["case_id"]: row for row in case_rows}
    pairs: list[dict] = []
    for row in case_rows:
        baseline_case_id = row.get("baseline_case_id")
        if not baseline_case_id:
            continue
        baseline = by_case.get(baseline_case_id)
        if baseline is None:
            continue
        pairs.append(
            {
                "rewrite_case_id": row["case_id"],
                "rewrite_label": row["paper_label"],
                "baseline_case_id": baseline_case_id,
                "baseline_label": baseline["paper_label"],
                "rewrite_type": row["rewrite_type"],
                "baseline_tokens_mean": baseline.get("tokens_per_second_mean_mean"),
                "baseline_tokens_ci95": baseline.get("tokens_per_second_mean_ci95_halfwidth"),
                "rewrite_tokens_mean": row.get("tokens_per_second_mean_mean"),
                "rewrite_tokens_ci95": row.get("tokens_per_second_mean_ci95_halfwidth"),
                "baseline_iter_ms_mean": baseline.get("iter_time_ms_mean_mean"),
                "rewrite_iter_ms_mean": row.get("iter_time_ms_mean_mean"),
            }
        )
    return pairs


def build_case_manifest(result_root: Path, output_dir: Path) -> list[dict]:
    manifest: list[dict] = []
    for case_config_path in sorted(result_root.rglob("case_config.json")):
        if "repeat_" not in case_config_path.parent.name:
            continue
        case_config = json.loads(case_config_path.read_text(encoding="utf-8"))
        repeat_dir = case_config_path.parent
        entry = {
            "case_id": case_config["id"],
            "repeat_index": case_config.get("repeat_index"),
            "phase": case_config.get("phase"),
            "paper_model_id": case_config.get("paper_model_id"),
            "repeat_dir": str(repeat_dir),
            "summary_json": str(repeat_dir / "summary.json"),
            "per_step_csv": str(repeat_dir / "per_step.csv"),
            "rank_memory_csv": str(repeat_dir / "rank_memory.csv"),
            "gpu_dmon_samples_csv": str(repeat_dir / "gpu_dmon_samples.csv"),
        }
        manifest.append(entry)
    manifest_path = output_dir / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def copy_hardware_artifacts(result_root: Path, output_dir: Path) -> None:
    hardware_root = result_root / "hardware"
    if not hardware_root.exists():
        return
    for name in [
        "pairwise_p2p_raw.csv",
        "pairwise_p2p_summary.csv",
        "nccl_collectives_raw.csv",
        "nccl_collectives_summary.csv",
        "nvidia-smi-query.csv",
        "hardware_summary.json",
        "nvidia-smi-topo.txt",
        "nvidia-smi-nvlink.txt",
        "lscpu.txt",
        "numactl-hardware.txt"
    ]:
        maybe_copy(hardware_root / name, output_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = collect_run_rows(result_root)
    if run_rows:
        write_csv(output_dir / "run_summaries.csv", run_rows, list(run_rows[0].keys()))
    case_rows = aggregate_case_rows(run_rows)
    if case_rows:
        write_csv(output_dir / "case_aggregates.csv", case_rows, list(case_rows[0].keys()))
        feasibility_rows = build_feasibility_matrix(case_rows)
        write_csv(output_dir / "table2_feasibility_matrix.csv", feasibility_rows, list(feasibility_rows[0].keys()))
        rewrite_rows = build_rewrite_pairs(case_rows)
        if rewrite_rows:
            write_csv(output_dir / "table5_rewrite_pairs.csv", rewrite_rows, list(rewrite_rows[0].keys()))

        figure_dir = output_dir / "figure_data"
        baseline_rows = [row for row in case_rows if row["phase"] == "baseline"]
        if baseline_rows:
            write_csv(figure_dir / "fig2_baselines.csv", baseline_rows, list(baseline_rows[0].keys()))
        nsys_rows = [row for row in case_rows if row["phase"] == "nsys"]
        if nsys_rows:
            write_csv(figure_dir / "fig4_nsys_cases.csv", nsys_rows, list(nsys_rows[0].keys()))
        rewrite_case_rows = [row for row in case_rows if row["phase"] == "rewrite"]
        if rewrite_case_rows:
            write_csv(figure_dir / "fig5_rewrites.csv", rewrite_case_rows, list(rewrite_case_rows[0].keys()))

    build_case_manifest(result_root, output_dir)
    copy_hardware_artifacts(result_root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
