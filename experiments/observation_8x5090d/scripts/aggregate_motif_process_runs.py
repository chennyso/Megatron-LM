#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from stats_utils import summarize_numeric


LARGE_MESSAGE_BYTES = 16 * 1024 * 1024
COMPUTE_METRICS = (
    "isolated_comm_ms",
    "isolated_compute_ms",
    "concurrent_ms",
    "contention_factor",
    "overlap_efficiency",
    "exposed_overhead_ms",
    "comm_launch_skew_us",
    "compute_launch_skew_us",
    "concurrent_launch_skew_us",
    "compute_reference_max_abs_error",
    "compute_reference_max_rel_error",
)
MOTIF_METRICS = (
    "wall_makespan_ms",
    "max_device_ms",
    "launch_skew_us",
    "aggregate_decimal_gbps",
    "slowdown_vs_isolated_max",
    "parallel_efficiency",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: object) -> bool:
    return str(value).lower() == "true"


def motif_launch_skew_fraction(row: dict) -> float:
    return float(row["launch_skew_us"]) / (float(row["wall_makespan_ms"]) * 1000.0)


def compute_launch_skew_fraction(row: dict) -> float:
    mode_fields = (
        ("comm_launch_skew_us", "isolated_comm_ms"),
        ("compute_launch_skew_us", "isolated_compute_ms"),
        ("concurrent_launch_skew_us", "concurrent_ms"),
    )
    return max(float(row[skew]) / (float(row[duration]) * 1000.0) for skew, duration in mode_fields)


def summarize_grouped(
    rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)

    summaries: list[dict] = []
    for key_values, bucket in sorted(grouped.items()):
        summary = dict(zip(keys, key_values))
        summary["sample_count"] = len(bucket)
        summary["process_count"] = len({row["process_run_id"] for row in bucket})
        for metric in metrics:
            stats = summarize_numeric(
                None if row.get(metric) in {None, ""} else float(row[metric])
                for row in bucket
            )
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summaries.append(summary)
    return summaries


def large_message_category(row: dict) -> str | None:
    if row["route_class"] == "endpoint_disjoint":
        return "endpoint_disjoint_oneway"
    if row["route_class"] == "endpoint_disjoint_bidirectional":
        return "endpoint_disjoint_bidirectional"
    if row["motif_id"] in {"shared_sink_2", "shared_sink_3", "shared_sink_4"}:
        return row["motif_id"]
    return None


def build_large_message_run_rows(motif_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in motif_rows:
        if int(row["size_bytes"]) < LARGE_MESSAGE_BYTES:
            continue
        category = large_message_category(row)
        if category is not None:
            grouped[(row["process_run_id"], category)].append(row)

    output: list[dict] = []
    for (run_id, category), bucket in sorted(grouped.items()):
        quality = [row for row in bucket if is_true(row["launch_quality_ok"])]
        output.append(
            {
                "process_run_id": run_id,
                "category": category,
                "sample_count_all": len(bucket),
                "sample_count_quality": len(quality),
                "slowdown_mean_all": statistics.mean(
                    float(row["slowdown_vs_isolated_max"]) for row in bucket
                ),
                "slowdown_mean_quality": statistics.mean(
                    float(row["slowdown_vs_isolated_max"]) for row in quality
                )
                if quality
                else None,
            }
        )
    return output


def collect_run(
    run_dir: Path, max_launch_skew_fraction: float
) -> tuple[dict, list[dict], list[dict]]:
    motif_dir = run_dir / "motif" if (run_dir / "motif").is_dir() else run_dir
    compute_dir = motif_dir / "compute_comm"
    required = (
        motif_dir / "DONE",
        compute_dir / "DONE",
        motif_dir / "motif_raw.csv",
        compute_dir / "compute_comm_raw.csv",
        motif_dir / "motif_metadata.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete run {run_dir}: missing {missing}")

    run_id = run_dir.name
    motif_rows = read_csv(motif_dir / "motif_raw.csv")
    compute_rows = read_csv(compute_dir / "compute_comm_raw.csv")
    motif_keys = [(row["motif_id"], row["size_bytes"], row["repeat"]) for row in motif_rows]
    compute_keys = [
        (row["case_id"], row["compute_location"], row["repeat"]) for row in compute_rows
    ]
    if len(motif_keys) != len(set(motif_keys)) or len(compute_keys) != len(set(compute_keys)):
        raise RuntimeError(f"duplicate measurement keys in {run_id}")
    if not all(is_true(row["payload_valid"]) for row in motif_rows + compute_rows):
        raise RuntimeError(f"payload validation failed in {run_id}")
    if not all(is_true(row["compute_valid"]) for row in compute_rows):
        raise RuntimeError(f"compute validation failed in {run_id}")

    for row in motif_rows:
        row["process_run_id"] = run_id
        row["launch_skew_fraction"] = motif_launch_skew_fraction(row)
        row["launch_quality_ok"] = row["launch_skew_fraction"] <= max_launch_skew_fraction
    for row in compute_rows:
        row["process_run_id"] = run_id
        row["launch_skew_fraction"] = compute_launch_skew_fraction(row)
        row["launch_quality_ok"] = row["launch_skew_fraction"] <= max_launch_skew_fraction

    metadata = json.loads((motif_dir / "motif_metadata.json").read_text(encoding="utf-8"))
    manifest = {
        "process_run_id": run_id,
        "git_commit": metadata.get("git_commit"),
        "hostname": metadata.get("hostname"),
        "topology_sha256": metadata.get("topology_sha256"),
        "motif_row_count": len(motif_rows),
        "compute_row_count": len(compute_rows),
        "motif_quality_failures": sum(not is_true(row["launch_quality_ok"]) for row in motif_rows),
        "compute_quality_failures": sum(
            not is_true(row["launch_quality_ok"]) for row in compute_rows
        ),
        "motif_max_launch_skew_fraction": max(
            float(row["launch_skew_fraction"]) for row in motif_rows
        ),
        "compute_max_launch_skew_fraction": max(
            float(row["launch_skew_fraction"]) for row in compute_rows
        ),
    }
    return manifest, motif_rows, compute_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-launch-skew-fraction", type=float, default=0.01)
    args = parser.parse_args()

    manifests: list[dict] = []
    motif_rows: list[dict] = []
    compute_rows: list[dict] = []
    expected_motif_keys: set[tuple[str, str]] | None = None
    expected_compute_keys: set[tuple[str, str]] | None = None
    for run_dir in args.run_dir:
        manifest, run_motif_rows, run_compute_rows = collect_run(
            run_dir, args.max_launch_skew_fraction
        )
        motif_keys = {(row["motif_id"], row["size_bytes"]) for row in run_motif_rows}
        compute_keys = {(row["case_id"], row["compute_location"]) for row in run_compute_rows}
        if expected_motif_keys is not None and motif_keys != expected_motif_keys:
            raise RuntimeError(f"motif key set differs in {run_dir}")
        if expected_compute_keys is not None and compute_keys != expected_compute_keys:
            raise RuntimeError(f"compute key set differs in {run_dir}")
        expected_motif_keys = motif_keys
        expected_compute_keys = compute_keys
        manifests.append(manifest)
        motif_rows.extend(run_motif_rows)
        compute_rows.extend(run_compute_rows)

    quality_motif_rows = [row for row in motif_rows if is_true(row["launch_quality_ok"])]
    quality_compute_rows = [row for row in compute_rows if is_true(row["launch_quality_ok"])]
    large_run_rows = build_large_message_run_rows(motif_rows)
    large_summary = summarize_grouped(
        large_run_rows,
        ("category",),
        ("slowdown_mean_all", "slowdown_mean_quality"),
    )

    output_dir = args.output_dir
    write_csv(output_dir / "run_manifest.csv", manifests)
    write_csv(output_dir / "motif_process_raw.csv", motif_rows)
    write_csv(output_dir / "compute_process_raw.csv", compute_rows)
    write_csv(
        output_dir / "motif_process_summary_all.csv",
        summarize_grouped(motif_rows, ("motif_id", "size_bytes"), MOTIF_METRICS),
    )
    write_csv(
        output_dir / "motif_process_summary_quality.csv",
        summarize_grouped(
            quality_motif_rows, ("motif_id", "size_bytes"), MOTIF_METRICS
        ),
    )
    write_csv(
        output_dir / "compute_process_summary_all.csv",
        summarize_grouped(
            compute_rows, ("case_id", "compute_location"), COMPUTE_METRICS
        ),
    )
    write_csv(
        output_dir / "compute_process_summary_quality.csv",
        summarize_grouped(
            quality_compute_rows, ("case_id", "compute_location"), COMPUTE_METRICS
        ),
    )
    write_csv(output_dir / "large_message_run_summary.csv", large_run_rows)
    write_csv(output_dir / "large_message_process_summary.csv", large_summary)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "process_run_count": len(manifests),
                "max_launch_skew_fraction": args.max_launch_skew_fraction,
                "large_message_min_bytes": LARGE_MESSAGE_BYTES,
                "student_t_ci": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
