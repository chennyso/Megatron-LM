#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_case_summaries(result_root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in result_root.rglob("summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "case_id": payload["case_id"],
            "log_path": payload["log_path"],
            "iter_time_mean_ms": payload["iter_time_ms"]["mean"],
            "iter_time_p50_ms": payload["iter_time_ms"]["p50"],
            "iter_time_p95_ms": payload["iter_time_ms"]["p95"],
            "tokens_per_second_mean": payload["tokens_per_second"]["mean"],
            "tokens_per_second_p50": payload["tokens_per_second"]["p50"],
            "tokens_per_second_p95": payload["tokens_per_second"]["p95"],
            "throughput_per_gpu_tflops_mean": payload["throughput_per_gpu_tflops"]["mean"],
            "loss_last": payload["loss"]["last"],
            "oom_or_runtime_error": payload["oom_or_runtime_error"],
        }
        dmon = payload.get("dmon", {})
        row["gpu_util_mean"] = dmon.get("gpu_util_mean")
        row["power_mean_w"] = dmon.get("power_mean_w")
        row["mem_metric_mean"] = dmon.get("mem_clock_or_mem_metric_mean")
        rows.append(row)
    return sorted(rows, key=lambda item: item["case_id"])


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = collect_case_summaries(result_root)
    (output_dir / "case_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if summaries:
        with (output_dir / "case_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    copy_if_exists(result_root / "hardware" / "pairwise_p2p.csv", output_dir / "pairwise_p2p.csv")
    copy_if_exists(result_root / "hardware" / "nccl_collectives.csv", output_dir / "nccl_collectives.csv")
    copy_if_exists(result_root / "hardware" / "nvidia-smi-query.csv", output_dir / "nvidia-smi-query.csv")
    copy_if_exists(result_root / "hardware" / "hardware_summary.json", output_dir / "hardware_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
