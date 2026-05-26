#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path


LINE_RE = re.compile(
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+) \| throughput per GPU \(TFLOP/s/GPU\): (?P<tflops>[0-9.]+)"
)


def parse_metrics(run_dir: Path) -> dict:
    log_candidates = sorted(run_dir.glob("remote_artifacts/megatron_stdout_node*.log"))
    values = []
    for path in log_candidates:
        for line in path.read_text(errors="ignore").splitlines():
            match = LINE_RE.search(line)
            if match:
                values.append(
                    {
                        "ms": float(match.group("ms")),
                        "tflops": float(match.group("tflops")),
                    }
                )
    env_path = run_dir / "env.txt"
    env_text = env_path.read_text(encoding="utf-8", errors="ignore") if env_path.exists() else ""
    layout = ""
    for line in env_text.splitlines():
        if line.startswith("PERF_PIPELINE_MODEL_PARALLEL_LAYOUT="):
            layout = line.split("=", 1)[1]
            break
    if not values:
        return {
            "run_dir": str(run_dir),
            "status": "no-metrics",
            "layout": layout,
        }
    warm_values = values[2:] if len(values) > 2 else values
    return {
        "run_dir": str(run_dir),
        "status": "ok",
        "layout": layout,
        "num_iters": len(values),
        "step_time_ms_mean": statistics.mean(v["ms"] for v in warm_values),
        "tflops_per_gpu_mean": statistics.mean(v["tflops"] for v in warm_values),
    }


def main() -> None:
    campaign_root = Path(os.environ["CAMPAIGN_ROOT"]).resolve()
    rows = []
    for run_dir in sorted(campaign_root.glob("runs/*/seq4096/repeat*/baseline*")):
        row = parse_metrics(run_dir)
        row["run_name"] = run_dir.parents[2].name
        row["repeat_id"] = run_dir.parents[0].name
        row["leaf"] = run_dir.name
        rows.append(row)
    rows.sort(key=lambda x: (x["status"] != "ok", x.get("step_time_ms_mean", 1e18)))
    (campaign_root / "topology_layout_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    lines = ["# Topology-Aware Layout Campaign", ""]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"- {row['leaf']}: no metrics")
            continue
        lines.append(
            f"- {row['leaf']}: {row['step_time_ms_mean']:.1f} ms/iter, "
            f"{row['tflops_per_gpu_mean']:.1f} TFLOP/s/GPU"
        )
    (campaign_root / "topology_layout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
