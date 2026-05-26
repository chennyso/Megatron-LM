#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import append_jsonl, dump_json, mean_std, parse_step_records, write_csv


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 06_parse_megatron_logs.py <run-dir>")
    run_dir = Path(sys.argv[1]).resolve()
    logs = []
    for path in sorted(run_dir.rglob("megatron_*.log")):
        logs.extend(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    rows = parse_step_records(logs)
    metrics_csv = run_dir / "metrics_megatron.csv"
    write_csv(metrics_csv, rows)
    append_jsonl(run_dir / "metrics_raw.jsonl", rows)

    tokens_mean, tokens_std = mean_std([r.get("tokens_per_second") for r in rows])
    step_mean, step_std = mean_std([r.get("elapsed_time_ms") for r in rows])
    tflops_mean, tflops_std = mean_std([r.get("tflops") for r in rows])
    mfu_mean, mfu_std = mean_std([r.get("mfu") for r in rows])

    summary = {
        "num_rows": len(rows),
        "step_time_ms_mean": step_mean,
        "step_time_ms_std": step_std,
        "tokens_per_second_mean": tokens_mean,
        "tokens_per_second_std": tokens_std,
        "tflops_mean": tflops_mean,
        "tflops_std": tflops_std,
        "mfu_mean": mfu_mean,
        "mfu_std": mfu_std,
        "oom": any("out of memory" in line.lower() for line in logs),
        "errors": [line for line in logs if "error" in line.lower()][:50],
        "skipped_reason": None,
    }
    dump_json(run_dir / "step_summary.json", summary)


if __name__ == "__main__":
    main()
