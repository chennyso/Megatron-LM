#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def case_dimensions(case_id: str) -> tuple[int, bool]:
    if case_id == "clean_1f1b_nosp":
        return 1, False
    if case_id == "clean_1f1b_sp":
        return 1, True
    match = re.fullmatch(r"clean_vpp(\d+)_(nosp|sp)", case_id)
    if match is None:
        raise ValueError(f"unsupported clean screening case: {case_id}")
    return int(match.group(1)), match.group(2) == "sp"


def read_repeat_summaries(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                vpp_size, sequence_parallel = case_dimensions(row["case_id"])
                key = (row["case_id"], int(row["repeat_id"]))
                if key in seen:
                    raise ValueError(f"duplicate case/repeat row: {key}")
                seen.add(key)
                rows.append(
                    {
                        "case_id": row["case_id"],
                        "repeat_id": int(row["repeat_id"]),
                        "vpp_size": vpp_size,
                        "transition_budget": 2 * vpp_size - 1,
                        "sequence_parallel": sequence_parallel,
                        "median_iter_ms": float(row["median_iter_ms"]),
                        "mean_iter_ms": float(row["mean_iter_ms"]),
                        "within_run_cv_percent": float(row["within_run_cv_percent"]),
                        "median_tokens_per_second": float(row["median_tokens_per_second"]),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    baseline = next(
        (row for row in rows if row["case_id"] == "clean_1f1b_nosp"), None
    )
    if baseline is None:
        raise ValueError("clean_1f1b_nosp is required as the speedup baseline")

    summary = []
    for row in sorted(rows, key=lambda item: (item["vpp_size"], item["sequence_parallel"])):
        item = dict(row)
        item["speedup_vs_1f1b_nosp"] = baseline["median_iter_ms"] / row["median_iter_ms"]
        item["throughput_gain_vs_1f1b_nosp_percent"] = (
            item["speedup_vs_1f1b_nosp"] - 1.0
        ) * 100.0
        summary.append(item)

    by_vpp: dict[int, dict[bool, dict]] = {}
    for row in rows:
        by_vpp.setdefault(row["vpp_size"], {})[row["sequence_parallel"]] = row
    crossover = []
    for vpp_size in sorted(by_vpp):
        variants = by_vpp[vpp_size]
        if set(variants) != {False, True}:
            continue
        no_sp = variants[False]
        sp = variants[True]
        gain = (no_sp["median_iter_ms"] / sp["median_iter_ms"] - 1.0) * 100.0
        crossover.append(
            {
                "vpp_size": vpp_size,
                "transition_budget": 2 * vpp_size - 1,
                "nosp_median_iter_ms": no_sp["median_iter_ms"],
                "sp_median_iter_ms": sp["median_iter_ms"],
                "sp_throughput_gain_percent": gain,
                "winner": "sp" if gain > 0 else "nosp",
            }
        )
    return summary, crossover


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_repeat_summaries(args.repeat_summary)
    summary, crossover = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "screening_summary.csv", summary)
    write_csv(args.output_dir / "sp_crossover.csv", crossover)
    manifest = {
        "repeat_summaries": [str(path) for path in args.repeat_summary],
        "row_count": len(summary),
        "sp_pair_count": len(crossover),
        "best_case": min(summary, key=lambda item: item["median_iter_ms"])["case_id"],
    }
    (args.output_dir / "screening_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
