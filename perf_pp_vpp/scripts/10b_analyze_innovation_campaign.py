#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_runs(campaign_root: Path) -> list[dict]:
    rows: list[dict] = []
    runs_root = campaign_root / "runs"
    for step_path in sorted(runs_root.glob("*/seq*/repeat*/**/step_summary.json")):
        payload = read_json(step_path)
        overlap_leaf = step_path.parent.name
        if "__" in overlap_leaf:
            overlap_mode, run_tag = overlap_leaf.split("__", 1)
        else:
            overlap_mode, run_tag = overlap_leaf, ""
        rows.append(
            {
                "experiment": step_path.parts[-5],
                "seq_len": step_path.parts[-4].replace("seq", ""),
                "repeat": step_path.parts[-3].replace("repeat", ""),
                "overlap_mode": overlap_mode,
                "run_tag": run_tag,
                "step_time_ms_mean": payload.get("step_time_ms_mean"),
                "tokens_per_second_mean": payload.get("tokens_per_second_mean"),
                "tflops_mean": payload.get("tflops_mean"),
                "mfu_mean": payload.get("mfu_mean"),
                "oom": bool(payload.get("oom")),
                "num_rows": int(payload.get("num_rows", 0) or 0),
                "errors": payload.get("errors") or [],
                "path": str(step_path),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["experiment"], row["seq_len"], row["overlap_mode"], row["run_tag"])
        buckets.setdefault(key, []).append(row)
    out: list[dict] = []
    for key, items in sorted(buckets.items()):
        valid = [x["step_time_ms_mean"] for x in items if x["step_time_ms_mean"] is not None and not x["oom"]]
        out.append(
            {
                "experiment": key[0],
                "seq_len": key[1],
                "overlap_mode": key[2],
                "run_tag": key[3],
                "repeats": len(items),
                "valid_repeats": len(valid),
                "mean_step_time_ms": mean(valid) if valid else None,
                "oom_repeats": sum(1 for x in items if x["oom"]),
                "paths": [x["path"] for x in items],
            }
        )
    return out


def write_report(campaign_root: Path, summary_rows: list[dict]) -> None:
    report = campaign_root / "innovation_report.md"
    lines = [
        "# Qwen3-32B Overlap Innovation Report",
        "",
        f"- campaign: `{campaign_root.name}`",
        "",
        "| experiment | seq | overlap | tag | repeats | valid | oom | mean_step_time_ms |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['seq_len']} | {row['overlap_mode']} | {row['run_tag'] or '-'} | "
            f"{row['repeats']} | {row['valid_repeats']} | {row['oom_repeats']} | "
            f"{row['mean_step_time_ms'] if row['mean_step_time_ms'] is not None else 'n/a'} |"
        )
    best = [x for x in summary_rows if x["mean_step_time_ms"] is not None]
    if best:
        best.sort(key=lambda x: x["mean_step_time_ms"])
        lines += [
            "",
            "## Best Candidates",
            "",
        ]
        for row in best[:3]:
            lines.append(
                f"- `{row['experiment']}` / `{row['overlap_mode']}` / `{row['run_tag'] or '-'}`: "
                f"{row['mean_step_time_ms']:.2f} ms"
            )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    campaign_root = Path(os.environ["CAMPAIGN_ROOT"])
    rows = summarize_runs(campaign_root)
    summary_rows = aggregate(rows)
    (campaign_root / "innovation_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(campaign_root, summary_rows)


if __name__ == "__main__":
    main()
