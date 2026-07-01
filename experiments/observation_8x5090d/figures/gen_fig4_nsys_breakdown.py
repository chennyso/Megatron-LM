#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, panel_label, rotate_labels, save_fig


def collect_kernel_breakdown(manifest: list[dict], case_id: str) -> dict[str, float]:
    for entry in manifest:
        if entry["case_id"] != case_id:
            continue
        repeat_dir = Path(entry["repeat_dir"])
        kernel_path = repeat_dir / "nsys_stats_gpukernsum.csv"
        if not kernel_path.exists():
            continue
        df = pd.read_csv(kernel_path, skiprows=2)
        name_col = next((column for column in df.columns if "Name" in column), None)
        time_col = next((column for column in df.columns if "Total Time" in column or "Time" in column), None)
        if name_col is None or time_col is None:
            continue
        totals = {"nccl": 0.0, "compute": 0.0, "other": 0.0}
        for _, row in df.iterrows():
            name = str(row[name_col]).lower()
            try:
                value = float(row[time_col])
            except (TypeError, ValueError):
                continue
            if "nccl" in name:
                totals["nccl"] += value
            elif "gemm" in name or "attention" in name or "softmax" in name:
                totals["compute"] += value
            else:
                totals["other"] += value
        return totals
    return {"nccl": 0.0, "compute": 0.0, "other": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"
    df = pd.read_csv(analysis_dir / "figure_data" / "fig4_nsys_cases.csv")
    manifest = json.loads((analysis_dir / "case_manifest.json").read_text(encoding="utf-8"))

    breakdown_rows = []
    for _, row in df.iterrows():
        totals = collect_kernel_breakdown(manifest, row["case_id"])
        total_time = sum(totals.values()) or 1.0
        breakdown_rows.append(
            {
                "paper_label": row["paper_label"],
                "iter_time_ms": row["iter_time_ms_mean_mean"],
                "compute_share": totals["compute"] / total_time,
                "nccl_share": totals["nccl"] / total_time,
                "other_share": totals["other"] / total_time,
            }
        )
    breakdown = pd.DataFrame(breakdown_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.bar(df["paper_label"], df["iter_time_ms_mean_mean"], color=COLORS[0])
    ax.errorbar(
        df["paper_label"],
        df["iter_time_ms_mean_mean"],
        yerr=df["iter_time_ms_mean_ci95_halfwidth"].fillna(0),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
    )
    ax.set_ylabel("Profiled step time (ms)")
    rotate_labels(ax, 25)
    panel_label(ax, "(a)")

    ax = axes[1]
    bottom = pd.Series([0.0] * len(breakdown))
    for color, column, label in [
        (COLORS[1], "compute_share", "Compute"),
        (COLORS[2], "nccl_share", "NCCL"),
        (COLORS[3], "other_share", "Other"),
    ]:
        ax.bar(breakdown["paper_label"], breakdown[column], bottom=bottom, color=color, label=label)
        bottom += breakdown[column]
    ax.set_ylabel("Kernel time share")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    rotate_labels(ax, 25)
    panel_label(ax, "(b)")

    save_fig(fig, fig_dir, "fig4_nsys_breakdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
