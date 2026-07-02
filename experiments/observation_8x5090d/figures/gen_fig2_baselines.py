#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, panel_label, rotate_labels, save_fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"
    df = pd.read_csv(analysis_dir / "figure_data" / "fig2_baselines.csv")
    df = df.sort_values(
        ["paper_model_id", "feasible", "tokens_per_second_mean_mean"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    model_ids = {model_id: idx for idx, model_id in enumerate(sorted(df["paper_model_id"].dropna().unique()))}
    colors = [COLORS[model_ids.get(model_id, 0) % len(COLORS)] for model_id in df["paper_model_id"]]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    ax = axes[0, 0]
    ax.bar(df["paper_label"], df["tokens_per_second_mean_mean"], color=colors)
    ax.errorbar(
        df["paper_label"],
        df["tokens_per_second_mean_mean"],
        yerr=df["tokens_per_second_mean_ci95_halfwidth"].fillna(0),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
    )
    ax.set_ylabel("Tokens/s")
    rotate_labels(ax, 35)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.bar(df["paper_label"], df["iter_time_ms_mean_mean"], color=colors)
    ax.errorbar(
        df["paper_label"],
        df["iter_time_ms_mean_mean"],
        yerr=df["iter_time_ms_mean_ci95_halfwidth"].fillna(0),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
    )
    ax.set_ylabel("Step time (ms)")
    rotate_labels(ax, 35)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.bar(df["paper_label"], df["peak_reserved_mb_mean_mean"] / 1024.0, color=colors)
    ax.errorbar(
        df["paper_label"],
        df["peak_reserved_mb_mean_mean"] / 1024.0,
        yerr=df["peak_reserved_mb_mean_ci95_halfwidth"].fillna(0) / 1024.0,
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2,
    )
    ax.set_ylabel("Peak reserved memory (GiB)")
    rotate_labels(ax, 35)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    status = df["feasible"].astype(int)
    ax.bar(df["paper_label"], status, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["error", "ok"])
    ax.set_ylabel("Feasibility")
    rotate_labels(ax, 35)
    panel_label(ax, "(d)")

    save_fig(fig, fig_dir, "fig2_baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
