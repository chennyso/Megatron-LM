#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from paper_plot_style import COLORS, save_fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"

    df = pd.read_csv(analysis_dir / "case_summaries.csv")
    baseline = df[~df["case_id"].str.contains("nsys|rewrite", regex=True)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(baseline["case_id"], baseline["tokens_per_second_mean"], color=COLORS[0])
    axes[0].set_title("Tokens/s Across Baselines")
    axes[0].set_ylabel("Tokens/s")
    axes[0].tick_params(axis="x", rotation=75)

    axes[1].bar(baseline["case_id"], baseline["iter_time_mean_ms"], color=COLORS[1])
    axes[1].set_title("Step Time Mean")
    axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", rotation=75)

    save_fig(fig, fig_dir, "fig2_baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
