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
    df = df[df["case_id"].str.contains("nsys")]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(df["case_id"], df["iter_time_mean_ms"], color=COLORS[4])
    ax.set_title("Representative NSYS Case Step Time")
    ax.set_ylabel("ms")
    ax.tick_params(axis="x", rotation=75)

    save_fig(fig, fig_dir, "fig4_nsys_breakdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
