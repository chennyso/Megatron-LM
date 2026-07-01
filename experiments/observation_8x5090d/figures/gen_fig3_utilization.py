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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df["case_id"], df["gpu_util_mean"], color=COLORS[2])
    axes[0].set_title("Mean GPU Utilization")
    axes[0].set_ylabel("Utilization")
    axes[0].tick_params(axis="x", rotation=75)

    axes[1].bar(df["case_id"], df["power_mean_w"], color=COLORS[3])
    axes[1].set_title("Mean Power")
    axes[1].set_ylabel("Watts")
    axes[1].tick_params(axis="x", rotation=75)

    save_fig(fig, fig_dir, "fig3_utilization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
