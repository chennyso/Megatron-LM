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
    df = df[df["case_id"].str.contains("rewrite")]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(df["case_id"], df["tokens_per_second_mean"], color=COLORS[5])
    ax.set_title("Manual Rewrite Throughput")
    ax.set_ylabel("Tokens/s")
    ax.tick_params(axis="x", rotation=75)

    save_fig(fig, fig_dir, "fig5_rewrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
