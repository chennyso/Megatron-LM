#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, panel_label, rotate_labels, save_fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"
    pairs = pd.read_csv(analysis_dir / "table5_rewrite_pairs.csv")

    x = np.arange(len(pairs))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    ax = axes[0]
    ax.bar(x - width / 2, pairs["baseline_tokens_mean"], width=width, color=COLORS[0], label="Baseline")
    ax.bar(x + width / 2, pairs["rewrite_tokens_mean"], width=width, color=COLORS[1], label="Rewrite")
    ax.errorbar(x - width / 2, pairs["baseline_tokens_mean"], yerr=pairs["baseline_tokens_ci95"].fillna(0), fmt="none", ecolor="black", capsize=2)
    ax.errorbar(x + width / 2, pairs["rewrite_tokens_mean"], yerr=pairs["rewrite_tokens_ci95"].fillna(0), fmt="none", ecolor="black", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs["rewrite_label"])
    ax.set_ylabel("Tokens/s")
    ax.legend(frameon=False)
    rotate_labels(ax, 25)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.bar(x - width / 2, pairs["baseline_iter_ms_mean"], width=width, color=COLORS[2], label="Baseline")
    ax.bar(x + width / 2, pairs["rewrite_iter_ms_mean"], width=width, color=COLORS[3], label="Rewrite")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs["rewrite_label"])
    ax.set_ylabel("Step time (ms)")
    rotate_labels(ax, 25)
    panel_label(ax, "(b)")

    ax = axes[2]
    gain = (pairs["rewrite_tokens_mean"] - pairs["baseline_tokens_mean"]) / pairs["baseline_tokens_mean"] * 100.0
    ax.bar(pairs["rewrite_label"], gain, color=COLORS[4])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Throughput gain (%)")
    rotate_labels(ax, 25)
    panel_label(ax, "(c)")

    save_fig(fig, fig_dir, "fig5_rewrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
