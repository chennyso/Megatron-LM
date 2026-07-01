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

    p2p = pd.read_csv(analysis_dir / "pairwise_p2p.csv")
    p2p = p2p[p2p["num_bytes"] == p2p["num_bytes"].max()]
    pivot = p2p.pivot(index="src_gpu", columns="dst_gpu", values="bandwidth_gbps")

    nccl = pd.read_csv(analysis_dir / "nccl_collectives.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(pivot.values, cmap="viridis")
    axes[0].set_title("P2P Bandwidth Heatmap")
    axes[0].set_xlabel("Dst GPU")
    axes[0].set_ylabel("Src GPU")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for idx, collective in enumerate(sorted(nccl["collective"].unique())):
        subset = nccl[nccl["collective"] == collective].sort_values(["gpu_count", "num_bytes"])
        grouped = subset.groupby("num_bytes")["algbw_gbps"].mean()
        axes[1].plot(grouped.index / (1024 ** 2), grouped.values, label=collective, color=COLORS[idx % len(COLORS)])
    axes[1].set_title("NCCL Bandwidth vs Message Size")
    axes[1].set_xlabel("Message Size (MiB)")
    axes[1].set_ylabel("AlgBW (GB/s)")
    axes[1].legend(frameon=False)

    save_fig(fig, fig_dir, "fig1_hardware_profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
