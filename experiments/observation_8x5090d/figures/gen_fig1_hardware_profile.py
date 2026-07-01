#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from paper_plot_style import COLORS, panel_label, save_fig


TOPOLOGY_ORDER = ["X", "PIX", "PXB", "PHB", "NODE", "SYS", "NV1", "NV2", "NV4", "NV8"]
TOPOLOGY_TO_INT = {label: idx for idx, label in enumerate(TOPOLOGY_ORDER)}
TOPOLOGY_CMAP = ListedColormap(
    [
        "#f7f7f7",
        "#d9f0a3",
        "#addd8e",
        "#78c679",
        "#41ab5d",
        "#238443",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#2171b5",
    ]
)


def parse_topology_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    header = None
    matrix_rows: list[list[str]] = []
    labels: list[str] = []
    for line in rows:
        if line.startswith("GPU0") or line.startswith("GPU1"):
            header = line.split()
            continue
        tokens = line.split()
        if tokens and tokens[0].startswith("GPU"):
            labels.append(tokens[0])
            matrix_rows.append(tokens[1 : 1 + len(labels) + (len(header or []) - len(labels))])
    if header is None or not matrix_rows:
        raise SystemExit("Failed to parse nvidia-smi topo output.")
    width = len(header)
    normalized = [row[:width] for row in matrix_rows[:width]]
    numeric = np.array(
        [[TOPOLOGY_TO_INT.get(cell, 0) for cell in row] for row in normalized],
        dtype=float,
    )
    return header, numeric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"

    header, topo_matrix = parse_topology_matrix(analysis_dir / "nvidia-smi-topo.txt")
    p2p = pd.read_csv(analysis_dir / "pairwise_p2p_summary.csv")
    nccl = pd.read_csv(analysis_dir / "nccl_collectives_summary.csv")
    p2p_max = p2p[p2p["num_bytes"] == p2p["num_bytes"].max()]
    p2p_heatmap = p2p_max.pivot(index="src_gpu", columns="dst_gpu", values="bandwidth_gbps_mean")
    target_bytes = 64 * 1024 * 1024
    nccl_latency = nccl[nccl["num_bytes"] == target_bytes]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    ax = axes[0, 0]
    topo_im = ax.imshow(topo_matrix, cmap=TOPOLOGY_CMAP, vmin=0, vmax=len(TOPOLOGY_ORDER) - 1)
    ax.set_xticks(range(len(header)))
    ax.set_xticklabels(header)
    ax.set_yticks(range(len(header)))
    ax.set_yticklabels(header)
    for i in range(topo_matrix.shape[0]):
        for j in range(topo_matrix.shape[1]):
            label = TOPOLOGY_ORDER[int(topo_matrix[i, j])]
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    ax.set_xlabel("Peer GPU")
    ax.set_ylabel("Source GPU")
    panel_label(ax, "(a)")
    fig.colorbar(topo_im, ax=ax, fraction=0.046, pad=0.04, ticks=range(len(TOPOLOGY_ORDER)))

    ax = axes[0, 1]
    p2p_im = ax.imshow(p2p_heatmap.values, cmap="viridis")
    ax.set_xticks(range(len(p2p_heatmap.columns)))
    ax.set_xticklabels(p2p_heatmap.columns)
    ax.set_yticks(range(len(p2p_heatmap.index)))
    ax.set_yticklabels(p2p_heatmap.index)
    ax.set_xlabel("Destination GPU")
    ax.set_ylabel("Source GPU")
    panel_label(ax, "(b)")
    fig.colorbar(p2p_im, ax=ax, fraction=0.046, pad=0.04, label="GB/s")

    ax = axes[1, 0]
    for idx, collective in enumerate(sorted(nccl["collective"].unique())):
        subset = nccl[nccl["collective"] == collective].sort_values("num_bytes")
        x = subset["num_bytes"] / (1024 ** 2)
        y = subset["algbw_gbps_mean"]
        yerr = subset["algbw_gbps_ci95_halfwidth"].fillna(0)
        ax.plot(x, y, marker="o", label=collective, color=COLORS[idx % len(COLORS)])
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=COLORS[idx % len(COLORS)])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (MiB)")
    ax.set_ylabel("AlgBW (GB/s)")
    ax.legend(frameon=False, ncol=2)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    for idx, collective in enumerate(sorted(nccl_latency["collective"].unique())):
        subset = nccl_latency[nccl_latency["collective"] == collective].sort_values("gpu_count")
        x = subset["gpu_count"]
        y = subset["time_value_mean"]
        yerr = subset["time_value_ci95_halfwidth"].fillna(0)
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.2, label=collective, color=COLORS[idx % len(COLORS)])
    ax.set_xticks(sorted(nccl_latency["gpu_count"].unique()))
    ax.set_xlabel("GPU count")
    ax.set_ylabel("Collective latency")
    ax.legend(frameon=False, ncol=2)
    panel_label(ax, "(d)")

    save_fig(fig, fig_dir, "fig1_hardware_profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
