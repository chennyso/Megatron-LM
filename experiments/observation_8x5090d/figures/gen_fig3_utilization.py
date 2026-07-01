#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, panel_label, save_fig


def pick_case(analysis_dir: Path) -> tuple[dict, Path]:
    case_aggregates = pd.read_csv(analysis_dir / "case_aggregates.csv")
    candidates = case_aggregates[case_aggregates["figure_membership"].fillna("").str.contains("fig3")]
    if candidates.empty:
        raise SystemExit("No case with fig3 membership found.")
    selected = candidates.sort_values("tokens_per_second_mean_mean", ascending=False).iloc[0].to_dict()
    manifest = json.loads((analysis_dir / "case_manifest.json").read_text(encoding="utf-8"))
    repeat_dir = None
    for entry in manifest:
        if entry["case_id"] == selected["case_id"]:
            repeat_dir = Path(entry["repeat_dir"])
            break
    if repeat_dir is None:
        raise SystemExit(f"Missing repeat artifact for case {selected['case_id']}.")
    return selected, repeat_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"
    selected, repeat_dir = pick_case(analysis_dir)
    per_step = pd.read_csv(repeat_dir / "per_step.csv")
    rank_memory = pd.read_csv(repeat_dir / "rank_memory.csv") if (repeat_dir / "rank_memory.csv").exists() else pd.DataFrame()
    dmon = pd.read_csv(repeat_dir / "gpu_dmon_samples.csv") if (repeat_dir / "gpu_dmon_samples.csv").exists() else pd.DataFrame()

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0))

    ax = axes[0, 0]
    ax.plot(per_step["iteration"], per_step["iter_time_ms"], color=COLORS[0], linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step time (ms)")
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(per_step["iteration"], per_step["tokens_per_second"], color=COLORS[1], linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tokens/s")
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.plot(per_step["iteration"], per_step["loss"], color=COLORS[2], linewidth=1.4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    if not dmon.empty:
        gpu_util = dmon.groupby("sample_index")["gpu_util"].mean()
        power = dmon.groupby("sample_index")["power_w"].mean()
        ax.plot(gpu_util.index, gpu_util.values, color=COLORS[3], linewidth=1.2, label="GPU util")
        ax2 = ax.twinx()
        ax2.plot(power.index, power.values, color=COLORS[4], linewidth=1.2, label="Power")
        ax.set_xlabel("Dmon sample")
        ax.set_ylabel("GPU util")
        ax2.set_ylabel("Power (W)")
    panel_label(ax, "(d)")

    ax = axes[2, 0]
    if not rank_memory.empty:
        peak_reserved = rank_memory.groupby("rank")["max_reserved_mb"].max() / 1024.0
        ax.bar(peak_reserved.index.astype(str), peak_reserved.values, color=COLORS[5])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak reserved (GiB)")
    panel_label(ax, "(e)")

    ax = axes[2, 1]
    if not dmon.empty:
        per_gpu = dmon.groupby("gpu")["gpu_util"].mean()
        ax.bar(per_gpu.index.astype(str), per_gpu.values, color=COLORS[6])
    ax.set_xlabel("GPU")
    ax.set_ylabel("Mean GPU util")
    panel_label(ax, "(f)")

    fig.suptitle("")
    save_fig(fig, fig_dir, "fig3_utilization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
