#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import panel_label, save_fig


CASE_ORDER = [
    "qwen8b_fsdp_only",
    "qwen8b_pp2_fsdp",
    "qwen8b_pp4_fsdp",
    "qwen8b_pp2_vpp2_fsdp",
]

CASE_LABELS = {
    "qwen8b_fsdp_only": "FSDP-only",
    "qwen8b_pp2_fsdp": "PP2 + FSDP",
    "qwen8b_pp4_fsdp": "PP4 + FSDP",
    "qwen8b_pp2_vpp2_fsdp": "PP2 + VPP2 + FSDP",
}

CASE_NOTES = {
    "qwen8b_fsdp_only": "OOM during FSDP buffer init",
    "qwen8b_pp2_fsdp": "highest throughput",
    "qwen8b_pp4_fsdp": "lower memory, lower throughput",
    "qwen8b_pp2_vpp2_fsdp": "GBS=4 illegal; GBS=8 OOM",
}

STAGE_COLORS = ["#6BAED6", "#74C476", "#FDAE6B", "#B39DDB"]
FWD = "#C6DBEF"
BWD = "#08519C"
BUBBLE = "#FFFFFF"
FAIL_COLOR = "#D73027"
OK_COLOR = "#1A9850"
TEXT = "#1F1F1F"
MUTED = "#6E6E6E"
GRID = "#D9D9D9"


def load_cases(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "case_aggregates.csv")
    df = df[df["case_id"].isin(CASE_ORDER)].copy()
    order = {case_id: i for i, case_id in enumerate(CASE_ORDER)}
    df["order"] = df["case_id"].map(order)
    df["memory_gib"] = df["peak_reserved_mb_mean_mean"].fillna(0).astype(float) / 1024.0
    df["memory_gib_ci"] = df["peak_reserved_mb_mean_ci95_halfwidth"].fillna(0).astype(float) / 1024.0
    return df.sort_values("order").reset_index(drop=True)


def stage_assignment(case_id: str) -> list[int]:
    if case_id == "qwen8b_fsdp_only":
        return [0] * 8
    if case_id in {"qwen8b_pp2_fsdp", "qwen8b_pp2_vpp2_fsdp"}:
        return [0, 0, 0, 0, 1, 1, 1, 1]
    return [0, 0, 1, 1, 2, 2, 3, 3]


def schedule_lanes(case_id: str) -> tuple[list[str], int, dict[int, list[tuple[int, int, str]]]]:
    if case_id == "qwen8b_fsdp_only":
        return ["DP shard"], 8, {0: [(0, 8, "full")]}
    if case_id == "qwen8b_pp2_fsdp":
        return ["Stage 0", "Stage 1"], 8, {
            0: [(0, 1, "fwd"), (1, 2, "fwd"), (3, 4, "bwd"), (4, 5, "bwd")],
            1: [(1, 2, "fwd"), (2, 3, "fwd"), (4, 5, "bwd"), (5, 6, "bwd")],
        }
    if case_id == "qwen8b_pp4_fsdp":
        return ["Stage 0", "Stage 1", "Stage 2", "Stage 3"], 10, {
            0: [(0, 1, "fwd"), (1, 2, "fwd"), (5, 6, "bwd"), (6, 7, "bwd")],
            1: [(1, 2, "fwd"), (2, 3, "fwd"), (6, 7, "bwd"), (7, 8, "bwd")],
            2: [(2, 3, "fwd"), (3, 4, "fwd"), (7, 8, "bwd"), (8, 9, "bwd")],
            3: [(3, 4, "fwd"), (4, 5, "fwd"), (8, 9, "bwd"), (9, 10, "bwd")],
        }
    return ["S0-V0", "S0-V1", "S1-V0", "S1-V1"], 10, {
        0: [(0, 1, "fwd"), (2, 3, "fwd"), (5, 6, "bwd"), (7, 8, "bwd")],
        1: [(1, 2, "fwd"), (3, 4, "fwd"), (6, 7, "bwd"), (8, 9, "bwd")],
        2: [(1, 2, "fwd"), (3, 4, "fwd"), (6, 7, "bwd"), (8, 9, "bwd")],
        3: [(2, 3, "fwd"), (4, 5, "fwd"), (7, 8, "bwd"), (9, 10, "bwd")],
    }


def draw_gpu_mapping(ax, case_id: str, x: float, y: float, w: float, h: float) -> None:
    stages = stage_assignment(case_id)
    box_w = w / 8.0
    for gpu, stage in enumerate(stages):
        rect = patches.Rectangle(
            (x + gpu * box_w, y),
            box_w * 0.92,
            h,
            facecolor=STAGE_COLORS[stage],
            edgecolor="white",
            lw=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + gpu * box_w + box_w * 0.46, y + h * 0.66, f"G{gpu}", ha="center", va="center", fontsize=6.0)
        label = "D" if case_id == "qwen8b_fsdp_only" else f"S{stage}"
        ax.text(x + gpu * box_w + box_w * 0.46, y + h * 0.30, label, ha="center", va="center", fontsize=6.5, fontweight="bold")

    stage_names = []
    if case_id == "qwen8b_fsdp_only":
        stage_names = [("D", "data-parallel shards", STAGE_COLORS[0])]
    elif case_id in {"qwen8b_pp2_fsdp", "qwen8b_pp2_vpp2_fsdp"}:
        stage_names = [("S0", "4 GPUs", STAGE_COLORS[0]), ("S1", "4 GPUs", STAGE_COLORS[1])]
    else:
        stage_names = [
            ("S0", "2 GPUs", STAGE_COLORS[0]),
            ("S1", "2 GPUs", STAGE_COLORS[1]),
            ("S2", "2 GPUs", STAGE_COLORS[2]),
            ("S3", "2 GPUs", STAGE_COLORS[3]),
        ]
    lx = x
    for name, text, color in stage_names:
        ax.add_patch(patches.Rectangle((lx, y - h * 0.55), h * 0.25, h * 0.25, facecolor=color, edgecolor="none"))
        ax.text(lx + h * 0.32, y - h * 0.43, f"{name}: {text}", ha="left", va="center", fontsize=5.8, color=MUTED)
        lx += w / max(len(stage_names), 1)


def draw_schedule(ax, case_id: str, x: float, y: float, w: float, h: float) -> None:
    lanes, slots, active = schedule_lanes(case_id)
    lane_h = h / len(lanes)
    slot_w = w / slots
    for lane_idx, label in enumerate(lanes):
        ly = y + (len(lanes) - 1 - lane_idx) * lane_h
        ax.text(x - 0.008, ly + lane_h * 0.5, label, ha="right", va="center", fontsize=5.8, color=TEXT)
        for slot in range(slots):
            ax.add_patch(
                patches.Rectangle(
                    (x + slot * slot_w, ly + lane_h * 0.08),
                    slot_w * 0.96,
                    lane_h * 0.72,
                    facecolor=BUBBLE,
                    edgecolor="#BDBDBD",
                    lw=0.35,
                )
            )
        for start, end, phase in active[lane_idx]:
            if phase == "full":
                color = "#9ECAE1"
                text = "full stack"
            else:
                color = FWD if phase == "fwd" else BWD
                text = "F" if phase == "fwd" else "B"
            ax.add_patch(
                patches.Rectangle(
                    (x + start * slot_w, ly + lane_h * 0.08),
                    (end - start) * slot_w * 0.96,
                    lane_h * 0.72,
                    facecolor=color,
                    edgecolor="white",
                    lw=0.35,
                )
            )
            if slot_w * (end - start) > 0.022:
                ax.text(
                    x + (start + end) * slot_w * 0.5,
                    ly + lane_h * 0.44,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color="white" if phase == "bwd" else TEXT,
                    fontweight="bold",
                )


def draw_outcome(ax, row: pd.Series, x: float, y: float, w: float, h: float) -> None:
    feasible = bool(row["feasible"])
    case_id = str(row["case_id"])
    color = OK_COLOR if feasible else FAIL_COLOR
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.007",
            facecolor="#F7FCF5" if feasible else "#FFF5F0",
            edgecolor=color,
            lw=0.9,
        )
    )
    if feasible:
        tps = float(row["tokens_per_second_mean_mean"])
        tps_ci = float(row["tokens_per_second_mean_ci95_halfwidth"])
        step = float(row["iter_time_ms_mean_mean"])
        mem = float(row["memory_gib"])
        repeats = int(row["valid_repeat_count"])
        ax.text(x + w * 0.06, y + h * 0.73, f"{tps:,.0f} tokens/s", ha="left", va="center", fontsize=8.0, fontweight="bold", color=TEXT)
        ax.text(x + w * 0.06, y + h * 0.50, f"95% CI +/- {tps_ci:.1f}, n={repeats}", ha="left", va="center", fontsize=5.9, color=MUTED)
        ax.text(x + w * 0.06, y + h * 0.30, f"{step:,.0f} ms/step, {mem:.1f} GiB", ha="left", va="center", fontsize=6.4, color=TEXT)
        ax.text(x + w * 0.06, y + h * 0.12, CASE_NOTES[case_id], ha="left", va="center", fontsize=5.8, color=MUTED)
    else:
        total = int(row["total_repeat_count"])
        failed = int(row["failed_repeat_count"])
        ax.text(x + w * 0.06, y + h * 0.66, "Not feasible", ha="left", va="center", fontsize=8.0, fontweight="bold", color=color)
        ax.text(x + w * 0.06, y + h * 0.42, f"{failed}/{total} attempts failed", ha="left", va="center", fontsize=6.5, color=TEXT)
        ax.text(x + w * 0.06, y + h * 0.19, CASE_NOTES[case_id], ha="left", va="center", fontsize=5.9, color=MUTED)


def draw_strategy_map(ax, df: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    headers = [
        ("Strategy", 0.02, 0.16),
        ("8-GPU assignment", 0.19, 0.30),
        ("Pipeline schedule", 0.52, 0.29),
        ("Observed outcome", 0.83, 0.15),
    ]
    for title, x, _ in headers:
        ax.text(x, 0.965, title, ha="left", va="center", fontsize=8.0, fontweight="bold")
    ax.plot([0.02, 0.98], [0.94, 0.94], color=GRID, lw=0.8)

    row_h = 0.205
    y_top = 0.91
    for i, case_id in enumerate(CASE_ORDER):
        row = df[df["case_id"] == case_id].iloc[0]
        y = y_top - (i + 1) * row_h
        feasible = bool(row["feasible"])
        ax.add_patch(
            patches.Rectangle(
                (0.02, y + 0.012),
                0.96,
                row_h - 0.018,
                facecolor="#FAFAFA" if i % 2 == 0 else "#FFFFFF",
                edgecolor="#E5E5E5",
                lw=0.6,
            )
        )
        status_color = OK_COLOR if feasible else FAIL_COLOR
        ax.add_patch(patches.Rectangle((0.02, y + 0.012), 0.006, row_h - 0.018, facecolor=status_color, edgecolor="none"))
        ax.text(0.035, y + row_h * 0.67, CASE_LABELS[case_id], ha="left", va="center", fontsize=7.8, fontweight="bold")
        ax.text(
            0.035,
            y + row_h * 0.37,
            "valid" if feasible else "failed",
            ha="left",
            va="center",
            fontsize=6.4,
            color=status_color,
            fontweight="bold",
        )
        draw_gpu_mapping(ax, case_id, 0.19, y + row_h * 0.42, 0.27, row_h * 0.26)
        draw_schedule(ax, case_id, 0.55, y + row_h * 0.28, 0.24, row_h * 0.46)
        draw_outcome(ax, row, 0.83, y + row_h * 0.25, 0.14, row_h * 0.54)

    ax.add_patch(patches.Rectangle((0.55, 0.025), 0.018, 0.018, facecolor=FWD, edgecolor=GRID, lw=0.4))
    ax.text(0.572, 0.034, "forward", fontsize=5.8, va="center")
    ax.add_patch(patches.Rectangle((0.63, 0.025), 0.018, 0.018, facecolor=BWD, edgecolor=GRID, lw=0.4))
    ax.text(0.652, 0.034, "backward", fontsize=5.8, va="center")
    ax.add_patch(patches.Rectangle((0.72, 0.025), 0.018, 0.018, facecolor=BUBBLE, edgecolor="#BDBDBD", lw=0.4))
    ax.text(0.742, 0.034, "pipeline bubble / idle slot", fontsize=5.8, va="center")


def draw_tokens_memory_tradeoff(ax, df: pd.DataFrame) -> None:
    feasible = df[df["feasible"]].copy()
    infeasible = df[~df["feasible"]].copy()
    for _, row in feasible.iterrows():
        color = STAGE_COLORS[1] if row["case_id"] == "qwen8b_pp2_fsdp" else STAGE_COLORS[2]
        ax.errorbar(
            row["memory_gib"],
            row["tokens_per_second_mean_mean"],
            xerr=row["memory_gib_ci"],
            yerr=row["tokens_per_second_mean_ci95_halfwidth"],
            fmt="o",
            ms=8,
            color=color,
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        ax.text(
            row["memory_gib"] + 0.08,
            row["tokens_per_second_mean_mean"] + 55,
            CASE_LABELS[row["case_id"]],
            fontsize=7.0,
            ha="left",
            va="bottom",
        )
    if not infeasible.empty:
        lines = ["No steady-state result:"]
        for _, row in infeasible.iterrows():
            lines.append(f"{CASE_LABELS[row['case_id']]} ({int(row['valid_repeat_count'])}/{int(row['total_repeat_count'])})")
        ax.text(
            0.035,
            0.075,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color=FAIL_COLOR,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF5F0", "ec": FAIL_COLOR, "lw": 0.7},
        )
    ax.set_xlabel("Peak reserved memory (GiB)")
    ax.set_ylabel("Tokens/s")
    ax.set_xlim(24.2, 28.6)
    ax.set_ylim(5600, 7400)
    ax.grid(axis="both", color="#ECECEC", lw=0.6)


def draw_normalized_comparison(ax, df: pd.DataFrame) -> None:
    feasible = df[df["feasible"]].copy()
    labels = [CASE_LABELS[c] for c in feasible["case_id"]]
    x = np.arange(len(labels))
    width = 0.26
    baseline = feasible[feasible["case_id"] == "qwen8b_pp2_fsdp"].iloc[0]
    metrics = [
        ("tokens/s", feasible["tokens_per_second_mean_mean"] / baseline["tokens_per_second_mean_mean"], OK_COLOR),
        ("step time", feasible["iter_time_ms_mean_mean"] / baseline["iter_time_ms_mean_mean"], "#4C78A8"),
        ("memory", feasible["memory_gib"] / baseline["memory_gib"], "#F58518"),
    ]
    for idx, (name, values, color) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, values, width=width, label=name, color=color, alpha=0.85)
    ax.axhline(1.0, color="#333333", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Relative to PP2 + FSDP")
    ax.set_ylim(0.0, 1.22)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)


def draw_feasibility_bar(ax, df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    valid = df["valid_repeat_count"].astype(float)
    failed = df["failed_repeat_count"].astype(float)
    ax.bar(x, valid, color=OK_COLOR, label="valid")
    ax.bar(x, failed, bottom=valid, color=FAIL_COLOR, label="failed")
    for idx, row in df.iterrows():
        total = int(row["total_repeat_count"])
        valid_n = int(row["valid_repeat_count"])
        ax.text(idx, valid_n + int(row["failed_repeat_count"]) + 0.08, f"{valid_n}/{total}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in df["case_id"]], rotation=22, ha="right")
    ax.set_ylabel("Runs")
    ax.set_ylim(0, max(df["total_repeat_count"].max() + 0.8, 4.8))
    ax.legend(ncols=2, loc="upper right", frameon=False)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    fig_dir = analysis_dir / "figures"
    df = load_cases(analysis_dir)

    fig = plt.figure(figsize=(13.4, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.35, wspace=0.33)

    ax = fig.add_subplot(gs[0, :])
    draw_strategy_map(ax, df)
    panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[1, 0])
    draw_tokens_memory_tradeoff(ax, df)
    panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 1])
    draw_normalized_comparison(ax, df)
    panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 2])
    draw_feasibility_bar(ax, df)
    panel_label(ax, "(d)")

    save_fig(fig, fig_dir, "fig2_strategy_performance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
