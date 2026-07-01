from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
    }
)


def save_fig(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600)


def panel_label(ax, text: str) -> None:
    ax.text(-0.14, 1.04, text, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def rotate_labels(ax, rotation: int = 30) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
