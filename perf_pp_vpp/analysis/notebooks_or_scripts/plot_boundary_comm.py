#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _common import read_csv


def main() -> None:
    perf_root = Path(__file__).resolve().parents[2]
    out = perf_root / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(perf_root / "outputs" / "summary" / "by_experiment_mean_std.csv")
    labels = [r["experiment"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), [0.0 for _ in labels])
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("boundary comm ms")
    plt.tight_layout()
    plt.savefig(out / "boundary_comm_stage7_stage8.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(labels)), [0.0 for _ in labels])
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("overlap mode comparison")
    plt.tight_layout()
    plt.savefig(out / "overlap_mode_comparison.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
