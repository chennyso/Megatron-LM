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
    values = [float(r["step_time_ms_mean"]) if r["step_time_ms_mean"] else 0.0 for r in rows]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("step time ms")
    plt.tight_layout()
    plt.savefig(out / "step_time_by_config.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    tok = [float(r["tokens_per_second_mean"]) if r["tokens_per_second_mean"] else 0.0 for r in rows]
    plt.bar(range(len(labels)), tok)
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right")
    plt.ylabel("tokens / s")
    plt.tight_layout()
    plt.savefig(out / "tokens_per_second_by_config.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.text(0.5, 0.5, "stage timeline placeholder\nrequires rank event attribution", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "stage_imbalance_heatmap.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
