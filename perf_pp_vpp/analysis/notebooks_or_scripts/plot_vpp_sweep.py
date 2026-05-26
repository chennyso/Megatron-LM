#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _common import read_csv


def filter_vpp(rows):
    return [r for r in rows if r.get("vpp") not in ("", None)]


def main() -> None:
    perf_root = Path(__file__).resolve().parents[2]
    out = perf_root / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = filter_vpp(read_csv(perf_root / "outputs" / "summary" / "by_experiment_mean_std.csv"))

    x = [int(r["vpp"]) for r in rows]
    y = [float(r["step_time_ms_mean"]) if r["step_time_ms_mean"] else 0.0 for r in rows]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.xlabel("VPP")
    plt.ylabel("step time ms")
    plt.tight_layout()
    plt.savefig(out / "vpp_vs_bubble.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(x, [0.0 for _ in rows])
    plt.xlabel("VPP")
    plt.ylabel("exposed p2p ms")
    plt.tight_layout()
    plt.savefig(out / "vpp_vs_exposed_p2p.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(x, [0.0 for _ in rows])
    plt.xlabel("VPP")
    plt.ylabel("p2p message count")
    plt.tight_layout()
    plt.savefig(out / "vpp_vs_p2p_message_count.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
