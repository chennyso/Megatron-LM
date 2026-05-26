#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_csv


def run_plot(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], check=False)


def main() -> None:
    perf_root = Path(__file__).resolve().parents[1]
    outputs = perf_root / "outputs"
    figures = outputs / "figures"
    reports = outputs / "reports"
    summary = outputs / "summary"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    for script_name in ("plot_stage_timeline.py", "plot_vpp_sweep.py", "plot_boundary_comm.py"):
        run_plot(perf_root / "analysis" / "notebooks_or_scripts" / script_name)

    all_rows = read_csv(summary / "all_experiments.csv")
    agg_rows = read_csv(summary / "by_experiment_mean_std.csv")
    skipped = read_csv(summary / "skipped_experiments.csv")
    findings = {}
    if (summary / "key_findings.json").exists():
        findings = json.loads((summary / "key_findings.json").read_text(encoding="utf-8"))

    fastest = findings.get("fastest_config")
    slowest = findings.get("slowest_config")

    lines = [
        "# perf_report",
        "",
        "## Executive Summary",
        "",
        f"- Fastest configuration: {fastest.get('experiment') if fastest else 'n/a'}",
        f"- Slowest configuration: {slowest.get('experiment') if slowest else 'n/a'}",
        "- Qwen3-14B should be treated as the profiling development model because it validates the pipeline mechanics at lower cost.",
        "- Qwen3-32B should be treated as the main VPP / cross-node communication model because 64 layers make VPP sweeps valid and visible.",
        "- VPP fragmentation, exposed P2P, and overlap effectiveness must be concluded only from rows that have real Nsight-derived values.",
        "",
        "## Hardware and Software",
        "",
        "- See `outputs/system_info/summary.txt`",
        "- See `outputs/nccl_tests/`",
        "",
        "## Experiment Matrix",
        "",
        f"- completed rows: {len(all_rows)}",
        f"- aggregated rows: {len(agg_rows)}",
        f"- skipped rows: {len(skipped)}",
        "",
        "### Skipped Experiments",
        "",
    ]
    if skipped:
        for row in skipped:
            lines.append(f"- {row['experiment']}: {row['skip_reason']}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Overall Performance",
        "",
        "| experiment | seq_len | overlap_mode | step_time_ms_mean | tokens_per_second_mean |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['experiment']} | {row['seq_len']} | {row['overlap_mode']} | {row['step_time_ms_mean']} | {row['tokens_per_second_mean']} |"
        )

    lines += [
        "",
        "## PP/VPP Analysis",
        "",
        "- See `outputs/figures/vpp_vs_bubble.png`",
        "- See `outputs/figures/vpp_vs_exposed_p2p.png`",
        "- See `outputs/figures/vpp_vs_p2p_message_count.png`",
        "",
        "## Cross-node Boundary Analysis",
        "",
        "- Focus boundary: stage7 -> stage8",
        "- See `outputs/figures/boundary_comm_stage7_stage8.png`",
        "",
        "## DP vs PP Communication",
        "",
        "- Compare `PP=8,DP=2` vs `PP=16,DP=1` in `outputs/summary/by_experiment_mean_std.csv`",
        "",
        "## Overlap Switch Analysis",
        "",
        "- Compare `baseline`, `p2p_overlap`, `dp_overlap`, `all_overlap`",
        "- See `outputs/figures/overlap_mode_comparison.png`",
        "",
        "## Bottleneck Attribution",
        "",
        "Observation:",
        "Null fields indicate that more NVTX or Nsight schema-specific parsing is still needed.",
        "Evidence:",
        "Rows with missing overlap and exposed communication metrics are preserved instead of fabricated.",
        "Root cause:",
        "Nsight SQLite schemas vary, and pipeline attribution needs version-aware parsing.",
        "Suggested next experiment:",
        "Use Qwen3-32B PP16 VPP1/2/4 at seq4096 with boundary ranks profiled.",
        "Potential innovation:",
        "Topology-aware non-uniform VPP and exposed-P2P-aware partitioning remain the top candidates.",
        "",
        "## Candidate Innovation Points",
        "",
        "- topology-aware non-uniform VPP",
        "- exposed-P2P-aware stage partition",
        "- PP-DP communication phase alignment",
        "- B_W bubble filling",
        "- readiness-aware schedule hint",
        "",
        "## Limitations",
        "",
        "- mock data suppresses loader effects by design",
        "- some Nsight-derived fields may remain null",
        "- B_IN and B_W may not be fully separated without deeper instrumentation",
        "- 5090D consumer topology results do not directly transfer to NVLink/H100 systems",
    ]

    (reports / "perf_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
