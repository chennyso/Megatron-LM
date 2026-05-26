#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import importlib.util
from pathlib import Path

from _common import maybe_float


def load_score_profile():
    script_path = Path(__file__).with_name("14b_score_covpipe_profile.py")
    spec = importlib.util.spec_from_file_location("covpipe_profile_score", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load scorer from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_profile


score_profile = load_score_profile()


LINE_RE = re.compile(
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+) \| throughput per GPU \(TFLOP/s/GPU\): (?P<tflops>[0-9.]+)"
)


def parse_step_lines(run_dir: Path) -> tuple[float | None, float | None, int]:
    values: list[tuple[float, float]] = []
    for path in sorted(run_dir.glob("remote_artifacts/megatron_stdout_node*.log")):
        for line in path.read_text(errors="ignore").splitlines():
            match = LINE_RE.search(line)
            if match:
                values.append((float(match.group("ms")), float(match.group("tflops"))))
    if not values:
        return None, None, 0
    warm = values[2:] if len(values) > 2 else values
    step_ms = sum(x[0] for x in warm) / len(warm)
    tflops = sum(x[1] for x in warm) / len(warm)
    return step_ms, tflops, len(values)


def candidate_name(run_dir: Path) -> str:
    env_path = run_dir / "env.txt"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("PERF_PIPELINE_MODEL_PARALLEL_LAYOUT="):
                return line.split("=", 1)[1]
    return run_dir.parts[-4]


def collect_profile_score(run_dir: Path) -> dict | None:
    nsys_jsons = sorted(run_dir.glob("remote_artifacts/nsys/*.analysis.json"))
    if not nsys_jsons:
        return None
    # Prefer boundary-adjacent ranks if available.
    preferred = []
    for path in nsys_jsons:
        if any(token in path.name for token in ("rank7", "rank8", "rank6", "rank9")):
            preferred.append(path)
    target = preferred[0] if preferred else nsys_jsons[0]
    return score_profile(target)


def summarize_campaign(campaign_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(campaign_root.glob("runs/*/seq*/repeat*/baseline*")):
        step_ms, tflops, samples = parse_step_lines(run_dir / "remote_artifacts")
        profile = collect_profile_score(run_dir)
        rows.append(
            {
                "run_dir": str(run_dir),
                "experiment": run_dir.parts[-4],
                "seq_len": run_dir.parts[-3].replace("seq", ""),
                "repeat": run_dir.parts[-2].replace("repeat", ""),
                "layout_or_candidate": candidate_name(run_dir),
                "step_time_ms_mean": step_ms,
                "tflops_per_gpu_mean": tflops,
                "num_metric_rows": samples,
                "profile_score": profile,
            }
        )
    rows.sort(
        key=lambda row: (
            maybe_float(row["tflops_per_gpu_mean"]) is None,
            -(maybe_float(row["tflops_per_gpu_mean"]) or -1e18),
            maybe_float(row["step_time_ms_mean"]) or 1e18,
        )
    )
    return rows


def write_report(campaign_root: Path, rows: list[dict]) -> None:
    report_lines = ["# COVPipe Search Report", ""]
    for row in rows:
        headline = f"- `{Path(row['run_dir']).parts[-2]}`: "
        if row["tflops_per_gpu_mean"] is None:
            report_lines.append(headline + "no throughput metrics")
            continue
        line = (
            headline
            + f"{row['step_time_ms_mean']:.1f} ms, {row['tflops_per_gpu_mean']:.1f} TFLOP/s/GPU"
        )
        score = row.get("profile_score")
        if score:
            line += (
                f", cover_ratio={score['covering_ratio']:.3f}, "
                f"exposed_proxy_ms={score['exposed_comm_proxy_ns'] / 1e6:.3f}"
            )
        report_lines.append(line)
    (campaign_root / "covpipe_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    campaign_root = Path(os.environ.get("CAMPAIGN_ROOT", "")).resolve()
    if not campaign_root.exists():
        raise SystemExit(f"campaign root does not exist: {campaign_root}")
    rows = summarize_campaign(campaign_root)
    (campaign_root / "covpipe_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(campaign_root, rows)


if __name__ == "__main__":
    main()
