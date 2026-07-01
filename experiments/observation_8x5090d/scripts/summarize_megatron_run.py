#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from stats_utils import summarize_numeric


ITER_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\s*\|"
    r".*?consumed samples:\s*(?P<consumed>\d+)\s*\|"
    r".*?elapsed time per iteration \(ms\):\s*(?P<iter_ms>[0-9.]+)\s*\|"
    r".*?learning rate:\s*(?P<lr>[0-9.Ee+-]+)\s*\|"
    r".*?global batch size:\s*(?P<global_batch_size>\d+)\s*\|"
    r".*?lm loss:\s*(?P<loss>[0-9.Ee+-]+)\s*\|"
    r".*?grad norm:\s*(?P<grad_norm>[0-9.Ee+-]+)\s*\|",
    re.IGNORECASE,
)
MEMORY_RE = re.compile(
    r"\[Rank\s+(?P<rank>\d+)\]\s+\(after\s+(?P<iteration>\d+)\s+iterations\)\s+memory \(MB\)\s+\|"
    r"\s+allocated:\s*(?P<allocated>[0-9.]+)\s+\|"
    r"\s+max allocated:\s*(?P<max_allocated>[0-9.]+)\s+\|"
    r"\s+reserved:\s*(?P<reserved>[0-9.]+)\s+\|"
    r"\s+max reserved:\s*(?P<max_reserved>[0-9.]+)",
    re.IGNORECASE,
)
THROUGHPUT_RE = re.compile(r"throughput per GPU.*?:\s*([0-9.]+)", re.IGNORECASE)
LOSS_RE = re.compile(r"lm loss value:\s*([0-9.Ee+-]+)")
OOM_RE = re.compile(r"(out of memory|cuda error|nccl error|traceback)", re.IGNORECASE)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_case_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_step_rows(text: str, seq_length: int | None) -> list[dict]:
    rows: list[dict] = []
    for match in ITER_RE.finditer(text):
        iter_ms = float(match.group("iter_ms"))
        global_batch_size = int(match.group("global_batch_size"))
        elapsed_s = iter_ms / 1000.0
        rows.append(
            {
                "iteration": int(match.group("iteration")),
                "iteration_total": int(match.group("total")),
                "consumed_samples": int(match.group("consumed")),
                "iter_time_ms": iter_ms,
                "learning_rate": float(match.group("lr")),
                "global_batch_size": global_batch_size,
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "samples_per_second": global_batch_size / elapsed_s if elapsed_s > 0 else None,
                "tokens_per_second": (global_batch_size * seq_length) / elapsed_s if seq_length and elapsed_s > 0 else None,
            }
        )
    return rows


def build_memory_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in MEMORY_RE.finditer(text):
        rows.append(
            {
                "iteration": int(match.group("iteration")),
                "rank": int(match.group("rank")),
                "allocated_mb": float(match.group("allocated")),
                "max_allocated_mb": float(match.group("max_allocated")),
                "reserved_mb": float(match.group("reserved")),
                "max_reserved_mb": float(match.group("max_reserved")),
            }
        )
    return rows


def parse_dmon(path: Path) -> tuple[list[dict], dict]:
    if not path.exists():
        return [], {}
    samples: list[dict] = []
    for sample_index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 4 or not tokens[0].isdigit():
            continue
        try:
            samples.append(
                {
                    "sample_index": sample_index,
                    "gpu": int(tokens[0]),
                    "power_w": float(tokens[1]),
                    "gpu_util": float(tokens[2]),
                    "mem_util_or_clock": float(tokens[3]),
                }
            )
        except ValueError:
            continue

    summary = {
        "gpu_util": summarize_numeric(sample["gpu_util"] for sample in samples),
        "power_w": summarize_numeric(sample["power_w"] for sample in samples),
        "mem_util_or_clock": summarize_numeric(sample["mem_util_or_clock"] for sample in samples),
        "gpu_util_by_gpu": {},
    }
    for gpu_id in sorted({sample["gpu"] for sample in samples}):
        per_gpu = [sample["gpu_util"] for sample in samples if sample["gpu"] == gpu_id]
        summary["gpu_util_by_gpu"][str(gpu_id)] = summarize_numeric(per_gpu)
    return samples, summary


def summarize_memory(memory_rows: list[dict]) -> dict:
    if not memory_rows:
        return {
            "peak_allocated_mb": summarize_numeric([]),
            "peak_reserved_mb": summarize_numeric([]),
            "rank_skew_percent": None,
        }
    peak_allocated_per_rank: dict[int, float] = {}
    peak_reserved_per_rank: dict[int, float] = {}
    for row in memory_rows:
        peak_allocated_per_rank[row["rank"]] = max(peak_allocated_per_rank.get(row["rank"], 0.0), row["max_allocated_mb"])
        peak_reserved_per_rank[row["rank"]] = max(peak_reserved_per_rank.get(row["rank"], 0.0), row["max_reserved_mb"])
    allocated_values = list(peak_allocated_per_rank.values())
    reserved_values = list(peak_reserved_per_rank.values())
    rank_skew_percent = None
    if allocated_values:
        mean_allocated = sum(allocated_values) / len(allocated_values)
        if mean_allocated > 0:
            rank_skew_percent = (max(allocated_values) - min(allocated_values)) / mean_allocated * 100.0
    return {
        "peak_allocated_mb": summarize_numeric(allocated_values),
        "peak_reserved_mb": summarize_numeric(reserved_values),
        "rank_skew_percent": rank_skew_percent,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--dmon-path")
    parser.add_argument("--case-config-path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    case_config = load_case_config(Path(args.case_config_path) if args.case_config_path else None)
    seq_length = case_config.get("model_spec", {}).get("seq_length")
    warmup_steps = case_config.get("warmup_steps", 0)

    step_rows = build_step_rows(text, seq_length)
    steady_state_rows = [row for row in step_rows if row["iteration"] > warmup_steps] if warmup_steps else step_rows
    memory_rows = build_memory_rows(text)
    dmon_samples, dmon_summary = parse_dmon(Path(args.dmon_path)) if args.dmon_path else ([], {})
    throughputs = [float(value) for value in THROUGHPUT_RE.findall(text)]
    losses = [float(value) for value in LOSS_RE.findall(text)]
    errors = sorted(set(match.group(0) for match in OOM_RE.finditer(text)))
    memory_summary = summarize_memory(memory_rows)

    per_step_csv = Path(args.output_path).with_name("per_step.csv")
    if step_rows:
        write_csv(per_step_csv, step_rows, list(step_rows[0].keys()))
    rank_memory_csv = Path(args.output_path).with_name("rank_memory.csv")
    if memory_rows:
        write_csv(rank_memory_csv, memory_rows, list(memory_rows[0].keys()))
    dmon_csv = Path(args.output_path).with_name("gpu_dmon_samples.csv")
    if dmon_samples:
        write_csv(dmon_csv, dmon_samples, list(dmon_samples[0].keys()))

    payload = {
        "case_id": args.case_id,
        "log_path": str(log_path),
        "phase": case_config.get("phase"),
        "paper_model_id": case_config.get("paper_model_id"),
        "dataset_spec": case_config.get("dataset_spec"),
        "repeat_index": case_config.get("repeat_index"),
        "warmup_steps": warmup_steps,
        "measure_steps": case_config.get("measure_steps"),
        "steady_state_step_count": len(steady_state_rows),
        "iter_time_ms": summarize_numeric(row["iter_time_ms"] for row in steady_state_rows),
        "samples_per_second": summarize_numeric(row["samples_per_second"] for row in steady_state_rows),
        "tokens_per_second": summarize_numeric(row["tokens_per_second"] for row in steady_state_rows),
        "grad_norm": summarize_numeric(row["grad_norm"] for row in steady_state_rows),
        "learning_rate": summarize_numeric(row["learning_rate"] for row in steady_state_rows),
        "throughput_per_gpu_tflops": summarize_numeric(throughputs),
        "loss": {
            "count": len(losses),
            "last": losses[-1] if losses else None,
            "steady_state": summarize_numeric(row["loss"] for row in steady_state_rows),
        },
        "memory": memory_summary,
        "dmon": dmon_summary,
        "error_markers": errors,
        "oom_or_runtime_error": bool(errors),
        "figure_membership": case_config.get("figure_membership", []),
        "claim_membership": case_config.get("claim_membership", []),
        "artifacts": {
            "per_step_csv": per_step_csv.name if step_rows else None,
            "rank_memory_csv": rank_memory_csv.name if memory_rows else None,
            "gpu_dmon_samples_csv": dmon_csv.name if dmon_samples else None,
        },
    }

    Path(args.output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
