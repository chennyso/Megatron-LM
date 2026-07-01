#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ITER_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
TOKENS_RE = re.compile(r"tokens-per-second[^0-9]*([0-9.]+)")
THROUGHPUT_RE = re.compile(r"throughput per GPU.*?:\s*([0-9.]+)")
LOSS_RE = re.compile(r"lm loss value:\s*([0-9.Ee+-]+)")
OOM_RE = re.compile(r"(out of memory|cuda error|nccl error)", re.IGNORECASE)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def parse_dmon(path: Path) -> dict:
    if not path.exists():
        return {}
    util_values: list[float] = []
    mem_values: list[float] = []
    power_values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 8:
            continue
        try:
            util_values.append(float(tokens[3]))
            power_values.append(float(tokens[4]))
            mem_values.append(float(tokens[5]))
        except ValueError:
            continue
    return {
        "gpu_util_mean": statistics.mean(util_values) if util_values else None,
        "power_mean_w": statistics.mean(power_values) if power_values else None,
        "mem_clock_or_mem_metric_mean": statistics.mean(mem_values) if mem_values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--dmon-path")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    iter_ms = [float(value) for value in ITER_RE.findall(text)]
    tokens = [float(value) for value in TOKENS_RE.findall(text)]
    throughputs = [float(value) for value in THROUGHPUT_RE.findall(text)]
    losses = [float(value) for value in LOSS_RE.findall(text)]
    errors = sorted(set(match.group(0) for match in OOM_RE.finditer(text)))

    payload = {
        "case_id": args.case_id,
        "log_path": str(log_path),
        "iter_time_ms": {
            "count": len(iter_ms),
            "mean": statistics.mean(iter_ms) if iter_ms else None,
            "p50": percentile(iter_ms, 0.5),
            "p95": percentile(iter_ms, 0.95),
        },
        "tokens_per_second": {
            "count": len(tokens),
            "mean": statistics.mean(tokens) if tokens else None,
            "p50": percentile(tokens, 0.5),
            "p95": percentile(tokens, 0.95),
        },
        "throughput_per_gpu_tflops": {
            "count": len(throughputs),
            "mean": statistics.mean(throughputs) if throughputs else None
        },
        "loss": {
            "count": len(losses),
            "last": losses[-1] if losses else None
        },
        "error_markers": errors,
        "oom_or_runtime_error": bool(errors),
    }

    if args.dmon_path:
        payload["dmon"] = parse_dmon(Path(args.dmon_path))

    Path(args.output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
