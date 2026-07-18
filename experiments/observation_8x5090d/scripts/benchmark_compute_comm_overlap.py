#!/usr/bin/env python3
"""Measure Qwen3-32B GEMM interference with PP-style NCCL P2P traffic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ComputeCase:
    case_id: str
    action_kind: str
    m: int
    k: int
    n: int


def compute_catalog() -> list[ComputeCase]:
    return [
        ComputeCase("qkv_forward", "F", 4096, 5120, 3200),
        ComputeCase("mlp_forward", "F", 4096, 5120, 25600),
        ComputeCase("mlp_dinput", "dI", 4096, 25600, 5120),
        ComputeCase("mlp_dweight", "dW", 5120, 4096, 25600),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def reduce_valid(value: bool, device: torch.device) -> bool:
    tensor = torch.tensor([int(value)], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["case_id"], row["compute_location"]), []).append(row)
    metrics = (
        "isolated_comm_ms",
        "isolated_compute_ms",
        "concurrent_ms",
        "contention_factor",
        "overlap_efficiency",
        "exposed_overhead_ms",
    )
    summary_rows: list[dict] = []
    for (case_id, compute_location), bucket in sorted(grouped.items()):
        summary = {
            "case_id": case_id,
            "action_kind": bucket[0]["action_kind"],
            "compute_location": compute_location,
            "n": len(bucket),
            "payload_valid": all(row["payload_valid"] for row in bucket),
            "compute_valid": all(row["compute_valid"] for row in bucket),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in bucket]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = 2.776 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            summary.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_ci95": ci95,
                    f"{metric}_p50": statistics.median(values),
                    f"{metric}_p95": percentile(values, 0.95),
                }
            )
        summary_rows.append(summary)
    return summary_rows


class OverlapRunner:
    def __init__(self, case: ComputeCase, compute_rank: int, cfg: dict, device: torch.device):
        self.case = case
        self.compute_rank = compute_rank
        self.cfg = cfg
        self.device = device
        self.rank = dist.get_rank()
        self.elements = math.ceil(int(cfg["message_size_bytes"]) / 2)
        self.send = None
        self.recv = None
        if self.rank == 0:
            self.send = torch.full(
                (self.elements,), 17.0, dtype=torch.bfloat16, device=self.device
            )
        elif self.rank == 1:
            self.recv = torch.full(
                (self.elements,), -1.0, dtype=torch.bfloat16, device=self.device
            )

        self.compute_stream = None
        self.left = None
        self.right = None
        self.output = None
        if self.rank == compute_rank:
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self.left = torch.randn(
                (case.m, case.k), dtype=torch.bfloat16, device=self.device
            )
            self.right = torch.randn(
                (case.k, case.n), dtype=torch.bfloat16, device=self.device
            )
            self.output = torch.empty(
                (case.m, case.n), dtype=torch.bfloat16, device=self.device
            )

    def issue_comm(self, count: int) -> None:
        ops: list[dist.P2POp] = []
        if self.rank == 0:
            ops.append(dist.P2POp(dist.isend, self.send, 1))
        elif self.rank == 1:
            ops.append(dist.P2POp(dist.irecv, self.recv, 0))
        if not ops:
            return
        for _ in range(count):
            for work in dist.batch_isend_irecv(ops):
                work.wait()

    def issue_compute(self, count: int) -> None:
        if self.rank != self.compute_rank:
            return
        with torch.cuda.stream(self.compute_stream):
            for _ in range(count):
                torch.mm(self.left, self.right, out=self.output)

    def measure(self, mode: str) -> tuple[float, bool, bool]:
        warmup = int(self.cfg["warmup_iterations"])
        comm_count = int(self.cfg["comm_iterations"])
        compute_count = int(self.cfg["compute_iterations"])
        dist.barrier()
        if mode in {"comm", "concurrent"}:
            self.issue_comm(warmup)
        if mode in {"compute", "concurrent"}:
            self.issue_compute(warmup)
        torch.cuda.synchronize()
        if self.recv is not None:
            self.recv.fill_(-1.0)
        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()
        if mode in {"compute", "concurrent"}:
            self.issue_compute(compute_count)
        if mode in {"comm", "concurrent"}:
            self.issue_comm(comm_count)
        torch.cuda.synchronize()
        elapsed_ms = reduce_max((time.perf_counter() - start) * 1000.0, self.device)

        payload_valid = True
        if self.recv is not None and mode in {"comm", "concurrent"}:
            payload_valid = float(self.recv[0].item()) == 17.0
            payload_valid = payload_valid and float(self.recv[-1].item()) == 17.0
        payload_valid = reduce_valid(payload_valid, self.device)
        compute_valid = True
        if self.output is not None and mode in {"compute", "concurrent"}:
            compute_valid = bool(torch.isfinite(self.output[0, 0]).item())
            compute_valid = compute_valid and bool(torch.isfinite(self.output[-1, -1]).item())
        compute_valid = reduce_valid(compute_valid, self.device)
        dist.barrier()
        return elapsed_ms, payload_valid, compute_valid


def make_record(
    case: ComputeCase,
    location: str,
    repeat: int,
    cfg: dict,
    measurements: dict[str, tuple[float, bool, bool]],
) -> dict:
    comm_ms = measurements["comm"][0]
    compute_ms = measurements["compute"][0]
    concurrent_ms = measurements["concurrent"][0]
    ideal_ms = max(comm_ms, compute_ms)
    overlap_denominator = min(comm_ms, compute_ms)
    return {
        "case_id": case.case_id,
        "action_kind": case.action_kind,
        "compute_location": location,
        "m": case.m,
        "k": case.k,
        "n": case.n,
        "message_size_bytes": int(cfg["message_size_bytes"]),
        "comm_iterations": int(cfg["comm_iterations"]),
        "compute_iterations": int(cfg["compute_iterations"]),
        "repeat": repeat,
        "isolated_comm_ms": comm_ms,
        "isolated_compute_ms": compute_ms,
        "concurrent_ms": concurrent_ms,
        "ideal_overlap_ms": ideal_ms,
        "serial_sum_ms": comm_ms + compute_ms,
        "contention_factor": concurrent_ms / ideal_ms,
        "overlap_efficiency": (comm_ms + compute_ms - concurrent_ms) / overlap_denominator,
        "exposed_overhead_ms": concurrent_ms - ideal_ms,
        "payload_valid": measurements["comm"][1] and measurements["concurrent"][1],
        "compute_valid": measurements["compute"][2] and measurements["concurrent"][2],
    }


def main() -> int:
    args = parse_args()
    matrix = json.loads(Path(args.matrix_path).read_text(encoding="utf-8"))
    cfg = dict(matrix["compute_comm_motifs"])
    if os.environ.get("OBS_REPEAT_COUNT_OVERRIDE"):
        cfg["repetitions"] = int(os.environ["OBS_REPEAT_COUNT_OVERRIDE"])
    if os.environ.get("OBS_SEED_BASE_OVERRIDE"):
        cfg["randomization_seed"] = int(os.environ["OBS_SEED_BASE_OVERRIDE"])
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("compute/communication overlap benchmark requires exactly eight ranks")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    output_dir = Path(args.output_dir)
    raw_jsonl = output_dir / "compute_comm_raw.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "DONE",
            "compute_comm_raw.csv",
            "compute_comm_raw.jsonl",
            "compute_comm_summary.csv",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()
        (output_dir / "compute_comm_config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
    dist.barrier()

    locations = {"sender": 0, "receiver": 1, "disjoint": 2}
    work = [
        (case, location, compute_rank)
        for case in compute_catalog()
        for location, compute_rank in locations.items()
    ]
    rows: list[dict] = []
    all_valid = True
    for repeat in range(1, int(cfg["repetitions"]) + 1):
        random.Random(int(cfg["randomization_seed"]) + repeat).shuffle(work)
        for case, location, compute_rank in work:
            runner = OverlapRunner(case, compute_rank, cfg, device)
            orders = (
                ("comm", "compute", "concurrent"),
                ("concurrent", "compute", "comm"),
                ("compute", "comm", "concurrent"),
            )
            order = orders[(repeat - 1) % len(orders)]
            measurements: dict[str, tuple[float, bool, bool]] = {}
            for mode in order:
                torch.cuda.nvtx.range_push(
                    f"overlap={case.case_id};location={location};mode={mode};repeat={repeat}"
                )
                try:
                    measurements[mode] = runner.measure(mode)
                finally:
                    torch.cuda.nvtx.range_pop()
            all_valid = all_valid and measurements["comm"][1]
            all_valid = all_valid and measurements["concurrent"][1]
            all_valid = all_valid and measurements["compute"][2]
            all_valid = all_valid and measurements["concurrent"][2]
            if rank == 0:
                record = make_record(case, location, repeat, cfg, measurements)
                rows.append(record)
                with raw_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "FORGEPIPE_COMPUTE_COMM_RESULT " + json.dumps(record, sort_keys=True),
                    flush=True,
                )
            del runner
            torch.cuda.empty_cache()
            dist.barrier()

    all_valid = reduce_valid(all_valid, device)
    if rank == 0:
        write_csv(output_dir / "compute_comm_raw.csv", rows)
        write_csv(output_dir / "compute_comm_summary.csv", summarize(rows))
        if all_valid:
            (output_dir / "DONE").touch()
    dist.barrier()
    dist.destroy_process_group()
    if not all_valid:
        raise RuntimeError("compute/communication payload validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
