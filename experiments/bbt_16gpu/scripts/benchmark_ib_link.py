#!/usr/bin/env python3
"""Measure NCCL collectives and PP-style P2P traffic across two GPU nodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import time
from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=8)
    parser.add_argument("--target-mib", type=int, default=2048)
    parser.add_argument("--min-iters", type=int, default=12)
    parser.add_argument("--max-iters", type=int, default=200)
    return parser.parse_args()


def iteration_count(size_mib: int, args: argparse.Namespace) -> int:
    return max(args.min_iters, min(args.max_iters, math.ceil(args.target_mib / size_mib)))


def barrier() -> None:
    dist.barrier()
    torch.cuda.synchronize()


def timed_ms(body: Callable[[], None], device: torch.device) -> float:
    barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    body()
    end.record()
    end.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def assert_distributed(condition: bool, message: str, device: torch.device) -> None:
    valid = torch.tensor([int(condition)], dtype=torch.int32, device=device)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid.item()):
        raise RuntimeError(message)


def p2p_ops(rank: int, send: torch.Tensor, recv: torch.Tensor, bidirectional: bool):
    peer = 1 - rank
    if bidirectional:
        return [
            dist.P2POp(dist.isend, send, peer),
            dist.P2POp(dist.irecv, recv, peer),
        ]
    if rank == 0:
        return [dist.P2POp(dist.isend, send, peer)]
    return [dist.P2POp(dist.irecv, recv, peer)]


def run_p2p(
    *,
    rank: int,
    size_bytes: int,
    iterations: int,
    warmup_iters: int,
    bidirectional: bool,
    device: torch.device,
) -> float:
    elements = size_bytes // 2
    send = torch.full((elements,), rank + 1.0, dtype=torch.bfloat16, device=device)
    recv = torch.full((elements,), -1.0, dtype=torch.bfloat16, device=device)

    def run(count: int) -> None:
        for _ in range(count):
            for work in dist.batch_isend_irecv(p2p_ops(rank, send, recv, bidirectional)):
                work.wait()

    run(warmup_iters)
    torch.cuda.synchronize()
    recv.fill_(-1.0)
    elapsed_ms = timed_ms(lambda: run(iterations), device)
    torch.cuda.synchronize()
    expected = 2.0 if rank == 0 else 1.0
    valid = rank == 0 if not bidirectional else True
    if bidirectional or rank == 1:
        valid = bool(torch.all(recv == expected).item())
    assert_distributed(valid, "P2P payload validation failed", device)
    return elapsed_ms


def run_all_reduce(
    *, size_bytes: int, iterations: int, warmup_iters: int, device: torch.device
) -> float:
    elements = size_bytes // 2
    tensor = torch.zeros((elements,), dtype=torch.bfloat16, device=device)

    def run(count: int) -> None:
        for _ in range(count):
            dist.all_reduce(tensor)

    run(warmup_iters)
    torch.cuda.synchronize()
    tensor.zero_()
    elapsed_ms = timed_ms(lambda: run(iterations), device)
    valid = bool(torch.all(tensor == 0).item())
    assert_distributed(valid, "all-reduce payload validation failed", device)
    return elapsed_ms


def emit(record: dict, rank: int, records: list[dict]) -> None:
    if rank == 0:
        records.append(record)
        print("FORGEPIPE_IB_RESULT " + json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("This benchmark requires exactly two distributed ranks")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    records: list[dict] = []

    if rank == 0:
        metadata = {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "hostname": socket.gethostname(),
            "nccl": torch.cuda.nccl.version(),
            "pytorch": torch.__version__,
            "timestamp_unix": time.time(),
            "world_size": dist.get_world_size(),
        }
        print("FORGEPIPE_IB_METADATA " + json.dumps(metadata, sort_keys=True), flush=True)

    operations = ("p2p_oneway", "p2p_bidirectional", "all_reduce")
    for size_mib in args.sizes_mib:
        size_bytes = size_mib * 1024 * 1024
        iterations = iteration_count(size_mib, args)
        for operation in operations:
            for repeat in range(args.repeats):
                range_name = f"{operation}_mib{size_mib}_repeat{repeat}"
                torch.cuda.nvtx.range_push(range_name)
                try:
                    if operation == "all_reduce":
                        elapsed_ms = run_all_reduce(
                            size_bytes=size_bytes,
                            iterations=iterations,
                            warmup_iters=args.warmup_iters,
                            device=device,
                        )
                        transferred_bytes = size_bytes * iterations
                    else:
                        bidirectional = operation == "p2p_bidirectional"
                        elapsed_ms = run_p2p(
                            rank=rank,
                            size_bytes=size_bytes,
                            iterations=iterations,
                            warmup_iters=args.warmup_iters,
                            bidirectional=bidirectional,
                            device=device,
                        )
                        transferred_bytes = size_bytes * iterations * (2 if bidirectional else 1)
                finally:
                    torch.cuda.nvtx.range_pop()

                seconds = elapsed_ms / 1000.0
                aggregate_gbps = transferred_bytes / seconds / 1e9
                emit(
                    {
                        "aggregate_gbps": aggregate_gbps,
                        "elapsed_ms": elapsed_ms,
                        "iterations": iterations,
                        "operation": operation,
                        "payload_mib": size_mib,
                        "repeat": repeat,
                    },
                    rank,
                    records,
                )

    if rank == 0:
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for record in records:
            grouped[(record["operation"], record["payload_mib"])].append(
                record["aggregate_gbps"]
            )
        summary = []
        for (operation, payload_mib), values in sorted(grouped.items()):
            summary.append(
                {
                    "max_gbps": max(values),
                    "mean_gbps": statistics.fmean(values),
                    "median_gbps": statistics.median(values),
                    "min_gbps": min(values),
                    "operation": operation,
                    "payload_mib": payload_mib,
                    "stdev_gbps": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
            )
        print("FORGEPIPE_IB_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
