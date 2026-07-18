from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GlobalInterval:
    elapsed_ms: float
    launch_skew_us: float


def synchronized_start_ns(device: torch.device, delay_ms: float) -> int:
    """Release same-node ranks against one shared monotonic-clock deadline."""
    if delay_ms <= 0:
        raise ValueError("delay_ms must be positive")
    target = torch.zeros(1, dtype=torch.int64, device=device)
    if dist.get_rank() == 0:
        target.fill_(time.perf_counter_ns() + int(delay_ms * 1_000_000))
    dist.broadcast(target, src=0)
    target_ns = int(target.item())
    while time.perf_counter_ns() < target_ns:
        pass
    return time.perf_counter_ns()


def finish_global_interval(start_ns: int, device: torch.device) -> GlobalInterval:
    end_ns = time.perf_counter_ns()
    start_min = torch.tensor([start_ns], dtype=torch.int64, device=device)
    start_max = start_min.clone()
    end_max = torch.tensor([end_ns], dtype=torch.int64, device=device)
    dist.all_reduce(start_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(start_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(end_max, op=dist.ReduceOp.MAX)
    return GlobalInterval(
        elapsed_ms=(int(end_max.item()) - int(start_min.item())) / 1_000_000.0,
        launch_skew_us=(int(start_max.item()) - int(start_min.item())) / 1_000.0,
    )
