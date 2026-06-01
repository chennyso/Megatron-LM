#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import pipelining as pp


class Block(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x + residual


class BenchModel(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(hidden_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                pp.pipe_split()
        return x


@dataclass(frozen=True)
class BenchConfig:
    schedule: str
    num_layers: int
    hidden_size: int
    seq_len: int
    micro_batch_size: int
    microbatches: int
    warmup: int
    iters: int


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=["gpipe", "dualpipev"], required=True)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    return BenchConfig(
        schedule=args.schedule,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        microbatches=args.microbatches,
        warmup=args.warmup,
        iters=args.iters,
    )


def init_dist() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_schedule(name: str):
    if name == "gpipe":
        return pp.ScheduleGPipe
    if name == "dualpipev":
        return pp.ScheduleDualPipeV
    raise ValueError(name)


def main() -> None:
    cfg = parse_args()
    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    model = BenchModel(cfg.num_layers, cfg.hidden_size).eval()
    sample = torch.randn(cfg.micro_batch_size, cfg.seq_len, cfg.hidden_size)
    pipe = pp.Pipe.from_tracing(model, example_args=(sample,))
    stage = pipe.build_stage(rank, device, dist.group.WORLD)
    stage = stage.to(dtype=torch.float16)

    loss_fn = nn.MSELoss()
    schedule_cls = build_schedule(cfg.schedule)
    schedule = schedule_cls(stage, cfg.microbatches, loss_fn=loss_fn, scale_grads=True)

    batch = torch.randn(
        cfg.micro_batch_size * cfg.microbatches,
        cfg.seq_len,
        cfg.hidden_size,
        device=device,
        dtype=torch.float16,
    )
    target = torch.randn_like(batch)

    total_params = sum(p.numel() for p in model.parameters())
    total_tokens = cfg.micro_batch_size * cfg.microbatches * cfg.seq_len

    for _ in range(cfg.warmup):
        schedule.step(batch, target=target, return_outputs=False)
        dist.barrier()

    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(cfg.iters):
        schedule.step(batch, target=target, return_outputs=False)
        dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - start

    if rank == 0:
        avg_s = elapsed / cfg.iters
        tflops = (6.0 * total_params * total_tokens) / (avg_s * 1e12)
        print("=" * 60)
        print(f"Schedule: {cfg.schedule}")
        print(f"Layers: {cfg.num_layers}, hidden: {cfg.hidden_size}, seq: {cfg.seq_len}")
        print(f"Micro batch: {cfg.micro_batch_size}, microbatches: {cfg.microbatches}")
        print(f"Params: {total_params}")
        print(f"Avg step: {avg_s * 1000:.2f} ms")
        print(f"Throughput: {tflops:.2f} TFLOP/s")
        print("=" * 60)


if __name__ == "__main__":
    main()
