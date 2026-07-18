#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel.p2p_communication import (
    PipelineP2PPeers,
    _batched_p2p_ops,
)
from megatron.core.pipeline_parallel.route_policy import PipelineEdge, PipelineRoute


def message_value(hierarchy_factor: int, edge_id: int, direction: str) -> float:
    direction_offset = 0.5 if direction == "backward" else 0.0
    return float(hierarchy_factor * 1000 + edge_id) + direction_offset


def exchange_edge(
    edge: PipelineEdge,
    *,
    direction: str,
    hierarchy_factor: int,
    numel: int,
) -> tuple[bool, float]:
    if direction == "forward":
        source_rank = edge.source.physical_rank
        target_rank = edge.target.physical_rank
    elif direction == "backward":
        source_rank = edge.target.physical_rank
        target_rank = edge.source.physical_rank
    else:
        raise ValueError(f"unknown direction: {direction}")

    rank = dist.get_rank()
    expected = message_value(hierarchy_factor, edge.edge_id, direction)
    send_tensor = None
    recv_tensor = None
    peers = PipelineP2PPeers()
    if rank == source_rank:
        send_tensor = torch.full((numel,), expected, device="cuda", dtype=torch.float32)
        if direction == "forward":
            peers = PipelineP2PPeers(send_forward=target_rank)
        else:
            peers = PipelineP2PPeers(send_backward=target_rank)
    elif rank == target_rank:
        recv_tensor = torch.empty((numel,), device="cuda", dtype=torch.float32)
        if direction == "forward":
            peers = PipelineP2PPeers(recv_forward=source_rank)
        else:
            peers = PipelineP2PPeers(recv_backward=source_rank)

    started = time.perf_counter()
    if direction == "forward":
        requests = _batched_p2p_ops(
            tensor_send_prev=None,
            tensor_recv_prev=recv_tensor,
            tensor_send_next=send_tensor,
            tensor_recv_next=None,
            group=dist.group.WORLD,
            prev_pipeline_rank=(rank - 1) % dist.get_world_size(),
            next_pipeline_rank=(rank + 1) % dist.get_world_size(),
            peers=peers,
        )
    else:
        requests = _batched_p2p_ops(
            tensor_send_prev=send_tensor,
            tensor_recv_prev=None,
            tensor_send_next=None,
            tensor_recv_next=recv_tensor,
            group=dist.group.WORLD,
            prev_pipeline_rank=(rank - 1) % dist.get_world_size(),
            next_pipeline_rank=(rank + 1) % dist.get_world_size(),
            peers=peers,
        )
    for request in requests:
        request.wait()
    dist.barrier()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    valid = True
    if recv_tensor is not None:
        observed = torch.stack((recv_tensor[0], recv_tensor[-1], recv_tensor.mean()))
        reference = torch.full((3,), expected, device="cuda", dtype=torch.float32)
        valid = bool(torch.equal(observed, reference))
    return valid, elapsed_ms


def run_route(
    *,
    pipeline_size: int,
    virtual_chunks: int,
    hierarchy_factor: int,
    numel: int,
) -> dict:
    node_by_rank = tuple(
        "domain-0" if rank < pipeline_size // 2 else "domain-1"
        for rank in range(pipeline_size)
    )
    route = PipelineRoute.topology_segmented(
        pipeline_size,
        virtual_chunks,
        node_by_rank,
        hierarchy_factor,
        rotate_endpoints=True,
    )
    route.verify()

    failures = 0
    elapsed_ms = 0.0
    remote_edges = 0
    local_edges = 0
    for direction in ("forward", "backward"):
        for edge in route.forward_edges:
            if edge.source.physical_rank == edge.target.physical_rank:
                local_edges += 1
                continue
            remote_edges += 1
            valid, edge_ms = exchange_edge(
                edge,
                direction=direction,
                hierarchy_factor=hierarchy_factor,
                numel=numel,
            )
            elapsed_ms += edge_ms
            failures += int(not valid)

    failure_tensor = torch.tensor(failures, device="cuda", dtype=torch.int32)
    dist.all_reduce(failure_tensor, op=dist.ReduceOp.SUM)
    max_elapsed = torch.tensor(elapsed_ms, device="cuda", dtype=torch.float64)
    dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
    return {
        "hierarchy_factor": hierarchy_factor,
        "transition_budget": len(route.cross_node_edges(node_by_rank)),
        "remote_edge_directions": remote_edges,
        "local_edge_directions": local_edges,
        "payload_bytes": numel * 4,
        "failure_count": int(failure_tensor.item()),
        "max_rank_elapsed_ms": float(max_elapsed.item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-size", type=int, default=8)
    parser.add_argument("--virtual-chunks", type=int, default=8)
    parser.add_argument("--hierarchy-factor", type=int, action="append", required=True)
    parser.add_argument("--numel", type=int, default=262144)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != args.pipeline_size:
        raise ValueError(
            f"world size {dist.get_world_size()} must equal pipeline size {args.pipeline_size}"
        )
    warmup = torch.ones(1, device="cuda")
    dist.all_reduce(warmup)

    results = [
        run_route(
            pipeline_size=args.pipeline_size,
            virtual_chunks=args.virtual_chunks,
            hierarchy_factor=hierarchy_factor,
            numel=args.numel,
        )
        for hierarchy_factor in args.hierarchy_factor
    ]
    if dist.get_rank() == 0:
        print(json.dumps({"route_peer_smoke": results}, indent=2), flush=True)
    failure_count = sum(item["failure_count"] for item in results)
    dist.destroy_process_group()
    return int(failure_count != 0)


if __name__ == "__main__":
    raise SystemExit(main())
