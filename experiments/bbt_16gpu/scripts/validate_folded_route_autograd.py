#!/usr/bin/env python3
"""Validate standard and folded PP routes with real distributed autograd shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core.pipeline_parallel.route_policy import PipelineRoute
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--atol", type=float, default=2e-5)
    return parser.parse_args()


def build_model(hidden_size: int, seed: int, device: torch.device) -> nn.ModuleList:
    torch.manual_seed(seed)
    blocks = []
    for stage_id in range(4):
        linear = nn.Linear(hidden_size, hidden_size)
        block = linear if stage_id == 3 else nn.Sequential(linear, nn.GELU())
        blocks.append(block.to(device))
    return nn.ModuleList(blocks)


def make_data(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    inputs = torch.randn(args.batch_size, args.hidden_size, generator=generator)
    targets = torch.randn(args.batch_size, args.hidden_size, generator=generator)
    return inputs.to(device), targets.to(device)


def reference_pass(
    model: nn.ModuleList,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    microbatches: int,
) -> tuple[float, list[torch.Tensor]]:
    input_chunks = inputs.chunk(microbatches)
    target_chunks = targets.chunk(microbatches)
    losses = []
    predictions = []
    for input_chunk, target_chunk in zip(input_chunks, target_chunks):
        activation = input_chunk
        for block in model:
            activation = block(activation)
        predictions.append(activation.detach())
        loss = nn.functional.mse_loss(activation, target_chunk) / microbatches
        loss.backward()
        losses.append(loss.detach())
    return float(torch.stack(losses).sum().item()), predictions


def p2p_send(tensor: torch.Tensor, peer: int) -> None:
    dist.send(tensor.detach().contiguous(), dst=peer)


def p2p_recv(shape: torch.Size, peer: int, device: torch.device) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float32, device=device)
    dist.recv(tensor, src=peer)
    return tensor


def assert_all_ranks(condition: bool, message: str, device: torch.device) -> None:
    valid = torch.tensor([int(condition)], dtype=torch.int32, device=device)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid.item()):
        raise RuntimeError(message)


def execute_route(
    *,
    route: PipelineRoute,
    reference: nn.ModuleList,
    reference_loss: float,
    reference_predictions: list[torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    microbatches: int,
    rank: int,
    device: torch.device,
    atol: float,
) -> dict:
    owned = {
        stage_id: copy.deepcopy(reference[stage_id]).to(device)
        for stage_id, stage in enumerate(route.stages)
        if stage.physical_rank == rank
    }
    for block in owned.values():
        block.zero_grad(set_to_none=True)

    input_chunks = inputs.chunk(microbatches)
    target_chunks = targets.chunk(microbatches)
    forward_sends = 0
    backward_sends = 0
    local_transitions = 0
    distributed_loss = 0.0
    prediction_max_abs = 0.0

    for microbatch_id, (input_chunk, target_chunk) in enumerate(
        zip(input_chunks, target_chunks)
    ):
        stage_inputs: dict[int, torch.Tensor] = {}
        stage_outputs: dict[int, torch.Tensor] = {}
        range_name = f"route_{microbatch_id}_forward"
        torch.cuda.nvtx.range_push(range_name)
        try:
            for stage_id, stage in enumerate(route.stages):
                if stage.physical_rank != rank:
                    continue
                predecessor = route.predecessor(stage)
                if predecessor is None:
                    stage_input = input_chunk.detach().requires_grad_(True)
                elif predecessor.physical_rank == rank:
                    stage_input = stage_outputs[stage_id - 1].detach().requires_grad_(True)
                    local_transitions += 1
                else:
                    stage_input = p2p_recv(input_chunk.shape, predecessor.physical_rank, device)
                    stage_input.requires_grad_(True)
                stage_inputs[stage_id] = stage_input
                stage_output = owned[stage_id](stage_input)
                stage_outputs[stage_id] = stage_output
                successor = route.successor(stage)
                if successor is not None and successor.physical_rank != rank:
                    p2p_send(stage_output, successor.physical_rank)
                    forward_sends += 1
        finally:
            torch.cuda.nvtx.range_pop()

        final_stage_id = len(route.stages) - 1
        final_owner = route.stages[-1].physical_rank
        if rank == final_owner:
            prediction = stage_outputs[final_stage_id]
            prediction_max_abs = max(
                prediction_max_abs,
                float(
                    (prediction.detach() - reference_predictions[microbatch_id])
                    .abs()
                    .max()
                    .item()
                ),
            )
            loss = nn.functional.mse_loss(prediction, target_chunk) / microbatches
            loss.backward()
            distributed_loss += float(loss.detach().item())

        torch.cuda.nvtx.range_push(f"route_{microbatch_id}_backward")
        try:
            for stage_id in range(final_stage_id, -1, -1):
                stage = route.stages[stage_id]
                if stage.physical_rank != rank:
                    continue
                if stage_id != final_stage_id:
                    successor = route.successor(stage)
                    if successor is None:
                        raise RuntimeError("non-final stage is missing a successor")
                    if successor.physical_rank == rank:
                        output_grad = stage_inputs[stage_id + 1].grad
                    else:
                        output_grad = p2p_recv(input_chunk.shape, successor.physical_rank, device)
                    stage_outputs[stage_id].backward(output_grad)
                predecessor = route.predecessor(stage)
                if predecessor is not None and predecessor.physical_rank != rank:
                    input_grad = stage_inputs[stage_id].grad
                    if input_grad is None:
                        raise RuntimeError(f"missing input gradient for stage {stage_id}")
                    p2p_send(input_grad, predecessor.physical_rank)
                    backward_sends += 1
        finally:
            torch.cuda.nvtx.range_pop()

    loss_tensor = torch.tensor([distributed_loss], dtype=torch.float64, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    prediction_tensor = torch.tensor([prediction_max_abs], dtype=torch.float64, device=device)
    dist.all_reduce(prediction_tensor, op=dist.ReduceOp.MAX)

    grad_max_abs = 0.0
    for stage_id, block in owned.items():
        for parameter, reference_parameter in zip(
            block.parameters(), reference[stage_id].parameters()
        ):
            if parameter.grad is None or reference_parameter.grad is None:
                raise RuntimeError(f"missing parameter gradient for stage {stage_id}")
            grad_max_abs = max(
                grad_max_abs,
                float((parameter.grad - reference_parameter.grad).abs().max().item()),
            )
    grad_tensor = torch.tensor([grad_max_abs], dtype=torch.float64, device=device)
    dist.all_reduce(grad_tensor, op=dist.ReduceOp.MAX)

    counts = torch.tensor(
        [forward_sends, backward_sends, local_transitions], dtype=torch.int64, device=device
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    remote_edges = len(route.message_signatures("forward", "activation"))
    expected_remote_sends = remote_edges * microbatches
    expected_local_transitions = (len(route.forward_edges) - remote_edges) * microbatches
    loss_abs = abs(float(loss_tensor.item()) - reference_loss)
    valid = (
        loss_abs <= atol
        and float(prediction_tensor.item()) <= atol
        and float(grad_tensor.item()) <= atol
        and int(counts[0].item()) == expected_remote_sends
        and int(counts[1].item()) == expected_remote_sends
        and int(counts[2].item()) == expected_local_transitions
    )
    assert_all_ranks(valid, f"route validation failed for {route.stages}", device)
    return {
        "backward_sends": int(counts[1].item()),
        "forward_sends": int(counts[0].item()),
        "grad_max_abs": float(grad_tensor.item()),
        "local_transitions": int(counts[2].item()),
        "loss_abs": loss_abs,
        "prediction_max_abs": float(prediction_tensor.item()),
        "remote_edges_per_direction_per_microbatch": remote_edges,
    }


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("This pilot requires exactly two distributed ranks")
    if args.batch_size % args.microbatches != 0:
        raise ValueError("batch_size must be divisible by microbatches")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    hostnames = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, socket.gethostname())

    routes = {
        "standard": PipelineRoute.standard(pipeline_size=2, virtual_chunks=2),
        "folded": PipelineRoute.folded(pipeline_size=2, virtual_chunks=2),
    }
    results = {}
    for name, route in routes.items():
        route.verify()
        reference = build_model(args.hidden_size, args.seed, device)
        inputs, targets = make_data(args, device)
        reference_loss, reference_predictions = reference_pass(
            reference, inputs, targets, args.microbatches
        )
        dist.barrier()
        torch.cuda.nvtx.range_push(f"validate_{name}")
        try:
            results[name] = execute_route(
                route=route,
                reference=reference,
                reference_loss=reference_loss,
                reference_predictions=reference_predictions,
                inputs=inputs,
                targets=targets,
                microbatches=args.microbatches,
                rank=rank,
                device=device,
                atol=args.atol,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        dist.barrier()

    if rank == 0:
        payload = {
            "cuda": torch.version.cuda,
            "hostnames": hostnames,
            "nccl": torch.cuda.nccl.version(),
            "pytorch": torch.__version__,
            "results": results,
            "world_size": dist.get_world_size(),
        }
        print("FORGEPIPE_ROUTE_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
