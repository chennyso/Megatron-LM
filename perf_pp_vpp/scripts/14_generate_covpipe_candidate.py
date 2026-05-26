#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_TRANSFORMS = {
    "move_boundary",
    "reshape_vpp_chunk",
    "swap_ready_chunks",
    "delay_wgrad",
    "set_recompute",
}


@dataclass
class CandidateContext:
    pp: int
    vpp: int
    num_layers: int
    chunks: list[list[int]]
    annotations: dict[str, Any]


def build_uniform_chunks(pp: int, vpp: int, num_layers: int) -> list[list[int]]:
    slots = pp * vpp
    if num_layers % slots != 0:
        raise ValueError(f"uniform seed requires num_layers divisible by pp*vpp, got {num_layers=} {pp=} {vpp=}")
    per_chunk = num_layers // slots
    return [[per_chunk for _ in range(vpp)] for _ in range(pp)]


def verify_chunks(ctx: CandidateContext) -> None:
    if len(ctx.chunks) != ctx.pp:
        raise ValueError(f"expected {ctx.pp} pp ranks, got {len(ctx.chunks)}")
    for rank, row in enumerate(ctx.chunks):
        if len(row) != ctx.vpp:
            raise ValueError(f"rank {rank} expected {ctx.vpp} vpp chunks, got {len(row)}")
        if any(x <= 0 for x in row):
            raise ValueError(f"rank {rank} has non-positive chunk sizes: {row}")
    total = sum(sum(row) for row in ctx.chunks)
    if total != ctx.num_layers:
        raise ValueError(f"candidate changed total layers: expected {ctx.num_layers}, got {total}")


def move_boundary(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    stage = int(transform["stage"])
    delta = int(transform["delta_layers"])
    donor = stage + 1 if delta > 0 else stage
    receiver = stage if delta > 0 else stage + 1
    magnitude = abs(delta)
    if not (0 <= stage < ctx.pp - 1):
        raise ValueError(f"move_boundary stage out of range: {stage}")
    if magnitude == 0:
        return
    donor_chunks = ctx.chunks[donor]
    receiver_chunks = ctx.chunks[receiver]
    for _ in range(magnitude):
        donor_idx = min(range(ctx.vpp), key=lambda idx: donor_chunks[idx])
        receiver_idx = max(range(ctx.vpp), key=lambda idx: receiver_chunks[idx])
        if donor_chunks[donor_idx] <= 1:
            raise ValueError(
                f"move_boundary would make chunk non-positive: donor rank={donor} chunks={donor_chunks}"
            )
        donor_chunks[donor_idx] -= 1
        receiver_chunks[receiver_idx] += 1


def reshape_vpp_chunk(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    rank = int(transform["rank"])
    chunks = [int(x) for x in transform["chunks"]]
    if not (0 <= rank < ctx.pp):
        raise ValueError(f"reshape_vpp_chunk rank out of range: {rank}")
    if len(chunks) != ctx.vpp:
        raise ValueError(f"reshape_vpp_chunk needs {ctx.vpp} chunks, got {len(chunks)}")
    if sum(chunks) != sum(ctx.chunks[rank]):
        raise ValueError(
            f"reshape_vpp_chunk must preserve rank total layers: rank={rank} "
            f"old={ctx.chunks[rank]} new={chunks}"
        )
    ctx.chunks[rank] = chunks


def swap_ready_chunks(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    rank = int(transform["rank"])
    chunk_a = int(transform["chunk_a"])
    chunk_b = int(transform["chunk_b"])
    if not (0 <= rank < ctx.pp):
        raise ValueError(f"swap_ready_chunks rank out of range: {rank}")
    if not (0 <= chunk_a < ctx.vpp and 0 <= chunk_b < ctx.vpp):
        raise ValueError(f"swap_ready_chunks chunk indices out of range: {chunk_a}, {chunk_b}")
    row = ctx.chunks[rank]
    row[chunk_a], row[chunk_b] = row[chunk_b], row[chunk_a]


def delay_wgrad(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    placements = deepcopy(ctx.annotations.setdefault("wgrad_placements", []))
    placements.append(
        {
            "rank": int(transform["rank"]),
            "chunk": int(transform["chunk"]),
            "target": transform["target"],
        }
    )
    ctx.annotations["wgrad_placements"] = placements
    ctx.annotations["delay_wgrad_compute"] = True


def set_recompute(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    ctx.annotations["recompute"] = {
        "granularity": transform.get("granularity", "full"),
        "method": transform.get("method", "uniform"),
        "num_layers": int(transform.get("num_layers", 1)),
        "modules": transform.get("modules", ""),
    }


def apply_transform(ctx: CandidateContext, transform: dict[str, Any]) -> None:
    kind = transform["kind"]
    if kind not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"unsupported transform kind: {kind}")
    if kind == "move_boundary":
        move_boundary(ctx, transform)
    elif kind == "reshape_vpp_chunk":
        reshape_vpp_chunk(ctx, transform)
    elif kind == "swap_ready_chunks":
        swap_ready_chunks(ctx, transform)
    elif kind == "delay_wgrad":
        delay_wgrad(ctx, transform)
    elif kind == "set_recompute":
        set_recompute(ctx, transform)
    else:
        raise AssertionError(f"unhandled transform kind: {kind}")
    verify_chunks(ctx)


def flatten_chunks(chunks: list[list[int]]) -> list[int]:
    flattened: list[int] = []
    for vp in range(len(chunks[0])):
        for pp_rank in range(len(chunks)):
            flattened.append(chunks[pp_rank][vp])
    return flattened


def counts_to_layout(counts: list[int]) -> str:
    stages: list[str] = []
    for idx, count in enumerate(counts):
        piece = ""
        if idx == 0:
            piece += "E"
        piece += "t" * count
        if idx == len(counts) - 1:
            piece += "L"
        stages.append(piece)
    return "|".join(stages)


def compile_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    pp = int(spec["pp"])
    vpp = int(spec["vpp"])
    num_layers = int(spec["num_layers"])
    seed = spec.get("seed", "uniform")
    if seed != "uniform":
        raise ValueError(f"unsupported seed: {seed}")
    ctx = CandidateContext(
        pp=pp,
        vpp=vpp,
        num_layers=num_layers,
        chunks=build_uniform_chunks(pp, vpp, num_layers),
        annotations={},
    )
    for transform in spec.get("transforms", []):
        apply_transform(ctx, transform)
    counts = flatten_chunks(ctx.chunks)
    layout = counts_to_layout(counts)
    env = {
        "PERF_PIPELINE_MODEL_PARALLEL_LAYOUT": layout,
    }
    if ctx.annotations.get("delay_wgrad_compute") and str(__import__("os").environ.get("PERF_ENABLE_DELAY_WGRAD_COVPIPE", "0")) == "1":
        env["PERF_DELAY_WGRAD_COMPUTE"] = "1"
    recompute = ctx.annotations.get("recompute")
    if recompute:
        env["PERF_RECOMPUTE_GRANULARITY"] = recompute["granularity"]
        env["PERF_RECOMPUTE_METHOD"] = recompute["method"]
        env["PERF_RECOMPUTE_NUM_LAYERS"] = str(recompute["num_layers"])
        if recompute["modules"]:
            env["PERF_RECOMPUTE_MODULES"] = recompute["modules"]
    return {
        "name": spec.get("name", "covpipe_candidate"),
        "pp": pp,
        "vpp": vpp,
        "num_layers": num_layers,
        "seed": seed,
        "transforms": spec.get("transforms", []),
        "counts_per_pp_rank": [sum(row) for row in ctx.chunks],
        "chunks_per_pp_rank": ctx.chunks,
        "flattened_counts": counts,
        "layout": layout,
        "env": env,
        "annotations": ctx.annotations,
    }


def default_example() -> dict[str, Any]:
    return {
        "name": "q32_pp8_vpp4_covpipe_boundary_shield",
        "pp": 8,
        "vpp": 4,
        "num_layers": 64,
        "seed": "uniform",
        "transforms": [
            {"kind": "move_boundary", "stage": 3, "delta_layers": 1},
            {"kind": "reshape_vpp_chunk", "rank": 3, "chunks": [2, 2, 2, 3]},
            {"kind": "reshape_vpp_chunk", "rank": 4, "chunks": [2, 2, 2, 1]},
            {"kind": "delay_wgrad", "rank": 3, "chunk": 0, "target": "recv_wait"},
            {"kind": "delay_wgrad", "rank": 4, "chunk": 0, "target": "send_inflight"},
        ],
    }


def load_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_example()
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile COVPipe-style verified transformations to Megatron env/layout.")
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--write-example", type=Path)
    parser.add_argument("--print-env", action="store_true")
    args = parser.parse_args()

    if args.write_example:
        args.write_example.write_text(
            json.dumps(default_example(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return

    compiled = compile_candidate(load_spec(args.candidate_json))
    if args.print_env:
        for key, value in compiled["env"].items():
            print(f"{key}={value}")
        return
    print(json.dumps(compiled, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
