#!/usr/bin/env python3
"""Measure non-additive NCCL P2P interference on one eight-GPU node."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from single_node_timing import finish_global_interval, synchronized_start_ns


Edge = tuple[int, int]


@dataclass(frozen=True)
class Motif:
    motif_id: str
    route_class: str
    edges: tuple[Edge, ...]
    primitive_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def motif_catalog() -> tuple[list[Motif], list[Motif]]:
    one_way_edges = {
        "oneway_0_1": ((0, 1),),
        "oneway_2_3": ((2, 3),),
        "oneway_4_5": ((4, 5),),
        "oneway_6_7": ((6, 7),),
        "oneway_2_1": ((2, 1),),
        "oneway_3_1": ((3, 1),),
        "oneway_4_1": ((4, 1),),
    }
    bidirectional_edges = {
        "bidir_0_1": ((0, 1), (1, 0)),
        "bidir_2_3": ((2, 3), (3, 2)),
        "bidir_4_5": ((4, 5), (5, 4)),
        "bidir_6_7": ((6, 7), (7, 6)),
    }
    primitives = [
        Motif(motif_id, "isolated_oneway", edges, (motif_id,))
        for motif_id, edges in one_way_edges.items()
    ] + [
        Motif(motif_id, "isolated_bidirectional", edges, (motif_id,))
        for motif_id, edges in bidirectional_edges.items()
    ]
    concurrent = [
        Motif(
            "disjoint_oneway_2",
            "endpoint_disjoint",
            ((0, 1), (2, 3)),
            ("oneway_0_1", "oneway_2_3"),
        ),
        Motif(
            "disjoint_oneway_3",
            "endpoint_disjoint",
            ((0, 1), (2, 3), (4, 5)),
            ("oneway_0_1", "oneway_2_3", "oneway_4_5"),
        ),
        Motif(
            "disjoint_oneway_4",
            "endpoint_disjoint",
            ((0, 1), (2, 3), (4, 5), (6, 7)),
            ("oneway_0_1", "oneway_2_3", "oneway_4_5", "oneway_6_7"),
        ),
        Motif(
            "shared_sink_2",
            "shared_endpoint",
            ((0, 1), (2, 1)),
            ("oneway_0_1", "oneway_2_1"),
        ),
        Motif(
            "shared_sink_3",
            "shared_endpoint",
            ((0, 1), (2, 1), (3, 1)),
            ("oneway_0_1", "oneway_2_1", "oneway_3_1"),
        ),
        Motif(
            "shared_sink_4",
            "shared_endpoint",
            ((0, 1), (2, 1), (3, 1), (4, 1)),
            ("oneway_0_1", "oneway_2_1", "oneway_3_1", "oneway_4_1"),
        ),
        Motif(
            "disjoint_bidir_2",
            "endpoint_disjoint_bidirectional",
            ((0, 1), (1, 0), (2, 3), (3, 2)),
            ("bidir_0_1", "bidir_2_3"),
        ),
        Motif(
            "disjoint_bidir_3",
            "endpoint_disjoint_bidirectional",
            ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)),
            ("bidir_0_1", "bidir_2_3", "bidir_4_5"),
        ),
        Motif(
            "disjoint_bidir_4",
            "endpoint_disjoint_bidirectional",
            (
                (0, 1),
                (1, 0),
                (2, 3),
                (3, 2),
                (4, 5),
                (5, 4),
                (6, 7),
                (7, 6),
            ),
            ("bidir_0_1", "bidir_2_3", "bidir_4_5", "bidir_6_7"),
        ),
    ]
    return primitives, concurrent


def iteration_count(size_bytes: int, cfg: dict) -> int:
    target = int(cfg["target_bytes_per_flow"])
    return max(
        int(cfg["min_iterations"]),
        min(int(cfg["max_iterations"]), math.ceil(target / size_bytes)),
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["motif_id"], row["size_bytes"]), []).append(row)
    summary_rows: list[dict] = []
    metrics = (
        "wall_makespan_ms",
        "launch_skew_us",
        "aggregate_decimal_gbps",
        "slowdown_vs_isolated_max",
        "ideal_max_prediction_error_pct",
        "serial_sum_prediction_error_pct",
    )
    for (motif_id, size_bytes), bucket in sorted(grouped.items()):
        summary = {
            "motif_id": motif_id,
            "route_class": bucket[0]["route_class"],
            "size_bytes": size_bytes,
            "concurrency": bucket[0]["concurrency"],
            "n": len(bucket),
            "payload_valid": all(row["payload_valid"] for row in bucket),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in bucket if row[metric] is not None]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            # Five process-local repeats are a shakeout statistic. Use the
            # two-sided Student-t critical value for df=4 instead of 1.96.
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


def reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def sample_positions(elements: int) -> list[int]:
    return sorted(
        {
            0,
            min(1, elements - 1),
            elements // 4,
            elements // 2,
            (3 * elements) // 4,
            max(0, elements - 2),
            elements - 1,
        }
    )


def payload_value(src: int, nonce: int, sample_index: int) -> float:
    return float(nonce * 16 + src * 2 + sample_index)


def validate_payload(
    recv_buffers: dict[int, torch.Tensor], nonce: int, device: torch.device
) -> bool:
    local_valid = True
    for src, tensor in recv_buffers.items():
        for sample_index, position in enumerate(sample_positions(tensor.numel())):
            expected = payload_value(src, nonce, sample_index)
            local_valid = local_valid and float(tensor[position].item()) == expected
    valid = torch.tensor([int(local_valid)], dtype=torch.int32, device=device)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    return bool(valid.item())


def run_motif(
    motif: Motif,
    size_bytes: int,
    iterations: int,
    warmup_iterations: int,
    payload_nonce: int,
    start_gate_delay_ms: float,
    device: torch.device,
) -> tuple[float, float, float, bool]:
    rank = dist.get_rank()
    elements = math.ceil(size_bytes / 2)
    outgoing = any(src == rank for src, _ in motif.edges)
    incoming_sources = sorted(src for src, dst in motif.edges if dst == rank)
    send_buffer = (
        torch.full(
            (elements,), payload_value(rank, payload_nonce, 0), dtype=torch.bfloat16, device=device
        )
        if outgoing
        else None
    )
    if send_buffer is not None:
        for sample_index, position in enumerate(sample_positions(elements)):
            send_buffer[position] = payload_value(rank, payload_nonce, sample_index)
    recv_buffers = {
        src: torch.full((elements,), -1.0, dtype=torch.bfloat16, device=device)
        for src in incoming_sources
    }

    def issue(count: int) -> None:
        ops: list[dist.P2POp] = []
        for src, dst in motif.edges:
            if src == rank:
                ops.append(dist.P2POp(dist.isend, send_buffer, dst))
            if dst == rank:
                ops.append(dist.P2POp(dist.irecv, recv_buffers[src], src))
        if not ops:
            return
        for _ in range(count):
            for work in dist.batch_isend_irecv(ops):
                work.wait()

    dist.barrier()
    issue(warmup_iterations)
    torch.cuda.synchronize()
    for tensor in recv_buffers.values():
        tensor.fill_(-1.0)
    dist.barrier()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_ns = synchronized_start_ns(device, start_gate_delay_ms)
    start_event.record()
    issue(iterations)
    end_event.record()
    end_event.synchronize()
    interval = finish_global_interval(start_ns, device)
    local_device_ms = float(start_event.elapsed_time(end_event))
    device_ms = reduce_max(local_device_ms, device)
    payload_valid = validate_payload(recv_buffers, payload_nonce, device)
    dist.barrier()
    return interval.elapsed_ms, device_ms, interval.launch_skew_us, payload_valid


def record_for(
    motif: Motif,
    size_bytes: int,
    iterations: int,
    repeat: int,
    wall_ms: float,
    device_ms: float,
    launch_skew_us: float,
    payload_valid: bool,
    baseline_times: dict[str, float],
) -> dict:
    isolated = [baseline_times[name] for name in motif.primitive_ids if name in baseline_times]
    isolated_max = max(isolated) if isolated else wall_ms
    isolated_sum = sum(isolated) if isolated else wall_ms
    total_bytes = len(motif.edges) * size_bytes * iterations
    return {
        "motif_id": motif.motif_id,
        "route_class": motif.route_class,
        "edges_json": json.dumps(motif.edges, separators=(",", ":")),
        "concurrency": len(motif.edges),
        "size_bytes": size_bytes,
        "iterations": iterations,
        "repeat": repeat,
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "unset"),
        "wall_makespan_ms": wall_ms,
        "max_device_ms": device_ms,
        "launch_skew_us": launch_skew_us,
        "aggregate_decimal_gbps": (total_bytes / 1e9) / (wall_ms / 1000.0),
        "isolated_max_ms": isolated_max,
        "isolated_sum_ms": isolated_sum,
        "slowdown_vs_isolated_max": wall_ms / isolated_max,
        "parallel_efficiency": isolated_sum / (wall_ms * len(isolated)) if isolated else 1.0,
        "ideal_max_prediction_error_pct": abs(wall_ms - isolated_max) / wall_ms * 100.0,
        "serial_sum_prediction_error_pct": abs(wall_ms - isolated_sum) / wall_ms * 100.0,
        "payload_valid": payload_valid,
    }


def capture_text(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout + result.stderr


def write_metadata(output_dir: Path, cfg: dict, rank_map: list[dict]) -> None:
    topology = capture_text(["nvidia-smi", "topo", "-m"])
    inventory = capture_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total",
            "--format=csv",
        ]
    )
    (output_dir / "nvidia-smi-topo.txt").write_text(topology, encoding="utf-8")
    (output_dir / "gpu-inventory.csv").write_text(inventory, encoding="utf-8")
    commit = capture_text(["git", "rev-parse", "HEAD"]).strip()
    metadata = {
        "hostname": socket.gethostname(),
        "git_commit": commit,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": dist.get_world_size(),
        "rank_map": rank_map,
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "unset"),
        "topology_sha256": hashlib.sha256(topology.encode("utf-8")).hexdigest(),
        "config": cfg,
        "timestamp_unix": time.time(),
    }
    (output_dir / "motif_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    matrix = json.loads(Path(args.matrix_path).read_text(encoding="utf-8"))
    cfg = dict(matrix["interference_motifs"])
    if os.environ.get("OBS_REPEAT_COUNT_OVERRIDE"):
        cfg["repetitions"] = int(os.environ["OBS_REPEAT_COUNT_OVERRIDE"])
    if os.environ.get("OBS_SEED_BASE_OVERRIDE"):
        cfg["randomization_seed"] = int(os.environ["OBS_SEED_BASE_OVERRIDE"])
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("concurrent motif benchmark requires exactly eight ranks")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    output_dir = Path(args.output_dir)
    rank_query = capture_text(
        [
            "nvidia-smi",
            "-i",
            str(local_rank),
            "--query-gpu=uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ]
    ).strip()
    rank_info = {
        "global_rank": rank,
        "local_rank": local_rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_identity": rank_query,
    }
    rank_map: list[dict | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rank_map, rank_info)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in ("DONE", "motif_raw.csv", "motif_raw.jsonl", "motif_summary.csv"):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        write_metadata(output_dir, cfg, [item for item in rank_map if item is not None])
    dist.barrier()

    primitives, concurrent = motif_catalog()
    rows: list[dict] = []
    raw_jsonl = output_dir / "motif_raw.jsonl"
    for size_bytes in cfg["message_sizes_bytes"]:
        iterations = iteration_count(int(size_bytes), cfg)
        for repeat in range(1, int(cfg["repetitions"]) + 1):
            primitive_order = list(primitives)
            concurrent_order = list(concurrent)
            rng = random.Random(int(cfg["randomization_seed"]) + int(size_bytes) + repeat)
            rng.shuffle(primitive_order)
            rng.shuffle(concurrent_order)
            primitive_results: dict[str, tuple[float, float, float, bool]] = {}
            concurrent_results: dict[str, tuple[float, float, float, bool]] = {}
            blocks = (
                (("primitive", primitive_order), ("concurrent", concurrent_order))
                if repeat % 2 == 1
                else (("concurrent", concurrent_order), ("primitive", primitive_order))
            )
            for block_name, motifs in blocks:
                for motif in motifs:
                    torch.cuda.nvtx.range_push(
                        f"motif={motif.motif_id};bytes={size_bytes};repeat={repeat}"
                    )
                    try:
                        result = run_motif(
                            motif,
                            int(size_bytes),
                            iterations,
                            int(cfg["warmup_iterations"]),
                            repeat,
                            float(cfg["start_gate_delay_ms"]),
                            device,
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()
                    if block_name == "primitive":
                        primitive_results[motif.motif_id] = result
                    else:
                        concurrent_results[motif.motif_id] = result

            if rank == 0:
                for motif in primitive_order + concurrent_order:
                    if motif.motif_id in primitive_results:
                        wall_ms, device_ms, launch_skew_us, valid = primitive_results[
                            motif.motif_id
                        ]
                        baselines = {motif.motif_id: wall_ms}
                    else:
                        wall_ms, device_ms, launch_skew_us, valid = concurrent_results[
                            motif.motif_id
                        ]
                        baselines = {
                            name: primitive_results[name][0] for name in motif.primitive_ids
                        }
                    record = record_for(
                        motif,
                        int(size_bytes),
                        iterations,
                        repeat,
                        wall_ms,
                        device_ms,
                        launch_skew_us,
                        valid,
                        baselines,
                    )
                    rows.append(record)
                    with raw_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(
                        "FORGEPIPE_MOTIF_RESULT " + json.dumps(record, sort_keys=True),
                        flush=True,
                    )

    if rank == 0:
        write_csv(output_dir / "motif_raw.csv", rows)
        write_csv(output_dir / "motif_summary.csv", summarize(rows))
        (output_dir / "DONE").touch()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
