#!/usr/bin/env python
"""Generate executable heterogeneous VPP artifacts for Megatron.

This is the first real LeasePipe/HeteroVPP mechanism: it changes Megatron's
PP/VPP layer layout and optional interleaved schedule table instead of only
profiling runtime events.

Megatron currently has a global virtual pipeline size.  This tool emulates
per-stage heterogeneous VPP by setting global_vpp=max(effective_vpp) and
emitting empty virtual chunks for stages that need fewer active chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ChunkSpec:
    pp_rank: int
    vpp_rank: int
    decoder_layers: int
    has_embedding: bool = False
    has_loss: bool = False

    @property
    def is_active_compute(self) -> bool:
        return self.decoder_layers > 0 or self.has_embedding or self.has_loss

    def layout_token(self) -> str:
        token = []
        if self.has_embedding:
            token.append("E")
        if self.decoder_layers:
            token.append("t" * self.decoder_layers)
        if self.has_loss:
            token.append("L")
        return "".join(token)


@dataclass(frozen=True)
class HeteroVPPPlan:
    num_layers: int
    pp_size: int
    global_vpp: int
    effective_vpp: list[int]
    stage_layer_counts: list[int]
    layout: str
    chunk_matrix: list[list[dict]]
    active_chunk_matrix: list[list[bool]]
    global_empty_chunks: list[int]
    schedule_table: list[dict]
    schedule_policy: str
    microbatches: int
    microbatch_group_size: int
    chunk_order: list[int]
    notes: list[str]


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        items = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list: {value}") from exc
    if not items:
        raise SystemExit(f"{name} must not be empty")
    return items


def parse_float_list(value: str, name: str) -> list[float]:
    try:
        items = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated numeric list: {value}") from exc
    if not items:
        raise SystemExit(f"{name} must not be empty")
    return items


def ensure_positive(values: Sequence[int], name: str) -> None:
    bad = [x for x in values if x <= 0]
    if bad:
        raise SystemExit(f"{name} must contain only positive integers, got {bad}")


def balanced_counts(total: int, buckets: int, weights: Sequence[float] | None = None) -> list[int]:
    """Split integer work across buckets, optionally using target capacities."""

    if total < 0:
        raise SystemExit("total must be non-negative")
    if buckets <= 0:
        raise SystemExit("buckets must be positive")
    if total == 0:
        return [0] * buckets
    if total < buckets:
        return [1 if i < total else 0 for i in range(buckets)]
    if weights is None:
        base = total // buckets
        rem = total % buckets
        return [base + (1 if i < rem else 0) for i in range(buckets)]
    if len(weights) != buckets:
        raise SystemExit(f"weights length {len(weights)} does not match buckets {buckets}")
    if any(w <= 0 for w in weights):
        raise SystemExit(f"weights must be positive: {weights}")

    weight_sum = sum(weights)
    raw = [total * w / weight_sum for w in weights]
    counts = [max(1, int(math.floor(x))) for x in raw]
    while sum(counts) > total:
        idx = max(range(buckets), key=lambda i: counts[i] - raw[i])
        counts[idx] -= 1
    while sum(counts) < total:
        idx = max(range(buckets), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    return counts


def infer_stage_layer_counts(
    *,
    num_layers: int,
    pp_size: int,
    stage_layer_counts: Sequence[int] | None,
    stage_costs: Sequence[float] | None,
    first_last_discount: int,
) -> list[int]:
    if stage_layer_counts is not None:
        if len(stage_layer_counts) != pp_size:
            raise SystemExit(
                f"--stage-layer-counts length must equal --pp-size ({pp_size}), "
                f"got {len(stage_layer_counts)}"
            )
        if sum(stage_layer_counts) != num_layers:
            raise SystemExit(
                f"--stage-layer-counts must sum to --num-layers ({num_layers}), "
                f"got {sum(stage_layer_counts)}"
            )
        if any(x < 0 for x in stage_layer_counts):
            raise SystemExit("--stage-layer-counts must be non-negative")
        return list(stage_layer_counts)

    if stage_costs is not None:
        if len(stage_costs) != pp_size:
            raise SystemExit(f"--stage-costs length must equal --pp-size ({pp_size})")
        if any(x <= 0 for x in stage_costs):
            raise SystemExit("--stage-costs must be positive")
        capacities = [1.0 / x for x in stage_costs]
        counts = balanced_counts(num_layers, pp_size, capacities)
    else:
        counts = balanced_counts(num_layers, pp_size)

    if first_last_discount > 0 and pp_size > 1:
        for edge in (0, pp_size - 1):
            movable = min(first_last_discount, max(0, counts[edge] - 1))
            counts[edge] -= movable
            for _ in range(movable):
                if pp_size > 2:
                    dst = min(range(1, pp_size - 1), key=lambda i: counts[i])
                else:
                    dst = 1 - edge
                counts[dst] += 1
    return counts


def split_stage_chunks(stage_layers: int, active_chunks: int) -> list[int]:
    if active_chunks <= 0:
        raise SystemExit("effective VPP values must be positive")
    if stage_layers == 0:
        return [0] * active_chunks
    return balanced_counts(stage_layers, active_chunks)


def build_chunk_matrix(
    *,
    num_layers: int,
    pp_size: int,
    effective_vpp: Sequence[int],
    stage_layer_counts: Sequence[int],
) -> list[list[ChunkSpec]]:
    if len(effective_vpp) != pp_size:
        raise SystemExit(
            f"--effective-vpp length must equal --pp-size ({pp_size}), got {len(effective_vpp)}"
        )
    ensure_positive(effective_vpp, "--effective-vpp")
    if len(stage_layer_counts) != pp_size:
        raise SystemExit("stage layer count length mismatch")
    if sum(stage_layer_counts) != num_layers:
        raise SystemExit("stage layer counts must sum to num_layers")

    global_vpp = max(effective_vpp)
    matrix = []
    for pp_rank in range(pp_size):
        active = effective_vpp[pp_rank]
        split = split_stage_chunks(stage_layer_counts[pp_rank], active)
        row = []
        for vpp_rank in range(global_vpp):
            row.append(
                ChunkSpec(
                    pp_rank=pp_rank,
                    vpp_rank=vpp_rank,
                    decoder_layers=split[vpp_rank] if vpp_rank < active else 0,
                    has_embedding=(pp_rank == 0 and vpp_rank == 0),
                    has_loss=(pp_rank == pp_size - 1 and vpp_rank == global_vpp - 1),
                )
            )
        matrix.append(row)
    return matrix


def layout_from_matrix(matrix: Sequence[Sequence[ChunkSpec]]) -> str:
    pp_size = len(matrix)
    global_vpp = len(matrix[0])
    flat = []
    for vpp_rank in range(global_vpp):
        for pp_rank in range(pp_size):
            flat.append(matrix[pp_rank][vpp_rank].layout_token())
    return "|".join(flat)


def chunk_decoder_loads(matrix: Sequence[Sequence[ChunkSpec]]) -> list[int]:
    global_vpp = len(matrix[0])
    return [sum(row[vpp].decoder_layers for row in matrix) for vpp in range(global_vpp)]


def chunk_active_counts(matrix: Sequence[Sequence[ChunkSpec]]) -> list[int]:
    global_vpp = len(matrix[0])
    return [sum(1 for row in matrix if row[vpp].is_active_compute) for vpp in range(global_vpp)]


def build_chunk_order(matrix: Sequence[Sequence[ChunkSpec]], policy: str) -> list[int]:
    global_vpp = len(matrix[0])
    chunks = list(range(global_vpp))
    loads = chunk_decoder_loads(matrix)
    active = chunk_active_counts(matrix)
    if policy in {"default", "wavefront"}:
        return chunks
    if policy == "heavy-first":
        return sorted(chunks, key=lambda c: (-loads[c], -active[c], c))
    if policy == "light-first":
        return sorted(chunks, key=lambda c: (loads[c], active[c], c))
    if policy == "edge-last":
        edge_chunks = {
            spec.vpp_rank for row in matrix for spec in row if spec.has_embedding or spec.has_loss
        }
        return sorted(chunks, key=lambda c: (c in edge_chunks, -loads[c], c))
    raise SystemExit(f"unknown schedule policy: {policy}")


def schedule_policy_is_megatron_safe(policy: str, chunk_order: Sequence[int]) -> bool:
    """Megatron requires each microbatch to visit VP chunks in dependency order."""

    return list(chunk_order) == list(range(len(chunk_order)))


def build_schedule_table(
    *,
    microbatches: int,
    global_vpp: int,
    microbatch_group_size: int,
    chunk_order: Sequence[int],
    schedule_policy: str,
) -> list[dict]:
    if sorted(chunk_order) != list(range(global_vpp)):
        raise SystemExit("chunk_order must be a permutation of all model chunks")
    if microbatches <= 0:
        raise SystemExit("--microbatches must be positive")
    if microbatch_group_size <= 0:
        raise SystemExit("--microbatch-group-size must be positive")

    table = []
    virtual_id = 0
    for min_mb in range(0, microbatches, microbatch_group_size):
        max_mb = min(min_mb + microbatch_group_size, microbatches)
        if schedule_policy == "wavefront":
            pairs = [
                (mb, chunk)
                for diagonal in range((max_mb - min_mb) + global_vpp - 1)
                for mb in range(min_mb, max_mb)
                for chunk in range(global_vpp)
                if (mb - min_mb) + chunk == diagonal
            ]
        else:
            pairs = [
                (mb, chunk)
                for chunk in chunk_order
                for mb in range(min_mb, max_mb)
            ]
        for mb, chunk in pairs:
                table.append(
                    {
                        "virtual_microbatch_id": virtual_id,
                        "microbatch_id": mb,
                        "model_chunk_id": int(chunk),
                    }
                )
                virtual_id += 1
    return table


def verify_forward_chunk_dependencies(
    schedule_table: Sequence[dict],
    *,
    microbatches: int,
    global_vpp: int,
) -> None:
    positions = {
        (int(item["microbatch_id"]), int(item["model_chunk_id"])): idx
        for idx, item in enumerate(schedule_table)
    }
    violations = []
    for mb in range(microbatches):
        for chunk in range(1, global_vpp):
            prev_pos = positions[(mb, chunk - 1)]
            cur_pos = positions[(mb, chunk)]
            if cur_pos < prev_pos:
                violations.append((mb, chunk - 1, prev_pos, chunk, cur_pos))
    if violations:
        raise SystemExit(
            "schedule violates forward chunk dependencies "
            "(microbatch, prev_chunk, prev_pos, chunk, chunk_pos): "
            f"{violations[:8]}"
        )


def matrix_to_dict(matrix: Sequence[Sequence[ChunkSpec]]) -> list[list[dict]]:
    return [[asdict(spec) for spec in row] for row in matrix]


def active_chunk_matrix(matrix: Sequence[Sequence[ChunkSpec]]) -> list[list[bool]]:
    return [[spec.is_active_compute for spec in row] for row in matrix]


def global_empty_chunks(matrix: Sequence[Sequence[ChunkSpec]]) -> list[int]:
    global_vpp = len(matrix[0])
    return [
        vpp_rank
        for vpp_rank in range(global_vpp)
        if not any(row[vpp_rank].is_active_compute for row in matrix)
    ]


def validate_with_megatron(layout: str, pp_size: int, num_layers: int) -> None:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from megatron.core.transformer.pipeline_parallel_layer_layout import (  # pylint: disable=import-outside-toplevel
        PipelineParallelLayerLayout,
    )

    parsed = PipelineParallelLayerLayout.from_str(layout, pp_size)
    parsed.validate_layer_layout(num_layers=num_layers, mtp_num_layers=None)


def write_json(path: str | os.PathLike[str], payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_plan(args: argparse.Namespace) -> HeteroVPPPlan:
    effective_vpp = parse_int_list(args.effective_vpp, "--effective-vpp")
    stage_layer_counts = (
        parse_int_list(args.stage_layer_counts, "--stage-layer-counts")
        if args.stage_layer_counts
        else None
    )
    stage_costs = parse_float_list(args.stage_costs, "--stage-costs") if args.stage_costs else None
    counts = infer_stage_layer_counts(
        num_layers=args.num_layers,
        pp_size=args.pp_size,
        stage_layer_counts=stage_layer_counts,
        stage_costs=stage_costs,
        first_last_discount=args.first_last_discount,
    )
    matrix = build_chunk_matrix(
        num_layers=args.num_layers,
        pp_size=args.pp_size,
        effective_vpp=effective_vpp,
        stage_layer_counts=counts,
    )
    layout = layout_from_matrix(matrix)
    global_vpp = max(effective_vpp)
    chunk_order = build_chunk_order(matrix, args.schedule_policy)
    if not schedule_policy_is_megatron_safe(args.schedule_policy, chunk_order):
        if not args.allow_unsafe_schedule:
            raise SystemExit(
                f"--schedule-policy={args.schedule_policy} produces chunk_order={chunk_order}, "
                "which is not safe for Megatron's current interleaved input/output queues. "
                "Use --schedule-policy default for executable plans, or pass "
                "--allow-unsafe-schedule only for offline analysis."
            )
    schedule_table = build_schedule_table(
        microbatches=args.microbatches,
        global_vpp=global_vpp,
        microbatch_group_size=args.microbatch_group_size,
        chunk_order=chunk_order,
        schedule_policy=args.schedule_policy,
    )
    verify_forward_chunk_dependencies(
        schedule_table,
        microbatches=args.microbatches,
        global_vpp=global_vpp,
    )
    notes = [
        "global_vpp is max(effective_vpp); inactive per-stage chunks are empty layout stages.",
        "schedule_table covers every (microbatch, model_chunk) once and can be passed through MEGATRON_CUSTOM_PP_SCHEDULE_TABLE.",
        "Megatron-safe schedule generation preserves per-microbatch forward chunk dependencies.",
        "Per-stage empty chunks are not skipped independently because neighboring PP ranks still need matched send/recv ordering.",
    ]
    if args.allow_unsafe_schedule and not schedule_policy_is_megatron_safe(args.schedule_policy, chunk_order):
        notes.append(
            "WARNING: non-default chunk orders are marked unsafe for the current Megatron runtime."
        )
    if any(v != global_vpp for v in effective_vpp):
        notes.append("This is effective heterogeneous VPP under Megatron's global VPP constraint.")

    return HeteroVPPPlan(
        num_layers=args.num_layers,
        pp_size=args.pp_size,
        global_vpp=global_vpp,
        effective_vpp=list(effective_vpp),
        stage_layer_counts=counts,
        layout=layout,
        chunk_matrix=matrix_to_dict(matrix),
        active_chunk_matrix=active_chunk_matrix(matrix),
        global_empty_chunks=global_empty_chunks(matrix),
        schedule_table=schedule_table,
        schedule_policy=args.schedule_policy,
        microbatches=args.microbatches,
        microbatch_group_size=args.microbatch_group_size,
        chunk_order=list(chunk_order),
        notes=notes,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument(
        "--effective-vpp",
        required=True,
        help="Comma-separated effective VPP chunks per PP rank, e.g. 2,3,3,1.",
    )
    parser.add_argument(
        "--stage-layer-counts",
        default=None,
        help="Optional comma-separated decoder layer counts per PP rank. Must sum to num-layers.",
    )
    parser.add_argument(
        "--stage-costs",
        default=None,
        help="Optional comma-separated relative per-layer stage costs; higher cost receives fewer layers.",
    )
    parser.add_argument(
        "--first-last-discount",
        type=int,
        default=0,
        help="Move this many decoder layers away from first and last PP stages for E/L overhead.",
    )
    parser.add_argument("--microbatches", type=int, default=16)
    parser.add_argument("--microbatch-group-size", type=int, default=1)
    parser.add_argument(
        "--schedule-policy",
        choices=["default", "wavefront", "heavy-first", "light-first", "edge-last"],
        default="default",
    )
    parser.add_argument(
        "--allow-unsafe-schedule",
        action="store_true",
        help=(
            "Allow non-default chunk orders for offline analysis. These orders are not safe for "
            "Megatron's current interleaved runtime without deeper queue/p2p changes."
        ),
    )
    parser.add_argument("--plan-out", default=None)
    parser.add_argument("--layout-out", default=None)
    parser.add_argument("--schedule-out", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.validate:
        validate_with_megatron(plan.layout, plan.pp_size, plan.num_layers)

    payload = asdict(plan)
    if args.plan_out:
        write_json(args.plan_out, payload)
    if args.layout_out:
        path = Path(args.layout_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plan.layout + "\n", encoding="utf-8")
    if args.schedule_out:
        write_json(args.schedule_out, {"schedule_table": plan.schedule_table})

    print(json.dumps(payload, indent=2))
    if args.print_env:
        if not args.schedule_out:
            raise SystemExit("--print-env requires --schedule-out")
        print("\n# Megatron runtime inputs")
        print(f"export MEGATRON_CUSTOM_PP_SCHEDULE_TABLE={args.schedule_out}")
        print(f"# add to torchrun args: --pipeline-model-parallel-layout '{plan.layout}'")


if __name__ == "__main__":
    main()
