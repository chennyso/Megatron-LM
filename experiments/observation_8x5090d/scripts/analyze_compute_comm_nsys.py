#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


OUTER_LABEL = re.compile(
    r"^overlap=(?P<case_id>[^;]+);location=(?P<location>[^;]+);"
    r"mode=(?P<mode>[^;]+);repeat=(?P<repeat>\d+)$"
)
GLOBAL_ID_TID_BITS = 24


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_intersection_ns(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def parse_outer_label(text: str) -> dict[str, str | int]:
    match = OUTER_LABEL.match(text)
    if match is None:
        raise ValueError(f"unrecognized overlap label: {text}")
    result: dict[str, str | int] = match.groupdict()
    result["repeat"] = int(result["repeat"])
    return result


def global_pid(global_tid: int) -> int:
    return global_tid & ~((1 << GLOBAL_ID_TID_BITS) - 1)


def classify_kernel(name: str) -> str:
    if "ncclDevKernel_SendRecv" in name:
        return "p2p"
    lowered = name.lower()
    if "cutlass" in lowered or "gemm" in lowered:
        return "gemm"
    if "nccl" in lowered:
        return "other_nccl"
    return "other"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_timed_ranges(connection: sqlite3.Connection) -> list[dict]:
    connection.row_factory = sqlite3.Row
    outer = connection.execute(
        "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'overlap=%'"
    ).fetchall()
    inner = connection.execute(
        "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE 'timed-window%'"
    ).fetchall()
    outer_by_tid: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in outer:
        outer_by_tid[int(row["globalTid"])].append(row)

    ranges: list[dict] = []
    for timed in inner:
        containing = [
            row
            for row in outer_by_tid[int(timed["globalTid"])]
            if int(row["start"]) <= int(timed["start"])
            and int(row["end"]) >= int(timed["end"])
        ]
        if len(containing) != 1:
            raise RuntimeError(
                f"expected one outer NVTX range for timed range, found {len(containing)}"
            )
        label = parse_outer_label(str(containing[0]["text"]))
        timed_mode = str(timed["text"]).split("=", 1)[-1]
        if timed_mode != label["mode"]:
            raise RuntimeError(f"inner/outer mode mismatch: {timed_mode} != {label['mode']}")
        ranges.append(
            {
                **label,
                "global_tid": int(timed["globalTid"]),
                "global_pid": global_pid(int(timed["globalTid"])),
                "start_ns": int(timed["start"]),
                "end_ns": int(timed["end"]),
            }
        )
    return ranges


def load_kernels(connection: sqlite3.Connection) -> dict[int, list[dict]]:
    rows = connection.execute(
        """
        SELECT k.start,k.end,k.globalPid,k.deviceId,k.streamId,p.pid,s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN PROCESSES AS p ON p.globalPid = k.globalPid
        JOIN StringIds AS s ON s.id = k.demangledName
        ORDER BY k.start
        """
    ).fetchall()
    by_process: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        record = dict(row)
        record["kernel_class"] = classify_kernel(str(record["name"]))
        by_process[int(record["globalPid"])].append(record)
    return by_process


def analyze(sqlite_path: Path) -> tuple[list[dict], list[dict]]:
    connection = sqlite3.connect(sqlite_path)
    try:
        timed_ranges = load_timed_ranges(connection)
        kernels_by_process = load_kernels(connection)
    finally:
        connection.close()

    grouped_ranges: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for timed in timed_ranges:
        key = (
            str(timed["case_id"]),
            str(timed["location"]),
            str(timed["mode"]),
            int(timed["repeat"]),
        )
        grouped_ranges[key].append(timed)

    kernel_rows: list[dict] = []
    summary_rows: list[dict] = []
    for (case_id, location, mode, repeat), ranges in sorted(grouped_ranges.items()):
        if len(ranges) != 8:
            raise RuntimeError(
                f"expected eight rank-local timed ranges for {(case_id, location, mode, repeat)}"
            )
        logical_start = min(int(row["start_ns"]) for row in ranges)
        logical_end = max(int(row["end_ns"]) for row in ranges)
        group_kernels: list[dict] = []
        for timed in ranges:
            for kernel in kernels_by_process[int(timed["global_pid"])]:
                clipped_start = max(int(kernel["start"]), int(timed["start_ns"]))
                clipped_end = min(int(kernel["end"]), int(timed["end_ns"]))
                if clipped_end <= clipped_start:
                    continue
                record = {
                    "case_id": case_id,
                    "location": location,
                    "mode": mode,
                    "repeat": repeat,
                    "pid": kernel["pid"],
                    "global_pid": kernel["globalPid"],
                    "device_id": kernel["deviceId"],
                    "stream_id": kernel["streamId"],
                    "kernel_class": kernel["kernel_class"],
                    "start_ms": (clipped_start - logical_start) / 1e6,
                    "end_ms": (clipped_end - logical_start) / 1e6,
                    "duration_ms": (clipped_end - clipped_start) / 1e6,
                    "name": kernel["name"],
                    "start_ns": clipped_start,
                    "end_ns": clipped_end,
                }
                group_kernels.append(record)
                kernel_rows.append(record)

        p2p = [
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in group_kernels
            if row["kernel_class"] == "p2p"
        ]
        gemm = [
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in group_kernels
            if row["kernel_class"] == "gemm"
        ]
        compute_endpoints = {
            (int(row["global_pid"]), int(row["device_id"]))
            for row in group_kernels
            if row["kernel_class"] == "gemm"
        }
        if len(compute_endpoints) > 1:
            raise RuntimeError(f"multiple compute endpoints in {(case_id, location, mode)}")
        endpoint_p2p = [
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in group_kernels
            if row["kernel_class"] == "p2p"
            and (int(row["global_pid"]), int(row["device_id"])) in compute_endpoints
        ]
        p2p_union = interval_union_ns(p2p)
        gemm_union = interval_union_ns(gemm)
        global_overlap = interval_intersection_ns(p2p, gemm)
        endpoint_overlap = interval_intersection_ns(endpoint_p2p, gemm)
        endpoint_serial_gap_ms = None
        if endpoint_p2p and gemm:
            endpoint_serial_gap_ms = (
                min(start for start, _ in endpoint_p2p)
                - max(end for _, end in gemm)
            ) / 1e6
        summary_rows.append(
            {
                "case_id": case_id,
                "location": location,
                "mode": mode,
                "repeat": repeat,
                "rank_range_count": len(ranges),
                "logical_window_ms": (logical_end - logical_start) / 1e6,
                "p2p_kernel_count": sum(
                    row["kernel_class"] == "p2p" for row in group_kernels
                ),
                "gemm_kernel_count": sum(
                    row["kernel_class"] == "gemm" for row in group_kernels
                ),
                "p2p_union_ms": p2p_union / 1e6,
                "gemm_union_ms": gemm_union / 1e6,
                "global_overlap_ms": global_overlap / 1e6,
                "global_gemm_overlap_fraction": global_overlap / gemm_union
                if gemm_union
                else None,
                "endpoint_p2p_union_ms": interval_union_ns(endpoint_p2p) / 1e6,
                "endpoint_overlap_ms": endpoint_overlap / 1e6,
                "endpoint_gemm_overlap_fraction": endpoint_overlap / gemm_union
                if gemm_union
                else None,
                "endpoint_serial_gap_ms": endpoint_serial_gap_ms,
                "p2p_streams": ",".join(
                    sorted(
                        {
                            f"gpu{row['device_id']}:s{row['stream_id']}"
                            for row in group_kernels
                            if row["kernel_class"] == "p2p"
                        }
                    )
                ),
                "gemm_streams": ",".join(
                    sorted(
                        {
                            f"gpu{row['device_id']}:s{row['stream_id']}"
                            for row in group_kernels
                            if row["kernel_class"] == "gemm"
                        }
                    )
                ),
            }
        )
    return kernel_rows, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    kernel_rows, summary_rows = analyze(args.sqlite)
    concurrent_rows = [row for row in summary_rows if row["mode"] == "concurrent"]
    write_csv(args.output_dir / "kernel_intervals.csv", kernel_rows)
    write_csv(args.output_dir / "timed_window_summary.csv", summary_rows)
    write_csv(args.output_dir / "concurrent_location_comparison.csv", concurrent_rows)
    (args.output_dir / "nsys_analysis_manifest.json").write_text(
        json.dumps(
            {
                "source_sqlite": str(args.sqlite),
                "kernel_interval_count": len(kernel_rows),
                "timed_window_count": len(summary_rows),
                "concurrent_location_count": len(concurrent_rows),
                "interval_method": "union_and_intersection_of_clipped_absolute_gpu_intervals",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
