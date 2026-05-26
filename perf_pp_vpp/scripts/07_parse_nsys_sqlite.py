#!/usr/bin/env python
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import dump_json, maybe_float, sqlite_tables, write_csv


def table_columns(conn: sqlite3.Connection, name: str):
    return [row[1] for row in conn.execute(f"PRAGMA table_info('{name}')").fetchall()]


def extract_table(conn: sqlite3.Connection, name: str, limit: int = 100000):
    cols = table_columns(conn, name)
    rows = conn.execute(f"SELECT * FROM '{name}' LIMIT {limit}").fetchall()
    return [dict(zip(cols, row)) for row in rows]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 07_parse_nsys_sqlite.py <run-dir>")
    run_dir = Path(sys.argv[1]).resolve()
    rank_events = []
    rank_summary = []
    boundary = []
    overlap = []

    for sqlite_path in sorted(run_dir.rglob("*.sqlite")):
        conn = sqlite3.connect(sqlite_path)
        try:
            tables = sqlite_tables(sqlite_path)
            nvtx_tables = [t for t in tables if "NVTX" in t.upper()]
            kernel_tables = [t for t in tables if "KERNEL" in t.upper()]
            mem_tables = [t for t in tables if "MEM" in t.upper()]

            total_profile_time_ms = None
            compute_kernel_time_ms = None
            nccl_kernel_time_ms = None
            memory_peak = None

            for table in nvtx_tables[:2]:
                for row in extract_table(conn, table, limit=5000):
                    rank_events.append({
                        "sqlite": str(sqlite_path),
                        "table": table,
                        "event_type": "nvtx",
                        **row,
                    })
            for table in kernel_tables[:2]:
                rows = extract_table(conn, table, limit=5000)
                if rows:
                    rank_events.extend({
                        "sqlite": str(sqlite_path),
                        "table": table,
                        "event_type": "kernel",
                        **row,
                    } for row in rows)
                if compute_kernel_time_ms is None:
                    durations = [maybe_float(r.get("end")) and maybe_float(r.get("start")) for r in rows]
                    compute_kernel_time_ms = None
            for table in mem_tables[:1]:
                rows = extract_table(conn, table, limit=5000)
                for row in rows[:100]:
                    value = maybe_float(row.get("value")) or maybe_float(row.get("bytes")) or maybe_float(row.get("size"))
                    if value is not None:
                        memory_peak = max(memory_peak or value, value)

            rank_summary.append({
                "sqlite": str(sqlite_path),
                "total_profile_time_ms": total_profile_time_ms,
                "compute_kernel_time_ms": compute_kernel_time_ms,
                "nccl_kernel_time_ms": nccl_kernel_time_ms,
                "p2p_send_recv_time_ms": None,
                "idle_or_gap_time_ms": None,
                "memory_peak": memory_peak,
                "notes": "schema-dependent parser; unavailable fields are null",
            })
            overlap.append({
                "sqlite": str(sqlite_path),
                "p2p_overlap_ratio": None,
                "exposed_p2p_ms": None,
                "nccl_overlap_ratio": None,
                "exposed_nccl_ms": None,
            })
            boundary.append({
                "sqlite": str(sqlite_path),
                "boundary": "stage7_to_stage8",
                "send_count": None,
                "recv_count": None,
                "total_send_recv_time": None,
                "avg_msg_size": None,
                "cross_node": True,
            })
        finally:
            conn.close()

    write_csv(run_dir / "nsys_rank_events.csv", rank_events)
    write_csv(run_dir / "nsys_rank_summary.csv", rank_summary)
    write_csv(run_dir / "nsys_boundary_comm.csv", boundary)
    write_csv(run_dir / "nsys_overlap_summary.csv", overlap)
    dump_json(run_dir / "nsys_parse_notes.json", {"limitations": ["SQLite schema varies by Nsight Systems version; parser emits null for unavailable metrics."]})


if __name__ == "__main__":
    main()
