#!/usr/bin/env python
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def fetch_all(conn: sqlite3.Connection, query: str) -> list[dict]:
    cur = conn.execute(query)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(
        conn.execute(
            "select count(*) from sqlite_master where type='table' and name=?",
            (name,),
        ).fetchone()[0]
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 11d_analyze_nsys_sqlite.py <sqlite-path>")

    path = Path(sys.argv[1]).resolve()
    conn = sqlite3.connect(path)
    try:
        out: dict[str, object] = {"sqlite": str(path)}

        if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
            out["kernel_summary"] = fetch_all(
                conn,
                """
                WITH kernel_summary AS (
                  SELECT
                    s.value AS name,
                    SUM(k.end - k.start) AS total_ns,
                    COUNT(*) AS calls,
                    AVG(k.end - k.start) AS avg_ns
                  FROM CUPTI_ACTIVITY_KIND_KERNEL k
                  LEFT JOIN StringIds s ON s.id = k.demangledName
                  GROUP BY s.value
                )
                SELECT
                  name,
                  total_ns,
                  calls,
                  avg_ns
                FROM kernel_summary
                ORDER BY total_ns DESC
                LIMIT 20
                """,
            )
            out["kernel_breakdown"] = fetch_all(
                conn,
                """
                SELECT
                  SUM(CASE WHEN LOWER(s.value) LIKE '%nccl%' THEN k.end - k.start ELSE 0 END) AS nccl_ns,
                  SUM(CASE WHEN LOWER(s.value) NOT LIKE '%nccl%' THEN k.end - k.start ELSE 0 END) AS compute_ns,
                  SUM(k.end - k.start) AS total_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds s ON s.id = k.demangledName
                """,
            )

        if table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            out["cuda_api_summary"] = fetch_all(
                conn,
                """
                SELECT
                  s.value AS name,
                  SUM(r.end - r.start) AS total_ns,
                  COUNT(*) AS calls,
                  AVG(r.end - r.start) AS avg_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                LEFT JOIN StringIds s ON s.id = r.nameId
                GROUP BY s.value
                ORDER BY total_ns DESC
                LIMIT 20
                """,
            )

        if table_exists(conn, "NVTX_EVENTS"):
            out["nvtx_summary"] = fetch_all(
                conn,
                """
                SELECT
                  COALESCE(s.value, n.text) AS name,
                  SUM(n.end - n.start) AS total_ns,
                  COUNT(*) AS calls,
                  AVG(n.end - n.start) AS avg_ns
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds s ON s.id = n.textId
                WHERE n.end IS NOT NULL
                  AND n.end > n.start
                  AND COALESCE(s.value, n.text) IS NOT NULL
                GROUP BY COALESCE(s.value, n.text)
                ORDER BY total_ns DESC
                LIMIT 30
                """,
            )

        if table_exists(conn, "DIAGNOSTIC_EVENT"):
            out["diagnostics"] = fetch_all(
                conn,
                """
                SELECT
                  severity,
                  source,
                  text
                FROM DIAGNOSTIC_EVENT
                ORDER BY timestamp DESC
                LIMIT 50
                """,
            )

        print(json.dumps(out, indent=2, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
