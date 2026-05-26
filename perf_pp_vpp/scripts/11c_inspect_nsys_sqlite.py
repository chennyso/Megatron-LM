#!/usr/bin/env python
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 11c_inspect_nsys_sqlite.py <sqlite-path>")

    path = Path(sys.argv[1]).resolve()
    conn = sqlite3.connect(path)
    try:
        names = [r[0] for r in conn.execute(
            "select name from sqlite_master where type='table' order by name"
        ).fetchall()]
        for name in names:
            cols = [r[1] for r in conn.execute(f"pragma table_info('{name}')").fetchall()]
            count = conn.execute(f"select count(*) from '{name}'").fetchone()[0]
            print(f"TABLE {name} rows={count}")
            print(",".join(cols[:24]))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
