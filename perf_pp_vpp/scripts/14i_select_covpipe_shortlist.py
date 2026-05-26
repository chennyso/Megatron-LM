#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the top-k ranked COVPipe candidates into a shortlist directory.")
    parser.add_argument("--ranking-json", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    rows = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for row in rows[: args.top_k]:
        name = row["candidate_name"]
        src = args.candidate_dir / f"{name}.json"
        dst = args.output_dir / src.name
        shutil.copy2(src, dst)
        manifest.append(
            {
                "name": name,
                "candidate_json": str(dst),
                "estimated_exposed_comm_proxy_ns": row["estimated_exposed_comm_proxy_ns"],
                "estimated_covering_ratio": row["estimated_covering_ratio"],
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"count": len(manifest), "candidates": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
