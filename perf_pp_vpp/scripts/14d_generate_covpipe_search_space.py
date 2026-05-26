#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_candidate(
    *,
    name: str,
    delta_layers: int,
    rank3_chunks: list[int],
    rank4_chunks: list[int],
    left_target: str,
    right_target: str,
) -> dict:
    return {
        "name": name,
        "pp": 8,
        "vpp": 4,
        "num_layers": 64,
        "seed": "uniform",
        "transforms": [
            {"kind": "move_boundary", "stage": 3, "delta_layers": delta_layers},
            {"kind": "reshape_vpp_chunk", "rank": 3, "chunks": rank3_chunks},
            {"kind": "reshape_vpp_chunk", "rank": 4, "chunks": rank4_chunks},
            {"kind": "delay_wgrad", "rank": 3, "chunk": 0, "target": left_target},
            {"kind": "delay_wgrad", "rank": 4, "chunk": 0, "target": right_target},
        ],
    }


def enumerate_candidates() -> list[dict]:
    shapes = [
        ("tail_heavy", [2, 2, 2, 3], [2, 2, 2, 1]),
        ("head_heavy", [3, 2, 2, 2], [1, 2, 2, 2]),
        ("mid_heavy_a", [2, 3, 2, 2], [2, 1, 2, 2]),
        ("mid_heavy_b", [2, 2, 3, 2], [2, 2, 1, 2]),
    ]
    targets = [
        ("recv_wait", "send_inflight"),
        ("recv_wait", "recv_wait"),
        ("send_inflight", "recv_wait"),
    ]

    candidates: list[dict] = []
    for delta_layers in (1, -1):
        for shape_name, rank3_chunks, rank4_chunks in shapes:
            if delta_layers < 0:
                rank3_chunks, rank4_chunks = rank4_chunks, rank3_chunks
            for left_target, right_target in targets:
                direction = "left_plus" if delta_layers > 0 else "right_plus"
                target_name = f"{left_target}__{right_target}"
                name = f"covpipe_{direction}__{shape_name}__{target_name}"
                candidates.append(
                    make_candidate(
                        name=name,
                        delta_layers=delta_layers,
                        rank3_chunks=rank3_chunks,
                        rank4_chunks=rank4_chunks,
                        left_target=left_target,
                        right_target=right_target,
                    )
                )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small verified COVPipe search space.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in enumerate_candidates():
        path = args.output_dir / f"{candidate['name']}.json"
        path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append({"name": candidate["name"], "path": str(path)})
    manifest = {"count": len(rows), "candidates": rows}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
