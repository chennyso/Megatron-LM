#!/usr/bin/env python3
"""Generate semantics-preserving fVPP boundary placements.

The global hybrid layer sequence is immutable. Candidates insert only ``|``
boundaries into that sequence, so they change PP/VPP ownership without
changing model architecture or parameter order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _compositions(total: int, parts: int, minimum: int, maximum: int):
    if parts == 1:
        if minimum <= total <= maximum:
            yield (total,)
        return
    for first in range(minimum, maximum + 1):
        remaining = total - first
        if minimum * (parts - 1) <= remaining <= maximum * (parts - 1):
            for suffix in _compositions(remaining, parts - 1, minimum, maximum):
                yield (first,) + suffix


def _segments(sequence: str, counts: tuple[int, ...]) -> list[str]:
    cursor = 0
    result = []
    for count in counts:
        result.append(sequence[cursor : cursor + count])
        cursor += count
    assert cursor == len(sequence)
    return result


def layouts(
    sequence: str,
    pp_size: int = 4,
    vpp_size: int = 2,
    minimum: int = 2,
    maximum: int = 6,
    require_rank_balance: bool = True,
) -> list[dict[str, object]]:
    segment_count = pp_size * vpp_size
    result = []
    for counts in _compositions(len(sequence), segment_count, minimum, maximum):
        rank_counts = [
            sum(counts[chunk * pp_size + rank] for chunk in range(vpp_size))
            for rank in range(pp_size)
        ]
        if require_rank_balance and len(set(rank_counts)) != 1:
            continue
        segment_list = _segments(sequence, counts)
        pattern = "|".join(segment_list)
        assert pattern.replace("|", "") == sequence
        result.append({
            "id": len(result),
            "counts": list(counts),
            "rank_counts": rank_counts,
            "segments": segment_list,
            "pattern": pattern,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="G-G*" * 8)
    parser.add_argument("--pp-size", type=int, default=4)
    parser.add_argument("--vpp-size", type=int, default=2)
    parser.add_argument("--min-segment-layers", type=int, default=2)
    parser.add_argument("--max-segment-layers", type=int, default=6)
    parser.add_argument("--allow-rank-imbalance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "sequence": args.sequence,
        "pp_size": args.pp_size,
        "vpp_size": args.vpp_size,
        "layouts": layouts(
            args.sequence,
            args.pp_size,
            args.vpp_size,
            args.min_segment_layers,
            args.max_segment_layers,
            not args.allow_rank_imbalance,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"layouts": len(payload["layouts"]), "layers": len(args.sequence)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
