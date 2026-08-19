#!/usr/bin/env python3
"""Enumerate composition-preserving fVPP layouts for the GDN/attention case.

Each segment has four layer symbols: two GDN layers, one MLP layer, and one
attention layer.  The only degree of freedom is the position of the
attention layer inside its segment.  Thus every generated layout has exactly
the same model layer multiset and the same segment length; only VPP placement
changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


POSITIONS = {
    "first": "*G-G",
    "middle": "G*-G",
    "last": "G-G*",
}


def layouts(segments: int, limit: int | None = None) -> list[dict[str, object]]:
    names = tuple(POSITIONS)
    result = []
    for index, choice in enumerate(itertools.product(names, repeat=segments)):
        pattern = "|".join(POSITIONS[item] for item in choice)
        # Every segment is four layers and has the same multiset.  These
        # checks make the invariant explicit and protect future edits.
        flat = pattern.replace("|", "")
        assert len(flat) == segments * 4
        assert flat.count("G") == segments * 2
        assert flat.count("*") == segments
        assert flat.count("-") == segments
        result.append({"id": index, "choices": list(choice), "pattern": pattern})
        if limit is not None and len(result) >= limit:
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "segments": args.segments,
        "positions": POSITIONS,
        "layouts": layouts(args.segments, args.limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"layouts": len(payload["layouts"]), "segments": args.segments}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
