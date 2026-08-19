from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = (
        Path(__file__).parents[3]
        / "experiments"
        / "bbt_16gpu"
        / "generate_hybrid_layouts.py"
    )
    spec = importlib.util.spec_from_file_location("hybrid_layouts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantics_preserving_layouts():
    module = _load()
    sequence = "G-G*" * 2
    result = module.layouts(sequence, pp_size=2, vpp_size=2, minimum=2, maximum=3)
    assert result
    for item in result:
        assert str(item["pattern"]).replace("|", "") == sequence
        assert item["rank_counts"] == [4, 4]


if __name__ == "__main__":
    test_semantics_preserving_layouts()
    print("hybrid layout search unit test passed")
