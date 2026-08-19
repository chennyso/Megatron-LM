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


def test_composition_preserving_layouts():
    module = _load()
    result = module.layouts(2)
    assert len(result) == 9
    for item in result:
        flat = str(item["pattern"]).replace("|", "")
        assert len(flat) == 8
        assert flat.count("G") == 4
        assert flat.count("*") == 2
        assert flat.count("-") == 2


if __name__ == "__main__":
    test_composition_preserving_layouts()
    print("hybrid layout search unit test passed")
