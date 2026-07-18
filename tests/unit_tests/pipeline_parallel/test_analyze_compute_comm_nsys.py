import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module():
    path = (
        REPO_ROOT
        / "experiments"
        / "observation_8x5090d"
        / "scripts"
        / "analyze_compute_comm_nsys.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_compute_comm_nsys", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_interval_union_and_intersection_do_not_double_count():
    module = load_module()
    left = [(0, 10), (5, 15), (20, 30)]
    right = [(8, 22)]

    assert module.interval_union_ns(left) == 25
    assert module.interval_intersection_ns(left, right) == 9


def test_parse_outer_label_extracts_strategy_dimensions():
    module = load_module()
    parsed = module.parse_outer_label(
        "overlap=mlp_forward;location=receiver;mode=concurrent;repeat=1"
    )

    assert parsed == {
        "case_id": "mlp_forward",
        "location": "receiver",
        "mode": "concurrent",
        "repeat": 1,
    }
