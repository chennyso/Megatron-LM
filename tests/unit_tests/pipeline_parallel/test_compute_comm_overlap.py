import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module():
    path = (
        REPO_ROOT
        / "experiments"
        / "observation_8x5090d"
        / "scripts"
        / "benchmark_compute_comm_overlap.py"
    )
    spec = importlib.util.spec_from_file_location("compute_comm_overlap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_catalog_covers_forward_dinput_and_dweight():
    module = load_module()
    catalog = module.compute_catalog()

    assert {case.action_kind for case in catalog} == {"F", "dI", "dW"}
    assert {case.case_id for case in catalog} == {
        "qkv_forward",
        "mlp_forward",
        "mlp_dinput",
        "mlp_dweight",
    }
    assert all(case.m > 0 and case.k > 0 and case.n > 0 for case in catalog)
