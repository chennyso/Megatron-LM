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
    sys.path.insert(0, str(path.parent))
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
    qkv = next(case for case in catalog if case.case_id == "qkv_forward")
    # Qwen3-32B: (64 Q heads + 2 * 8 KV heads) * head_dim 128 / TP2.
    assert qkv.n == 5120


def test_sampled_reference_error_checks_values_not_only_finiteness():
    module = load_module()
    torch = module.torch
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    output = left @ right

    valid, max_abs_error, _ = module.sampled_reference_error(
        left, right, output, rtol=0.0, atol=0.0
    )
    assert valid
    assert max_abs_error == 0.0

    output[1, 1] += 1.0
    valid, max_abs_error, _ = module.sampled_reference_error(
        left, right, output, rtol=0.0, atol=0.0
    )
    assert not valid
    assert max_abs_error == 1.0
