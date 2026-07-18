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
        / "benchmark_concurrent_motifs.py"
    )
    spec = importlib.util.spec_from_file_location("concurrent_motif_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_has_matched_two_three_four_way_baselines():
    module = load_module()
    primitives, concurrent = module.motif_catalog()
    primitive_ids = {motif.motif_id for motif in primitives}

    disjoint_concurrency = {
        len(motif.edges) for motif in concurrent if motif.route_class == "endpoint_disjoint"
    }
    assert disjoint_concurrency == {
        2,
        3,
        4,
    }
    assert {len(motif.edges) for motif in concurrent if motif.route_class == "shared_endpoint"} == {
        2,
        3,
        4,
    }
    for motif in concurrent:
        assert set(motif.primitive_ids) <= primitive_ids


def test_iteration_count_respects_byte_target_and_bounds():
    module = load_module()
    cfg = {
        "target_bytes_per_flow": 1024,
        "min_iterations": 4,
        "max_iterations": 20,
    }

    assert module.iteration_count(1, cfg) == 20
    assert module.iteration_count(256, cfg) == 4
    assert module.iteration_count(4096, cfg) == 4
