import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module():
    path = REPO_ROOT / "tools" / "search_topology_segmented_vpp.py"
    spec = importlib.util.spec_from_file_location("search_topology_segmented_vpp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vpp8_search_covers_intermediate_transition_budgets():
    module = load_module()
    candidates = module.build_candidates(8, 8, ("g5",) * 4 + ("g6",) * 4)

    assert {item["topology_transition_count"] for item in candidates} == {1, 3, 7, 15}
    assert len(candidates) == 7
    assert all(len(item["route"]) == 64 for item in candidates)


def test_endpoint_rotation_reduces_peak_load_at_fixed_budget():
    module = load_module()
    candidates = module.build_candidates(8, 8, ("g5",) * 4 + ("g6",) * 4)
    by_name = {item["name"]: item for item in candidates}

    assert by_name["segmented-h2"]["topology_transition_count"] == 7
    assert by_name["segmented-h2-rotated"]["topology_transition_count"] == 7
    assert by_name["segmented-h2-rotated"]["endpoint_peak_load"] < by_name[
        "segmented-h2"
    ]["endpoint_peak_load"]


def test_pareto_front_keeps_transition_reuse_tradeoffs():
    module = load_module()
    candidates = module.build_candidates(8, 8, ("g5",) * 4 + ("g6",) * 4)
    pareto = module.pareto_front(candidates)

    assert {item["hierarchy_factor"] for item in pareto} == {1, 2, 4, 8}
