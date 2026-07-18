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
        / "aggregate_motif_process_runs.py"
    )
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("aggregate_motif_process_runs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launch_skew_quality_uses_fraction_of_measured_duration():
    module = load_module()
    row = {"launch_skew_us": "50", "wall_makespan_ms": "10"}
    assert module.motif_launch_skew_fraction(row) == 0.005


def test_large_message_summary_preserves_independent_process_unit():
    module = load_module()
    rows = []
    for run_id, slowdown in (("run-1", 1.0), ("run-2", 1.2)):
        for size in (16 * 1024 * 1024, 64 * 1024 * 1024):
            rows.append(
                {
                    "process_run_id": run_id,
                    "route_class": "endpoint_disjoint",
                    "motif_id": "disjoint_oneway_2",
                    "size_bytes": str(size),
                    "slowdown_vs_isolated_max": str(slowdown),
                    "launch_quality_ok": True,
                }
            )

    run_rows = module.build_large_message_run_rows(rows)
    summary = module.summarize_grouped(
        run_rows,
        ("category",),
        ("slowdown_mean_quality",),
    )[0]

    assert len(run_rows) == 2
    assert summary["process_count"] == 2
    assert summary["slowdown_mean_quality_mean"] == 1.1
