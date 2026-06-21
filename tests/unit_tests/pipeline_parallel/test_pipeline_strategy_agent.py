import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _load_pipeline_strategy_agent():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "pipeline_strategy_agent.py"
    spec = importlib.util.spec_from_file_location("pipeline_strategy_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_search_pipeline_strategy():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "search_pipeline_strategy.py"
    spec = importlib.util.spec_from_file_location("search_pipeline_strategy", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_events_ignores_trailing_invalid_json_after_event_array():
    module = _load_pipeline_strategy_agent()
    content = """[
  {"name": "forward_step", "pp_rank": 0, "elapsed_ms": 1.0},
  {"name": "backward_step", "pp_rank": 0, "elapsed_ms": 2.0}
]
    }
  }
]"""

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.json"
        trace_path.write_text(content, encoding="utf-8")
        events = module.load_events([str(trace_path)])

    assert len(events) == 2
    assert events[0]["name"] == "forward_step"
    assert events[1]["name"] == "backward_step"


def test_boundary_aware_layout_never_emits_empty_stage():
    module = _load_search_pipeline_strategy()

    assert module._build_boundary_aware_layout(64, pipeline_parallel_size=8, vpp_size=8) is None

    layout = module._build_boundary_aware_layout(128, pipeline_parallel_size=8, vpp_size=4)
    assert layout is not None
    assert all(segment.count("t") >= 1 for segment in layout.split("|"))


def test_bcp_stats_tracks_critical_path_and_budget_pressure():
    module = _load_search_pipeline_strategy()
    events = [
        {
            "name": "forward_step",
            "pp_rank": 0,
            "model_chunk_id": 0,
            "microbatch_id": 0,
            "elapsed_ms": 4.0,
            "start_ts": 0.000,
            "end_ts": 0.004,
            "memory_mb": 100.0,
        },
        {
            "name": "p2p_recv_wait_forward",
            "pp_rank": 0,
            "model_chunk_id": 0,
            "microbatch_id": 0,
            "elapsed_ms": 3.0,
            "wait_ms": 3.0,
            "start_ts": 0.004,
            "end_ts": 0.007,
            "memory_mb": 110.0,
        },
        {
            "name": "backward_step",
            "pp_rank": 0,
            "model_chunk_id": 0,
            "microbatch_id": 0,
            "elapsed_ms": 5.0,
            "start_ts": 0.007,
            "end_ts": 0.012,
            "memory_mb": 120.0,
        },
    ]

    loose = module._bcp_stats(
        events,
        group_size=4,
        policy="default",
        vpp_size=2,
        baseline_vpp_size=2,
        layout=None,
        runtime="fixed",
        budget=module.BcpBudget(),
    )
    tight = module._bcp_stats(
        events,
        group_size=4,
        policy="default",
        vpp_size=2,
        baseline_vpp_size=2,
        layout=None,
        runtime="fixed",
        budget=module.BcpBudget(activation_peak_mb=64.0, p2p_credit=0, fb_delay_steps=1),
    )

    assert loose.critical_path_ms > 0
    assert loose.exposed_p2p_wait_ms == 3.0
    assert loose.activation_peak_mb == 120.0
    assert loose.fb_delay_steps > 0
    assert tight.score > loose.score


def test_bcp_candidate_specs_include_rewrite_rationale_and_sorting():
    module = _load_search_pipeline_strategy()
    args = SimpleNamespace(
        num_model_chunks=2,
        num_layers=16,
        pipeline_parallel_size=2,
        microbatch_group_size=2,
        candidate_budget=4,
    )
    proposal = SimpleNamespace(target={"search": "boundary_aware_layout"})
    diagnostics = {
        "overlap": {"exposed_wait_ms": 20.0},
        "hot_chunks": [{"chunk": 0, "pressure": 10.0}, {"chunk": 1, "pressure": 5.0}],
    }

    specs = module._build_candidate_specs(args, [proposal], diagnostics)

    assert specs
    assert specs == sorted(specs, key=lambda item: item.quick_score)
    assert any(spec.rewrite == "front_loaded_group" for spec in specs)
    assert all(spec.rationale for spec in specs)


def test_trace_diagnostics_reports_overlap_categories():
    module = _load_search_pipeline_strategy()
    events = [
        {
            "name": "forward_step",
            "pp_rank": 0,
            "model_chunk_id": 0,
            "microbatch_id": 0,
            "elapsed_ms": 10.0,
            "start_ts": 0.000,
            "end_ts": 0.010,
            "memory_mb": 100.0,
        },
        {
            "name": "p2p_comm_wait",
            "pp_rank": 0,
            "model_chunk_id": 0,
            "microbatch_id": 0,
            "elapsed_ms": 4.0,
            "wait_ms": 0.0,
            "start_ts": 0.002,
            "end_ts": 0.006,
            "memory_mb": 100.0,
        },
        {
            "name": "p2p_recv_wait_forward",
            "pp_rank": 0,
            "model_chunk_id": 1,
            "microbatch_id": 0,
            "elapsed_ms": 3.0,
            "wait_ms": 3.0,
            "start_ts": 0.010,
            "end_ts": 0.013,
            "memory_mb": 120.0,
        },
    ]

    diagnostics = module._trace_diagnostics(events, vpp_size=2)

    assert diagnostics["overlap"]["useful_overlap_ms"] > 0
    assert diagnostics["overlap"]["exposed_wait_ms"] == 3.0
    assert diagnostics["hot_chunks"]
