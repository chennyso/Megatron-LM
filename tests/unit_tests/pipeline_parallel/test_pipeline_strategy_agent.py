import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _load_pipeline_strategy_agent():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "pipeline_strategy_agent.py"
    spec = importlib.util.spec_from_file_location("pipeline_strategy_agent", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_search_pipeline_strategy():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "search_pipeline_strategy.py"
    spec = importlib.util.spec_from_file_location("search_pipeline_strategy", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_effective_overlap():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "tools" / "analyze_effective_overlap.py"
    spec = importlib.util.spec_from_file_location("analyze_effective_overlap", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_strategy_runtime():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "megatron" / "core" / "pipeline_parallel" / "strategy_runtime.py"
    spec = importlib.util.spec_from_file_location("strategy_runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ManualTimer:
    elapsed_ms = 2.5

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


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
        num_microbatches=8,
        runtime="ready-set",
        candidate_budget=8,
    )
    proposal = SimpleNamespace(target={"search": "boundary_aware_layout"})
    diagnostics = {
        "overlap": {"exposed_wait_ms": 20.0},
        "hot_chunks": [{"chunk": 0, "pressure": 10.0}, {"chunk": 1, "pressure": 5.0}],
    }

    specs = module._build_candidate_specs(args, [proposal], diagnostics)

    assert specs
    assert any(spec.policy == "seam-staggered" for spec in specs)
    assert any(spec.policy == "default" for spec in specs)
    assert any(spec.policy != "default" for spec in specs)
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


def test_task_dag_critical_path_uses_dependencies():
    module = _load_search_pipeline_strategy()
    tasks = (
        SimpleNamespace(task_id="F:r0:c0:m0", est_compute_ms=5.0, est_comm_ms=0.0, deps=()),
        SimpleNamespace(
            task_id="SEND_F:r0:c0:m0",
            est_compute_ms=0.0,
            est_comm_ms=2.0,
            deps=("F:r0:c0:m0",),
        ),
        SimpleNamespace(
            task_id="F:r1:c0:m0",
            est_compute_ms=7.0,
            est_comm_ms=0.0,
            deps=("SEND_F:r0:c0:m0",),
        ),
        SimpleNamespace(task_id="unrelated", est_compute_ms=3.0, est_comm_ms=0.0, deps=()),
    )

    assert module._task_dag_critical_path_ms(tasks) == 14.0


def test_effective_overlap_classifies_useful_harmful_and_fake():
    module = _load_effective_overlap()
    events = [
        module.TimelineEvent("gemm", "compute", 0.0, 10.0),
        module.TimelineEvent("gemm", "compute", 20.0, 35.0),
        module.TimelineEvent("p2p_recv", "comm", 22.0, 30.0),
        module.TimelineEvent("p2p_send", "comm", 40.0, 45.0),
    ]

    report = module.classify_effective_overlap(
        events,
        harmful_slowdown_threshold=0.2,
    )

    assert report.harmful_overlap_ms == 8.0
    assert report.fake_overlap_ms == 5.0
    assert report.useful_overlap_ms == 0.0
    assert report.harmful_overlap_ratio == 1.0


def test_bcp_ready_runtime_runs_local_bubble_fill_work():
    module = _load_strategy_runtime()
    calls = []
    runtime = module.BCPReadyRuntime(mode="bcp-ready")
    runtime.register_work(
        module.BubbleFillWork(
            name="delayed_wgrad",
            run=lambda: calls.append("ran"),
            priority=10.0,
        )
    )

    result = runtime.run_one_fill(_ManualTimer)

    assert calls == ["ran"]
    assert result.ran
    assert result.name == "delayed_wgrad"
    assert result.elapsed_ms == 2.5
    assert runtime.fill_runs == 1


def test_bcp_ready_runtime_respects_p2p_credit_budget():
    module = _load_strategy_runtime()
    calls = []
    runtime = module.BCPReadyRuntime(mode="bcp-ready", p2p_credit_budget=0)
    runtime.mark_p2p_issued()
    runtime.register_work(
        module.BubbleFillWork(
            name="delayed_wgrad",
            run=lambda: calls.append("ran"),
            priority=10.0,
        )
    )

    result = runtime.run_one_fill(_ManualTimer)

    assert calls == []
    assert not result.ran
    assert result.reason == "p2p_credit_budget_exceeded"


def test_strategy_verifier_accepts_bcp_ready_without_out_of_order_p2p():
    module = _load_search_pipeline_strategy()
    synth = module._load_module(
        "strategy_synthesizer_for_bcp_ready_test",
        "megatron/core/pipeline_parallel/strategy_synthesizer.py",
    )
    constraints = synth.StrategyConstraints(
        num_microbatches=2,
        num_model_chunks=2,
        microbatch_group_size=2,
        pipeline_parallel_size=2,
    )
    candidate = synth.build_strategy_schedule_table("default", constraints)
    plan = synth.strategy_candidate_to_plan(
        candidate,
        microbatch_group_size=2,
        runtime_policy={"runtime": "bcp-ready", "allow_out_of_order_p2p": False},
    )

    synth.StrategyVerifier(constraints).verify(plan)

    invalid = synth.strategy_candidate_to_plan(
        candidate,
        microbatch_group_size=2,
        runtime_policy={"runtime": "bcp-ready", "allow_out_of_order_p2p": True},
    )
    try:
        synth.StrategyVerifier(constraints).verify(invalid)
    except ValueError as exc:
        assert "out-of-order P2P" in str(exc)
    else:
        raise AssertionError("out-of-order bcp-ready plan should be rejected")


def test_seam_staggered_schedule_is_ready_set_legal_but_fixed_rejected():
    module = _load_search_pipeline_strategy()
    synth = module._load_module(
        "strategy_synthesizer_for_seam_staggered_test",
        "megatron/core/pipeline_parallel/strategy_synthesizer.py",
    )
    constraints = synth.StrategyConstraints(
        num_microbatches=8,
        num_model_chunks=2,
        microbatch_group_size=4,
        pipeline_parallel_size=2,
    )

    candidate = synth.build_strategy_schedule_table("seam-staggered", constraints)
    default_candidate = synth.build_strategy_schedule_table("default", constraints)

    assert candidate.name == "seam-staggered"
    assert tuple(candidate.schedule_table) != tuple(default_candidate.schedule_table)

    ready_set_plan = synth.strategy_candidate_to_plan(
        candidate,
        microbatch_group_size=4,
        runtime_policy={"runtime": "ready-set", "allow_out_of_order_p2p": False},
    )
    synth.StrategyVerifier(constraints).verify(ready_set_plan)

    fixed_plan = synth.strategy_candidate_to_plan(
        candidate,
        microbatch_group_size=4,
        runtime_policy={"runtime": "fixed"},
    )
    try:
        synth.StrategyVerifier(constraints).verify(fixed_plan)
    except ValueError as exc:
        assert "requires the default table" in str(exc)
    else:
        raise AssertionError("fixed runtime should reject seam-staggered schedule table")


def test_twincut_specs_expand_policy_and_group_neighborhoods():
    module = _load_search_pipeline_strategy()

    class _TwincutStub:
        @staticmethod
        def build_pipeline_layout(segment_decoder_counts):
            return "|".join("t" * count for count in segment_decoder_counts)

        @staticmethod
        def propose_twincut_specs(**_kwargs):
            return [
                SimpleNamespace(
                    name="solver-root",
                    segment_decoder_counts=(2, 2, 3, 1),
                    segment_boundaries=(0, 2, 4, 7, 8),
                    cross_node_boundaries=(2,),
                    node_assignment=(0, 0, 1, 1),
                    vpp_packing=((2, 3), (2, 1)),
                    memory_actions=((1, "retain"), (2, "recompute")),
                    objective=73.0,
                )
            ]

    args = SimpleNamespace(
        num_model_chunks=2,
        num_layers=8,
        pipeline_parallel_size=2,
        microbatch_group_size=4,
        num_microbatches=8,
        runtime="bcp-ready",
        memory_budget_mb=None,
        candidate_budget=16,
    )
    diagnostics = {
        "overlap": {"exposed_wait_ms": 12.0},
        "p2p_credit_pressure": 1.0,
        "hot_chunks": [{"chunk": 0, "pressure": 8.0}, {"chunk": 1, "pressure": 3.0}],
    }
    events = [
        {"name": "forward_step", "pp_rank": 0, "elapsed_ms": 4.0, "memory_mb": 100.0},
        {"name": "backward_step", "pp_rank": 0, "elapsed_ms": 5.0, "memory_mb": 120.0},
        {"name": "p2p_recv_wait_forward", "pp_rank": 0, "elapsed_ms": 2.0, "wait_ms": 2.0},
    ]

    specs = module._build_twincut_specs(args, diagnostics, events, [2], _TwincutStub())

    assert specs
    assert any(spec.policy == "seam-staggered" for spec in specs)
    assert any(spec.group_size != args.microbatch_group_size for spec in specs)
    assert all(spec.rewrite == "twincut_partition" for spec in specs)
    assert all(spec.memory_actions for spec in specs)
