import json
import sys
import tempfile
from types import SimpleNamespace

from megatron.training.arguments import _apply_pipeline_strategy_plan_overrides


def test_strategy_plan_overrides_layout_and_group_size():
    payload = {
        "name": "agent-plan",
        "schedule_table": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "pipeline_layout": "Et|tL",
        "microbatch_group_size": 3,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        handle.flush()
        plan_path = handle.name

    args = SimpleNamespace(
        pipeline_strategy_plan=plan_path,
        pipeline_model_parallel_layout=None,
        num_virtual_stages_per_pipeline_rank=4,
        num_layers_per_virtual_pipeline_stage=None,
        microbatch_group_size_per_vp_stage=None,
        rank=0,
    )

    _apply_pipeline_strategy_plan_overrides(args)

    assert args.pipeline_model_parallel_layout == "Et|tL"
    assert args.num_virtual_stages_per_pipeline_rank is None
    assert args.microbatch_group_size_per_vp_stage == 3


def test_strategy_plan_overrides_virtual_pipeline_degree_without_layout():
    payload = {
        "name": "agent-plan-vp8",
        "schedule_table": [[0, chunk] for chunk in range(8)],
        "num_virtual_stages_per_pipeline_rank": 8,
        "microbatch_group_size": 4,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        handle.flush()
        plan_path = handle.name

    args = SimpleNamespace(
        pipeline_strategy_plan=plan_path,
        pipeline_model_parallel_layout=None,
        num_virtual_stages_per_pipeline_rank=4,
        num_layers_per_virtual_pipeline_stage=2,
        microbatch_group_size_per_vp_stage=8,
        rank=0,
    )

    _apply_pipeline_strategy_plan_overrides(args)

    assert args.pipeline_model_parallel_layout is None
    assert args.num_virtual_stages_per_pipeline_rank == 8
    assert args.num_layers_per_virtual_pipeline_stage is None
    assert args.microbatch_group_size_per_vp_stage == 4


def test_strategy_policy_argument_accepts_seam_staggered():
    from megatron.training.arguments import parse_args

    argv = [
        "test-parse-args",
        "--num-layers", "8",
        "--hidden-size", "128",
        "--num-attention-heads", "4",
        "--max-position-embeddings", "128",
        "--micro-batch-size", "1",
        "--pipeline-model-parallel-size", "2",
        "--pipeline-strategy-policy", "seam-staggered",
    ]

    old_argv = sys.argv
    sys.argv = argv
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = old_argv

    assert args.pipeline_strategy_policy == "seam-staggered"
