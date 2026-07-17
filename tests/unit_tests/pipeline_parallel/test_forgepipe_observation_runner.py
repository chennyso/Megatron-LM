import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "experiments"
        / "observation_8x5090d"
        / "scripts"
        / "run_megatron_observation.py"
    )
    spec = importlib.util.spec_from_file_location("forgepipe_observation_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_summarizer():
    repo_root = Path(__file__).resolve().parents[3]
    script_dir = repo_root / "experiments" / "observation_8x5090d" / "scripts"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "forgepipe_observation_summarizer", script_dir / "summarize_megatron_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_strategy_args_lowers_explicit_policy_and_trace():
    module = _load_runner()
    case = {
        "strategy": {
            "policy": "seam-staggered",
            "runtime": "fixed",
            "microbatch_group_size": 4,
            "trace": True,
            "profile_steps": 16,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        args = module.build_strategy_args(case, Path(tmpdir))

    assert args[args.index("--pipeline-strategy-policy") + 1] == "seam-staggered"
    assert args[args.index("--microbatch-group-size-per-vp-stage") + 1] == "4"
    trace_path = args[args.index("--pipeline-strategy-trace-path") + 1]
    assert trace_path.endswith("strategy_traces/rank{rank}.json")


def test_build_strategy_args_is_empty_without_strategy_block():
    module = _load_runner()
    assert module.build_strategy_args({}, Path("/tmp/unused")) == []


def test_failure_classifier_separates_oom_from_incomplete_progress():
    module = _load_summarizer()
    result = module.classify_failure(
        "RuntimeError: CUDA out of memory", return_code=1, completed_steps=2, expected_steps=8
    )
    assert result["classes"] == ["cuda_oom"]


def test_failure_classifier_marks_non_oom_early_exit():
    module = _load_summarizer()
    result = module.classify_failure(
        "Traceback: unexpected failure", return_code=1, completed_steps=0, expected_steps=8
    )
    assert result["classes"] == ["incomplete_progress", "runtime_exception"]


def test_renderer_records_explicit_nccl_p2p_path():
    repo_root = Path(__file__).resolve().parents[3]
    renderer_path = (
        repo_root
        / "experiments"
        / "observation_8x5090d"
        / "scripts"
        / "render_volcano_job.py"
    )
    spec = importlib.util.spec_from_file_location("forgepipe_renderer", renderer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        disable_sriov_ib_network=False,
        nccl_p2p_disable="0",
        tolerate_stage0_excluded=True,
        job_name="dry-run",
        node="g5",
        image="image",
        gpu_resource_name="nvidia.com/gpu",
        gpu_count="8",
        phase="proxy",
        run_id="run",
        case_id="case",
        matrix_path="matrix.json",
        git_remote_url="remote",
        git_branch="branch",
        workspace_pvc="workspace",
        model_pvc="models",
        cpu_request="1",
        cpu_limit="2",
        mem_request="1Gi",
        mem_limit="2Gi",
        shm_size="1Gi",
        obs_repeat_count_override=None,
        obs_seed_base_override=None,
    )
    assert "export NCCL_P2P_DISABLE=0" in module.render(args)
    assert 'nvidia.com/gpu: "8"' in module.render(args)
    assert "bbt.sspu.edu.cn/stage0-excluded" in module.render(args)
