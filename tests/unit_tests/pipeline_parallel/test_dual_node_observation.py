import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_observation_matrix_uses_real_32b_workload_and_expected_schedules():
    config = json.loads(
        (REPO_ROOT / "experiments/bbt_16gpu/configs/observation_16gpu.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["workload"]["model_path"] == "/models/qwen3-32B"
    assert "fineweb_edu_formal_text_document" in config["workload"]["data_path"]
    assert config["workload"]["fsdp"] is False
    assert {config["cases"][case]["vpp_size"] for case in config["cases"] if case.startswith("screen_")} == {1, 2, 4, 8}
    assert config["repeat_policy"]["throughput_repeats"] == 5


def test_renderer_requests_exactly_eight_standard_gpus_per_node():
    renderer = load_module(
        "dual_node_observation_renderer",
        "experiments/bbt_16gpu/scripts/render_dual_node_observation.py",
    )
    manifest = renderer.render(
        SimpleNamespace(
            config=REPO_ROOT / "experiments/bbt_16gpu/configs/observation_16gpu.json",
            case_id="screen_vpp4_overlap",
            run_id="test-run",
            repeat_id=1,
            git_ref="test-branch",
            profile_mode="throughput",
            master_port=29500,
            warmup_steps=None,
            measure_steps=None,
            git_remote=None,
            image=None,
            workspace_pvc=None,
            model_pvc=None,
        )
    )
    assert manifest.count('nvidia.com/gpu: "8"') == 4
    assert manifest.count('nvidia.com/mlnxnics: "1"') == 4
    assert "--mock-data" not in manifest
    assert "fineweb_edu_formal_text_document" in manifest
    assert 'value: "4"' in manifest
    assert "bbt.sspu.edu.cn/stage0-excluded" in manifest


def test_analyzer_excludes_warmup_steps(tmp_path):
    analyzer = load_module(
        "dual_node_observation_analyzer",
        "experiments/bbt_16gpu/scripts/analyze_dual_node_observation.py",
    )
    log_path = tmp_path / "training.log"
    lines = []
    for iteration, iter_ms in [(1, 100.0), (2, 90.0), (3, 80.0), (4, 70.0)]:
        lines.append(
            f"iteration {iteration}/4 | consumed samples: {iteration * 16} | "
            f"elapsed time per iteration (ms): {iter_ms} | learning rate: 1.0E-4 | "
            "global batch size: 16 | lm loss: 2.0E+0 | grad norm: 1.0E+0 |"
        )
    log_path.write_text("\n".join(lines), encoding="utf-8")
    rows, summary = analyzer.parse_steps(log_path, warmup_steps=2, seq_length=4096)
    assert len(rows) == 4
    assert summary["steady_steps"] == 2
    assert summary["iteration_time_ms"]["median"] == 75.0


def test_analyzer_iteration_pattern_matches_real_dual_node_log(tmp_path):
    analyzer = load_module(
        "dual_node_observation_analyzer_real_log",
        "experiments/bbt_16gpu/scripts/analyze_dual_node_observation.py",
    )
    log_path = tmp_path / "training.log"
    log_path.write_text(
        " [2026-07-18 02:41:45.146306] iteration        4/       4 | "
        "consumed samples:           64 | elapsed time per iteration (ms): 9347.3 | "
        "throughput per GPU (TFLOP/s/GPU): 81.7 | learning rate: 1.000000E-04 | "
        "global batch size:    16 | lm loss: 1.831113E+01 | loss scale: 1.0 | "
        "grad norm: 16.288 | number of skipped iterations:   0 |\n",
        encoding="utf-8",
    )
    rows, summary = analyzer.parse_steps(log_path, warmup_steps=2, seq_length=4096)
    assert len(rows) == 1
    assert summary["iteration_time_ms"]["median"] == 9347.3


def test_matrix_controller_resolves_branch_to_immutable_commit():
    controller = load_module(
        "dual_node_observation_controller",
        "experiments/bbt_16gpu/scripts/run_observation_matrix.py",
    )
    resolved = controller.resolve_git_ref("HEAD")
    assert len(resolved) == 40
    assert all(character in "0123456789abcdef" for character in resolved)
