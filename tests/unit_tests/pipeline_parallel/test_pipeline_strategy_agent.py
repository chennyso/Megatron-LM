import importlib.util
import tempfile
from pathlib import Path


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
