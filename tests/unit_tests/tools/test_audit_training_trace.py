import importlib.util
import json
from pathlib import Path


_AUDIT_PATH = Path(__file__).parents[3] / "tools" / "audit_training_trace.py"
_SPEC = importlib.util.spec_from_file_location("audit_training_trace", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _event(op, iteration, **kwargs):
    return {
        "name": "collective_issue",
        "op": op,
        "issue_ts_ns": 1_000_000,
        "context": {"iteration": iteration, "phase": "forward", "vp_chunk": 2},
        **kwargs,
    }


def test_audit_aligns_zero_based_trace_and_aggregates_p2p_wait(tmp_path):
    run = tmp_path / "run"
    traces = run / "traces"
    traces.mkdir(parents=True)
    (run / "node.g5.log").write_text(
        "iteration        1/       2 | elapsed time per iteration (ms): 10.0\n"
        "iteration        2/       2 | elapsed time per iteration (ms): 20.0\n"
    )
    rows = [
        _event("all_reduce", 0, api_ms=1.0),
        _event("p2p_issue", 0, action_class="PP_FWD", send_next=True),
        _event("p2p_wait", 0, action_class="PP_FWD", send_next=True, wait_ms=0.25),
        _event("all_reduce", 1, api_ms=1.0),
        _event("p2p_issue", 1, action_class="PP_FWD", send_next=True),
        _event("p2p_wait", 1, action_class="PP_FWD", send_next=True, wait_ms=1.5),
    ]
    (traces / "rank0.json.phase.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    result = audit.audit_run(run, warmup=1, trace_iteration_offset=1)

    assert [row["trace_iteration"] for row in result["iteration_rows"]] == [0, 1]
    rank0 = result["iteration_rows"][1]["rank_trace"]["0"]
    assert rank0["p2p_wait_ms"] == 1.5
    assert rank0["p2p_by_edge"]["PP_FWD|forward|chunk=2|sn"]["wait_count"] == 1
    assert result["p2p_slow_fast_associations"][0]["edge"] == "rank=0|PP_FWD|forward|chunk=2|sn"
    assert result["p2p_slow_fast_associations"][0]["delta_wait_ms"] == 1.25
