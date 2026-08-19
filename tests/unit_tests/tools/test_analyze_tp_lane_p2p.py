import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[3] / "tools" / "analyze_tp_lane_p2p.py"
_SPEC = importlib.util.spec_from_file_location("analyze_tp_lane_p2p", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def _event(rank, label, tag, wait_ms):
    return {
        "op": "p2p_issue",
        "rank": rank,
        "issue_ts_ns": 1_000_000 + rank,
        "context": {"tp_rank": 0},
        "p2p_message_tags": {label: tag},
        "p2p_request_waits": [{"request_label": label, "wait_ms": wait_ms}],
    }


def test_pairs_semantic_send_and_receive_by_tp_lane():
    tag = "source_pp=3|tp=0|vp=1|mb=2|vmb=4|direction=forward"
    result = module.analyze([
        _event(6, "send_next", tag, 0.1),
        _event(8, "recv_prev", tag, 2.0),
    ])

    assert result["paired_messages"] == 1
    assert result["unmatched_sends"] == 0
    assert result["unmatched_receives"] == 0
    assert result["by_tp_lane"]["0"]["receiver_wait"]["mean_ms"] == 2.0
