# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "bbt_16gpu" / "scripts" / "summarize_clean_screening.py"
SPEC = importlib.util.spec_from_file_location("summarize_clean_screening", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(case_id, median_iter_ms):
    vpp_size, sequence_parallel = MODULE.case_dimensions(case_id)
    return {
        "case_id": case_id,
        "repeat_id": 1,
        "vpp_size": vpp_size,
        "transition_budget": 2 * vpp_size - 1,
        "sequence_parallel": sequence_parallel,
        "median_iter_ms": median_iter_ms,
        "mean_iter_ms": median_iter_ms,
        "within_run_cv_percent": 0.1,
        "median_tokens_per_second": 1.0,
    }


def test_case_dimensions_maps_1f1b_and_vpp_to_transition_budget_inputs():
    assert MODULE.case_dimensions("clean_1f1b_nosp") == (1, False)
    assert MODULE.case_dimensions("clean_vpp8_sp") == (8, True)

    with pytest.raises(ValueError, match="unsupported"):
        MODULE.case_dimensions("screen_vpp4_overlap")


def test_summary_computes_speedup_and_sp_crossover():
    rows = [
        row("clean_1f1b_nosp", 10.0),
        row("clean_1f1b_sp", 10.5),
        row("clean_vpp8_nosp", 8.6),
        row("clean_vpp8_sp", 8.4),
    ]

    summary, crossover = MODULE.summarize(rows)

    best = next(item for item in summary if item["case_id"] == "clean_vpp8_sp")
    assert best["speedup_vs_1f1b_nosp"] == pytest.approx(10.0 / 8.4)
    assert crossover == [
        {
            "vpp_size": 1,
            "transition_budget": 1,
            "nosp_median_iter_ms": 10.0,
            "sp_median_iter_ms": 10.5,
            "sp_throughput_gain_percent": pytest.approx((10.0 / 10.5 - 1.0) * 100.0),
            "winner": "nosp",
        },
        {
            "vpp_size": 8,
            "transition_budget": 15,
            "nosp_median_iter_ms": 8.6,
            "sp_median_iter_ms": 8.4,
            "sp_throughput_gain_percent": pytest.approx((8.6 / 8.4 - 1.0) * 100.0),
            "winner": "sp",
        },
    ]
