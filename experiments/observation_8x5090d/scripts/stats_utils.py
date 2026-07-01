from __future__ import annotations

import math
import statistics
from typing import Iterable


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def clean_numeric(values: Iterable[float | int | None]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        cleaned.append(numeric)
    return cleaned


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ci95_halfwidth(values: Iterable[float | int | None]) -> float | None:
    cleaned = clean_numeric(values)
    n = len(cleaned)
    if n <= 1:
        return None
    std = statistics.stdev(cleaned)
    critical = T_CRITICAL_95.get(n - 1, 1.96)
    return critical * std / math.sqrt(n)


def summarize_numeric(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    cleaned = clean_numeric(values)
    if not cleaned:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_halfwidth": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "n": len(cleaned),
        "mean": statistics.mean(cleaned),
        "std": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
        "ci95_halfwidth": ci95_halfwidth(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "p50": percentile(cleaned, 0.5),
        "p95": percentile(cleaned, 0.95),
    }
