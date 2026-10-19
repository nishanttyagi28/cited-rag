"""Pick a refusal threshold on half of the golden set, report it on the other half.

Rows are split by position (even -> calibration, odd -> held out) so the reported
held-out accuracy is not measured on the cases the threshold was chosen on.
"""

from __future__ import annotations

from typing import Any


def accuracy(rows: list[dict[str, Any]], threshold: float) -> float:
    if not rows:
        return 0.0
    ok = 0
    for r in rows:
        refused = r["confidence"] < threshold
        ok += refused != r["answerable"]
    return ok / len(rows)


def best_threshold(rows: list[dict[str, Any]]) -> float:
    confs = sorted({r["confidence"] for r in rows})
    cands = [0.0] + [(a + b) / 2 for a, b in zip(confs, confs[1:], strict=False)] + [1.01]
    best = max(cands, key=lambda t: (accuracy(rows, t), -t))
    return round(best, 4)


def calibrate(rows: list[dict[str, Any]]) -> dict[str, float]:
    cal, held = rows[0::2], rows[1::2]
    t = best_threshold(cal)
    return {
        "threshold": t,
        "calibration_accuracy": accuracy(cal, t),
        "held_out_accuracy": accuracy(held, t),
        "all_accuracy": accuracy(rows, t),
        "n_calibration": len(cal),
        "n_held_out": len(held),
    }
