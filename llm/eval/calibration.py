"""
llm/eval/calibration.py - Uncertainty and calibration metrics for model predictions.

Computes:
- Binary/Categorical Brier score
- Expected Calibration Error (ECE) across confidence bins
- Selective classification risk-at-coverage thresholds
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_brier_score(confidences: Sequence[float], correctness: Sequence[bool | int]) -> float | None:
    """
    Computes Brier score: mean((confidence - is_correct)^2).
    Lower is better (0.0 is perfect calibration, 1.0 is completely wrong/uncalibrated).
    """
    if not confidences or len(confidences) != len(correctness):
        return None
    return sum((c - int(ok)) ** 2 for c, ok in zip(confidences, correctness)) / len(confidences)


def compute_ece(confidences: Sequence[float], correctness: Sequence[bool | int], bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE):
    Partitions predictions into `bins` equally-spaced intervals [0, 0.1), ..., [0.9, 1.0],
    and computes the weighted absolute difference between confidence and empirical accuracy.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    ece = 0.0
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        # Include upper boundary in the last bin
        indices = [
            i for i, c in enumerate(confidences)
            if (low <= c < high) or (b == bins - 1 and low <= c <= high)
        ]
        if not indices:
            continue

        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(int(correctness[i]) for i in indices) / len(indices)
        weight = len(indices) / n
        ece += weight * abs(bin_conf - bin_acc)

    return ece


def compute_risk_coverage(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    target_coverages: Sequence[float] = (0.5, 0.8, 1.0),
) -> dict[str, dict[str, float]]:
    """
    Computes selective classification risk (1 - accuracy) when keeping only predictions
    with confidence >= threshold to meet target coverage fractions.
    """
    n = len(confidences)
    if n == 0:
        return {}

    sorted_scores = sorted(set(confidences), reverse=True)
    results: dict[str, dict[str, float]] = {}

    for target in target_coverages:
        # Find threshold that covers at least `target` fraction of items
        threshold = sorted_scores[-1]
        for score in sorted_scores:
            cov = sum(1 for c in confidences if c >= score) / n
            if cov >= target:
                threshold = score
                break

        selected = [ok for c, ok in zip(confidences, correctness) if c >= threshold]
        cov_actual = len(selected) / n
        risk = 1.0 - (sum(int(ok) for ok in selected) / max(len(selected), 1))
        results[f"{target:.1f}"] = {
            "coverage": round(cov_actual, 4),
            "risk": round(risk, 4),
            "threshold": round(threshold, 4),
        }

    return results


def evaluate_calibration(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    bins: int = 10,
) -> dict[str, Any]:
    """
    Full calibration summary dictionary.
    """
    if not confidences:
        return {"n": 0, "brier": None, "ece": None, "risk_at_coverage": {}}

    return {
        "n": len(confidences),
        "brier": round(compute_brier_score(confidences, correctness) or 0.0, 5),
        "ece": round(compute_ece(confidences, correctness, bins=bins), 5),
        "risk_at_coverage": compute_risk_coverage(confidences, correctness),
    }
