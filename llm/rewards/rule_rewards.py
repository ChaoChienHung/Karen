"""
llm/rewards/rule_rewards.py - Verifiable rule-based rewards for RLVR, DPO, and GRPO.

Provides deterministic, rule-based reward terms:
- correctness: Verdict match with optional asymmetric harm matrix
- evidence_f1 / evidence_score: Citation grounding to prevent hallucinations
- calibration_reward: Proper Brier scoring for uncertainty honesty
- total: Unified reward aggregation with ablation switches
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NEI = "NOT_ENOUGH_INFO"

L1_EVIDENCE = 0.5
L2_CALIBRATION = 0.5
L3_COST = 0.02

# (gold, predicted) -> harm of that wrong verdict.
HARM_MATRIX: dict[tuple[str, str], float] = {
    (REFUTED, SUPPORTED): 1.5,
    (SUPPORTED, REFUTED): 1.0,
    (NEI, SUPPORTED): 1.75,
    (NEI, REFUTED): 1.0,
}

NEI_EMPTY_REWARD = 1.0
NEI_CITED_PENALTY = 0.5


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "full"  # "full", "format_only", "random"
    calibration: bool = True
    harm: bool = True
    evidence: bool = True


FULL_REWARD_CONFIG = RewardConfig()


def correctness(gold: str, pred: str, harm: bool = True) -> float:
    """Returns 1.0 for matching prediction, (1 - harm) for wrong prediction."""
    if gold == pred:
        return 1.0
    if not harm:
        return 0.0
    return 1.0 - HARM_MATRIX.get((gold, pred), 1.0)


def evidence_f1(gold_ids: Sequence[str], cited_ids: Sequence[str]) -> float:
    """Calculates F1 score between gold sentence IDs and model cited IDs."""
    if not gold_ids or not cited_ids:
        return 0.0
    g, c = set(gold_ids), set(cited_ids)
    tp = len(g & c)
    if tp == 0:
        return 0.0
    precision = tp / len(c)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def evidence_score(gold: str, pred: str, gold_ids: Sequence[str], cited_ids: Sequence[str]) -> float:
    """Evidence F1 conditioned on correct verdict; penalizes citing on gold NEI."""
    if pred != gold:
        return -NEI_CITED_PENALTY if gold == NEI and cited_ids else 0.0
    if gold == NEI:
        return NEI_EMPTY_REWARD if not cited_ids else -NEI_CITED_PENALTY
    return evidence_f1(gold_ids, cited_ids)


def calibration_reward(gold: str, pred: str, confidence: float) -> float:
    """
    Negative Brier score of stated confidence in [-1, 0].
    Strictly proper scoring rule: incentivizes honest probability assessment.
    """
    outcome = 1.0 if gold == pred else 0.0
    return -((confidence - outcome) ** 2)


def compute_total_reward(
    gold: str,
    pred: str,
    confidence: float = 1.0,
    gold_ids: Sequence[str] = (),
    cited_ids: Sequence[str] = (),
    n_ops: int = 0,
    format_ok: bool = True,
    cfg: RewardConfig = FULL_REWARD_CONFIG,
    rng: random.Random | None = None,
) -> tuple[float, dict[str, Any]]:
    """Aggregates all components into a scalar reward value."""
    if cfg.mode == "random":
        r = (rng or random).random()
        return r, {"random": r}
    if cfg.mode == "format_only":
        r = 1.0 if format_ok else 0.0
        return r, {"format": r}
    if cfg.mode != "full":
        raise ValueError(f"Unknown reward mode: {cfg.mode!r}")

    parts = {
        "correctness": correctness(gold, pred, harm=cfg.harm),
        "evidence": evidence_score(gold, pred, gold_ids, cited_ids) if cfg.evidence else 0.0,
        "calibration": calibration_reward(gold, pred, confidence) if cfg.calibration else 0.0,
        "cost": float(n_ops),
        "evidence_f1": evidence_f1(gold_ids, cited_ids),
    }

    r = (
        parts["correctness"]
        + L1_EVIDENCE * parts["evidence"]
        + L2_CALIBRATION * parts["calibration"]
        - L3_COST * parts["cost"]
    )
    return r, parts
