"""Probability calibration utilities.

Probabilities are scored independently — they do NOT need to sum to 1.
We only clamp each value to [0.05, 0.95] individually.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def clamp(probabilities: Dict[str, float], lo: float = 0.05, hi: float = 0.95) -> Dict[str, float]:
    """Clamp each probability independently to [lo, hi]. No normalization."""
    result: Dict[str, float] = {}
    for outcome, p in probabilities.items():
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            logger.warning("Bad probability for '%s': %s — replacing with 0.5", outcome, p)
            p = 0.5
        result[outcome] = max(lo, min(hi, float(p)))
    return result


def fill_missing(
    probabilities: Dict[str, float],
    outcomes: List[str],
    default: float = 0.5,
) -> Dict[str, float]:
    """Ensure every outcome in `outcomes` has an entry, filling gaps with `default`."""
    filled = dict(probabilities)
    for outcome in outcomes:
        if outcome not in filled:
            logger.warning("LLM did not return probability for '%s' — using %.2f", outcome, default)
            filled[outcome] = default
    return filled


def pull_toward_half(probabilities: Dict[str, float], weight: float = 0.25) -> Dict[str, float]:
    """Pull each probability toward 0.5 by `weight` (use when evidence is thin)."""
    return {
        outcome: (1.0 - weight) * p + weight * 0.5
        for outcome, p in probabilities.items()
    }


def uniform(outcomes: List[str], value: float = 0.5) -> Dict[str, float]:
    """Uniform fallback — 0.5 per outcome for binary; 1/N for multi-outcome is also fine
    but 0.5 is safe since probabilities are scored independently."""
    return {outcome: value for outcome in outcomes}
