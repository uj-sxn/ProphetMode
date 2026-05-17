"""Pydantic schemas that match the Prophet Arena wire format exactly.

Input  : Event (from ai_prophet_core if available, own definition otherwise)
Output : PredictionResponse — {"probabilities": [{"market": str, "probability": float}]}
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Event input — use the official SDK type when available
# ---------------------------------------------------------------------------

try:
    from ai_prophet_core.forecast.schemas import Event  # noqa: F401
except ImportError:
    from datetime import datetime

    class Event(BaseModel):  # type: ignore[no-redef]
        event_ticker: str
        market_ticker: str
        title: str
        subtitle: Optional[str] = None
        description: Optional[str] = None
        category: str
        rules: Optional[str] = None
        close_time: datetime
        outcomes: List[str]
        resolved_outcome: Optional[str] = None


# ---------------------------------------------------------------------------
# Prediction output — the official wire format
# ---------------------------------------------------------------------------

class ProbabilityEntry(BaseModel):
    market: str       # must match an outcomes label exactly (case-sensitive)
    probability: float  # in [0, 1]; values across outcomes do NOT need to sum to 1


class PredictionResponse(BaseModel):
    probabilities: List[ProbabilityEntry]
