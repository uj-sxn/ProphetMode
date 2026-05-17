"""ProphetMode forecasting pipeline.

Exposes two interfaces:
  1. predict(event: dict) -> dict   — sync, for: prophet forecast predict --local agent.forecaster
  2. async_predict(event: dict) -> dict — async, used by the FastAPI endpoint in main.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from agent.calibrator import clamp, fill_missing, pull_toward_half, uniform
from agent.retriever import gather_evidence
from agent.reasoner import reason

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core async pipeline
# ---------------------------------------------------------------------------


async def async_predict(event: dict) -> dict:
    """Run the full pipeline for one event. Never raises.

    Returns:
        {"probabilities": [{"market": str, "probability": float}, ...]}
    """
    from config import get_settings

    cfg = get_settings()
    outcomes: List[str] = event.get("outcomes", [])
    ticker: str = event.get("market_ticker", event.get("event_ticker", "unknown"))
    title: str = event.get("title", "")

    if not outcomes:
        logger.warning("%s has no outcomes — returning empty probabilities", ticker)
        return {"probabilities": []}

    logger.info("START %s — %s", ticker, title[:70])

    # ── Step 1: Web search ────────────────────────────────────────────────
    try:
        evidence = await gather_evidence(event, cfg.tavily_api_key, cfg.search_timeout)
    except Exception as exc:
        logger.warning("Search failed for %s: %s", ticker, exc)
        evidence = "Search unavailable — LLM using prior knowledge only."

    evidence_is_thin = (
        evidence.startswith("Search")
        or evidence.startswith("No")
        or evidence.startswith("tavily")
    )

    # ── Step 2: LLM reasoning ─────────────────────────────────────────────
    raw: Dict[str, float] = {}
    reasoning_text: str = ""

    if cfg.openrouter_api_key:
        raw, reasoning_text = await reason(
            event, evidence, cfg.openrouter_api_key, cfg.anthropic_model
        )
    else:
        logger.warning("OPENROUTER_API_KEY not set — using uniform fallback for %s", ticker)

    # ── Step 3: Fallback if reasoning returned nothing ────────────────────
    if not raw:
        logger.warning("Using uniform fallback for %s", ticker)
        raw = uniform(outcomes)

    # ── Step 4: Fill any missing outcomes, optionally soften, then clamp ──
    raw = fill_missing(raw, outcomes)

    if evidence_is_thin:
        max_p = max(raw.values(), default=0.5)
        if max_p > 0.80:
            raw = pull_toward_half(raw, weight=0.20)

    calibrated = clamp(raw)

    # ── Step 5: Build output in official wire format ──────────────────────
    probs_list = [
        {"market": outcome, "probability": calibrated[outcome]}
        for outcome in outcomes
        if outcome in calibrated
    ]

    # ── Step 6: Structured prediction log ────────────────────────────────
    _log_prediction(ticker, title, probs_list, reasoning_text)

    return {"probabilities": probs_list}


def _log_prediction(
    ticker: str,
    title: str,
    probs_list: List[Dict[str, Any]],
    reasoning_text: str,
) -> None:
    """Log a complete prediction record to the console."""
    timestamp = datetime.now(timezone.utc).isoformat()
    probs_summary = " | ".join(
        f"{p['market']}={p['probability']:.3f}" for p in probs_list
    )

    logger.info(
        "PREDICTION | timestamp=%s | ticker=%s | title=%s | probs=[%s]",
        timestamp,
        ticker,
        title[:80],
        probs_summary,
    )

    if reasoning_text:
        logger.info(
            "REASONING  | ticker=%s | %s",
            ticker,
            reasoning_text[:800],
        )
    else:
        logger.info("REASONING  | ticker=%s | (fallback — no LLM reasoning)", ticker)


# ---------------------------------------------------------------------------
# Sync wrapper — required by `prophet forecast predict --local agent.forecaster`
# ---------------------------------------------------------------------------


def predict(event: dict) -> dict:
    """Synchronous entry point for the Prophet Arena CLI --local flag.

    Usage:
        prophet forecast predict --events events.json --local agent.forecaster
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Standard CLI context — no running loop
        return asyncio.run(async_predict(event))

    # Called from inside an already-running loop (e.g. Jupyter / test harness)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, async_predict(event)).result()
