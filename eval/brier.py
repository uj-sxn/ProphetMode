#!/usr/bin/env python3
"""Local Brier score calculator.

Works with the file formats produced by the Prophet Arena CLI:

  predictions.json  — output of `prophet forecast predict`
  actuals.json      — flat map: {"market_ticker": "resolved_label", ...}

Usage:
    # Pull the hackathon dataset and predict
    prophet forecast retrieve --dataset hackathon-day -o events.json
    prophet forecast predict --events events.json --local agent.forecaster -o predictions.json

    # Build actuals.json as outcomes resolve (see README)
    # Then score:
    python eval/brier.py --predictions predictions.json --actuals actuals.json

    # Or score against the running HTTP endpoint directly:
    python eval/brier.py --events events.json --actuals actuals.json \\
                         --host http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def brier_score_event(probabilities: List[Dict[str, Any]], resolved: str) -> float:
    """Per-event Brier score: Σ(p_i − outcome_i)² across all submitted outcomes."""
    score = 0.0
    for entry in probabilities:
        market = entry["market"]
        p = float(entry["probability"])
        o = 1.0 if market == resolved else 0.0
        score += (p - o) ** 2
    return score


# ---------------------------------------------------------------------------
# Load predictions from a file OR live endpoint
# ---------------------------------------------------------------------------


def load_predictions_file(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load predictions.json → {market_ticker: [{market, probability}, ...]}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, List[Dict]] = {}

    # Format: list of {"market_ticker": ..., "probabilities": [...]}
    if isinstance(data, list):
        for item in data:
            ticker = item.get("market_ticker") or item.get("event_ticker")
            if ticker and "probabilities" in item:
                result[ticker] = item["probabilities"]
        return result

    # Format: {"predictions": [...]}
    if isinstance(data, dict) and "predictions" in data:
        for item in data["predictions"]:
            ticker = item.get("market_ticker") or item.get("event_ticker")
            if ticker and "probabilities" in item:
                result[ticker] = item["probabilities"]
        return result

    print("ERROR: Unrecognised predictions.json format", file=sys.stderr)
    sys.exit(1)


async def fetch_predictions_from_endpoint(
    events: List[dict], host: str
) -> Dict[str, List[Dict]]:
    import httpx

    result: Dict[str, List[Dict]] = {}
    async with httpx.AsyncClient(timeout=600.0) as client:
        for event in events:
            ticker = event.get("market_ticker") or event.get("event_ticker")
            try:
                resp = await client.post(f"{host}/predict", json=event)
                resp.raise_for_status()
                data = resp.json()
                result[ticker] = data.get("probabilities", [])
            except Exception as exc:
                print(f"  WARNING: {ticker} — {exc}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def evaluate(args: argparse.Namespace) -> None:
    # Load actuals
    with open(args.actuals, encoding="utf-8") as f:
        actuals: Dict[str, str] = json.load(f)
    print(f"Actuals loaded: {len(actuals)} resolved events")

    # Load predictions
    if args.predictions:
        predictions = load_predictions_file(args.predictions)
        print(f"Predictions loaded from file: {len(predictions)} events")
    elif args.events and args.host:
        with open(args.events, encoding="utf-8") as f:
            raw = json.load(f)
        events = raw if isinstance(raw, list) else raw.get("events", [])
        print(f"Fetching predictions from {args.host} for {len(events)} events …")
        predictions = await fetch_predictions_from_endpoint(events, args.host)
        print(f"Predictions received: {len(predictions)} events")
    else:
        print("ERROR: provide --predictions OR (--events + --host)", file=sys.stderr)
        sys.exit(1)

    # Score
    scores: List[float] = []
    missed: List[str] = []
    rows = []

    for ticker, resolved in actuals.items():
        if ticker not in predictions:
            missed.append(ticker)
            continue
        probs = predictions[ticker]
        score = brier_score_event(probs, resolved)
        p_correct = next((e["probability"] for e in probs if e["market"] == resolved), 0.0)
        scores.append(score)
        rows.append((ticker, resolved, p_correct, score))

    if missed:
        print(f"\nWARNING: {len(missed)} actuals not found in predictions: {missed[:5]}")

    # Print per-event table
    print()
    header = f"{'Ticker':<45}  {'Resolved':<18}  {'P(correct)':>10}  {'Brier':>7}"
    print(header)
    print("─" * len(header))
    for ticker, resolved, p_correct, score in sorted(rows, key=lambda r: r[3], reverse=True):
        print(f"{ticker[:44]:<45}  {resolved[:17]:<18}  {p_correct:>10.3f}  {score:>7.4f}")

    if not scores:
        print("\nNo events scored.")
        return

    # Summary
    mean_bs = sum(scores) / len(scores)
    completion = len(scores) / max(len(actuals), 1)

    # Random baseline = 1 - 1/N for each event (averaged)
    n_list: List[int] = []
    for ticker in actuals:
        if ticker in predictions:
            n_list.append(len(predictions[ticker]))
    random_bs = sum(1.0 - 1.0 / n for n in n_list) / len(n_list) if n_list else 0.5

    print(f"\n{'='*55}")
    print(f"  Events scored   : {len(scores)} / {len(actuals)}")
    print(f"  Completion rate : {completion:.1%}")
    print(f"  Mean Brier      : {mean_bs:.4f}  (lower is better)")
    print(f"  Random baseline : {random_bs:.4f}")
    delta = random_bs - mean_bs
    verdict = "BETTER" if delta > 0 else "WORSE"
    print(f"  vs random       : {verdict} by {abs(delta):.4f}")
    print(f"{'='*55}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Brier score for ProphetMode predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--predictions", help="predictions.json from `prophet forecast predict`")
    parser.add_argument("--actuals", required=True, help='actuals.json: {"market_ticker": "label"}')
    parser.add_argument("--events", help="events.json (needed with --host)")
    parser.add_argument("--host", default="http://localhost:8000", help="Agent endpoint for live scoring")
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
