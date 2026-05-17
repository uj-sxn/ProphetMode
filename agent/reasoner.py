"""LLM reasoning via OpenRouter (OpenAI-compatible) — returns raw probability dict per event."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — superforecaster, from spec verbatim
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are ProphetMode, a superforecaster in the Tetlock tradition.

For each outcome in the event, estimate the probability it resolves positively.
Probabilities are scored independently — they do not need to sum to 1.

Think step by step:
1. BASE RATE: How often does this type of outcome occur historically?
2. EVIDENCE: What do the search results tell you?
3. RESOLUTION RULES: Read the rules field carefully — it defines exactly \
what makes an outcome resolve to 1.
4. FOR EACH OUTCOME: What is the probability it resolves positively?

RULES:
- Return a probability for EVERY outcome in the list
- Each probability is between 0.05 and 0.95
- They do NOT need to sum to 1
- Never refuse — always give your best estimate
- Output ONLY valid JSON:
  {"probabilities": [{"market": "Label", "probability": 0.X}, ...]}"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _user_prompt(event: dict, evidence: str) -> str:
    outcomes: List[str] = event.get("outcomes", [])
    outcomes_block = "\n".join(f"  - {o}" for o in outcomes)
    example_json = json.dumps(
        {"probabilities": [{"market": o, "probability": 0.5} for o in outcomes[:2]]},
        indent=2,
    )

    parts = [
        f"Title: {event.get('title', '')}",
        f"Category: {event.get('category', '')}",
        f"Close time: {event.get('close_time', '')}",
    ]
    if event.get("rules"):
        parts.append(f"\nResolution rules (read carefully):\n{event['rules']}")
    parts += [
        f"\nOutcomes — assign a probability to EVERY one:\n{outcomes_block}",
        f"\nEvidence from web search:\n{evidence}",
        f"\nOutput ONLY valid JSON matching this shape (fill in real probabilities):\n{example_json}",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction — three fallback strategies
# ---------------------------------------------------------------------------


def _extract(raw: str, outcomes: List[str]) -> Dict[str, float]:
    def _from_parsed(obj: dict) -> Dict[str, float] | None:
        probs = obj.get("probabilities")
        if isinstance(probs, list):
            result = {}
            for entry in probs:
                if isinstance(entry, dict) and "market" in entry and "probability" in entry:
                    result[str(entry["market"])] = float(entry["probability"])
            if result:
                return result
        return None

    # Strategy 1: whole response is JSON
    try:
        obj = json.loads(raw)
        result = _from_parsed(obj)
        if result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find JSON object with "probabilities" key in the text
    for match in re.finditer(r'\{[^{}]*"probabilities"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.DOTALL):
        try:
            obj = json.loads(match.group())
            result = _from_parsed(obj)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            result = _from_parsed(obj)
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse LLM JSON (first 300 chars): %s", raw[:300])
    return {}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def reason(
    event: dict, evidence: str, api_key: str, model: str
) -> Tuple[Dict[str, float], str]:
    """Call the LLM via OpenRouter.

    Returns:
        (probabilities_dict, raw_reasoning_text)
        probabilities_dict is {} on failure; raw_reasoning_text is "" on failure.
    """
    outcomes: List[str] = event.get("outcomes", [])

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(event, evidence)},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        return _extract(raw, outcomes), raw

    except openai.APIError as exc:
        logger.error("OpenRouter API error for %s: %s", event.get("market_ticker"), exc)
        return {}, ""
    except Exception as exc:
        logger.error("Reasoner unexpected error for %s: %s", event.get("market_ticker"), exc)
        return {}, ""
