"""Tavily web search — gathers evidence for each event before LLM reasoning."""
from __future__ import annotations

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category-aware query generation
# ---------------------------------------------------------------------------


def _build_queries(event: dict) -> List[str]:
    title = event.get("title", "")
    category = event.get("category", "")
    outcomes = event.get("outcomes", [])

    if category == "Sports":
        entities = " vs ".join(outcomes[:2]) if len(outcomes) >= 2 else title
        return [
            f"{title} 2026 prediction odds",
            f"{entities} recent form head to head injury report",
            f"{entities} betting odds expert picks stats",
        ]

    if category == "Economics":
        return [
            f"{title} 2026 economic forecast",
            f"{title} latest data analyst forecast Fed statement",
            f"{title} prediction market outlook indicators",
        ]

    if category == "Entertainment":
        return [
            f"{title} 2026 prediction",
            f"{title} polls expert picks betting odds favorites",
            f"{title} winner forecast nominations frontrunner",
        ]

    if category in ("Politics", "Elections"):
        return [
            f"{title} 2026 forecast",
            f"{title} polling averages prediction markets",
            f"{title} expert forecast election odds",
        ]

    # Default — generic
    return [
        f"{title} 2026",
        f"{title} prediction analysis forecast",
        f"{' '.join(outcomes[:3])} statistics history performance",
    ]


# ---------------------------------------------------------------------------
# Single search call
# ---------------------------------------------------------------------------


async def _search_one(client, query: str, timeout: float) -> str:
    try:
        result = await asyncio.wait_for(
            client.search(query, max_results=3, search_depth="basic"),
            timeout=timeout,
        )
        parts = []
        for r in result.get("results", [])[:3]:
            snippet = r.get("content", "")[:500].strip()
            parts.append(f"[{r.get('title', '')}]\n{snippet}")
        return "\n\n".join(parts)
    except asyncio.TimeoutError:
        logger.warning("Tavily timeout: %s", query)
        return ""
    except Exception as exc:
        logger.warning("Tavily error '%s': %s", query, exc)
        return ""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def gather_evidence(event: dict, api_key: str, timeout: float = 12.0) -> str:
    """Run 3 category-aware searches and return concatenated evidence. Never raises."""
    if not api_key:
        return "No TAVILY_API_KEY configured — LLM will use prior knowledge only."

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=api_key)
        queries = _build_queries(event)
        chunks = await asyncio.gather(*[_search_one(client, q, timeout) for q in queries])
        evidence = [c for c in chunks if c]

        if not evidence:
            return "Search returned no results — LLM will use prior knowledge only."

        return ("\n\n" + "─" * 60 + "\n\n").join(evidence)

    except ImportError:
        return "tavily-python not installed — LLM will use prior knowledge only."
    except Exception as exc:
        logger.warning("Evidence gathering failed: %s", exc)
        return f"Search unavailable ({exc}) — LLM will use prior knowledge only."
