"""ProphetMode FastAPI server.

Start:
    uvicorn main:app --reload --port 8000

CLI usage:
    prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from agent.forecaster import async_predict
from config import get_settings
from models.schemas import Event, PredictionResponse, ProbabilityEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    if not cfg.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set — will use uniform fallback for all events")
    if not cfg.tavily_api_key:
        logger.warning("TAVILY_API_KEY not set — LLM-only mode (no web search)")
    logger.info("ProphetMode ready  model=%s", cfg.anthropic_model)
    yield


app = FastAPI(
    title="ProphetMode",
    description="Calibrated AI forecasting agent — Prophet Hacks 2026",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    cfg = get_settings()
    return {
        "name": "ProphetMode",
        "team": "ProphetMode — Prophet Hacks 2026",
        "model": cfg.anthropic_model,
        "endpoints": {"predict": "POST /predict", "health": "GET /health"},
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: Event) -> PredictionResponse:
    """Receive a single Event and return calibrated probabilities.

    The CLI calls this once per event:
        prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
    """
    result = await async_predict(event.model_dump(mode="json"))
    entries = [ProbabilityEntry(**p) for p in result["probabilities"]]
    return PredictionResponse(probabilities=entries)


def main() -> None:
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", cfg.port)),
        reload=True,
    )


if __name__ == "__main__":
    main()
