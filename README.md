# Pythia — AI Forecasting Agent

**Team ProphetMode · Prophet Hacks 2026 · University of Chicago - SIGMA Labs**

> Pythia queries the web, reasons with LLMs, and returns calibrated 
> probability scores for real-world events — beating the market, 
> one forecast at a time.

---

## What It Does

Pythia is a calibrated AI forecasting agent built for the Prophet Arena 
benchmark. It receives real-world forecasting questions, retrieves live 
web evidence, reasons through the evidence using a structured 
superforecaster prompt, and returns a calibrated probability per outcome.

**Pipeline:**
```
Question → Web Search (Tavily) → LLM Reasoning (Claude) → Calibration → Probability
```
---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/YOURNAME/ProphetMode.git
cd ProphetMode
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Fill in your keys in .env
```

### 3. Run locally
```bash
uvicorn main:app --reload --port 8000
```

### 4. Test the endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "event_ticker": "test-001",
    "market_ticker": "test-001",
    "title": "Will Cleveland beat Detroit in Game 6?",
    "description": "Resolves yes if Cleveland wins.",
    "category": "Sports",
    "rules": "Resolves to the official winner after the game is final.",
    "close_time": "2026-05-18T23:59:59Z",
    "outcomes": ["Cleveland", "Detroit"],
    "resolved_outcome": null
  }'
```

**Expected response:**
```json
{
  "probabilities": [
    {"market": "Cleveland", "probability": 0.95},
    {"market": "Detroit", "probability": 0.05}
  ]
}
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Submit one event, receive probabilities |
| GET | `/health` | Uptime check |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM API access via OpenRouter |
| `TAVILY_API_KEY` | Yes | Web search for evidence retrieval |
| `PA_SERVER_API_KEY` | No | Prophet Arena leaderboard access |
| `ANTHROPIC_MODEL` | No | Defaults to claude-sonnet-4-5 |
| `PORT` | No | Defaults to 8000 |

---

## Architecture
```
main.py (FastAPI)
└── agent/
├── forecaster.py   # Pipeline orchestrator
├── retriever.py    # Tavily web search, category-aware queries
├── 
reasoner.py     # LLM reasoning via OpenRouter + Claude
└── calibrator.py   # Probability clamping [0.05, 0.95]
models/
└── schemas.py      # Pydantic event + prediction schemas
eval/
└── brier.py        # Local Brier score evaluation
```
---

## Self-Evaluation

```bash
# Pull resolved dataset
prophet forecast retrieve \
  --dataset sample-resolved --include-resolved -o resolved.json

# Run agent against it
prophet forecast predict \
  --events resolved.json \
  --agent-url http://localhost:8000/predict

# Score locally
prophet forecast evaluate \
  --submission predictions.json \
  --actuals actuals.json
```

---

## Deployment

Deployed on **Render** (free tier). Environment variables set via 
Render dashboard — no keys in the repository.

Live endpoint: `https://prophetmode.onrender.com/predict`

---

## Team

| Name |
|---|
| Urjita Saxena |
| Akshat Behera |

**Prophet Hacks 2026 · Forecasting Track**

---

## License

MIT
