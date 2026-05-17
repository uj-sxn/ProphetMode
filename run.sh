#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Starting ProphetMode on port ${PORT:-8000}..."
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --reload \
  --log-level info
