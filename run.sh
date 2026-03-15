#!/usr/bin/env bash
set -euo pipefail

# Load env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Start uvicorn
PYTHON_GIL=0 uv run uvicorn fabricai_inference_server.app:app --host 0.0.0.0 --port 8000 --reload
