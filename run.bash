#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uvicorn sam_server:app --host 127.0.0.1 --port 8001 &
SAM_PID=$!

cd "$ROOT_DIR"
python3 -m http.server 8000

trap 'kill "$SAM_PID" 2>/dev/null || true' EXIT
