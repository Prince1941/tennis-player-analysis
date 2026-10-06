#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

echo "🎾 [CourtVision AI] Starting Backend Server..."

# Locate virtual environment python
if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$ROOT_DIR/.venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

echo "🐍 Python Interpreter: $PYTHON_EXEC"
echo "🚀 Uvicorn will listen on http://localhost:8000"

cd "$BACKEND_DIR"
exec "$PYTHON_EXEC" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
