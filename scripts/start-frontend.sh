#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
PORT=3000

echo "🌐 [CourtVision AI] Starting Frontend Dashboard..."
echo "🚀 Serving on http://localhost:$PORT"

if command -v python3 &>/dev/null; then
    exec python3 -m http.server "$PORT" --directory "$FRONTEND_DIR"
elif command -v python &>/dev/null; then
    exec python -m http.server "$PORT" --directory "$FRONTEND_DIR"
elif command -v npx &>/dev/null; then
    exec npx --yes serve "$FRONTEND_DIR" -l "$PORT"
else
    echo "❌ Error: Neither python3 nor npx found to serve static files."
    exit 1
fi
