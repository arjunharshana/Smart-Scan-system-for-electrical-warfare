#!/usr/bin/env bash
set -e

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "=========================================================================="
echo "🚀 LAUNCHING SIH26055 TACTICAL EW SCANNER DASHBOARD (V4.1)"
echo "   URL: http://${HOST}:${PORT}"
echo "=========================================================================="

python -m app serve --host "${HOST}" --port "${PORT}"
