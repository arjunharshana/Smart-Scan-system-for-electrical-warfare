#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=========================================================================="
echo "📊 EXECUTING V4.1 CANONICAL BENCHMARK SUITE"
echo "=========================================================================="

python benchmarks/run_v4_1_benchmark.py
