#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${PORT:-8000}"
IMAGE_NAME="sih26055-ew-dashboard:v4.1"

echo "=========================================================================="
echo "🐳 BUILDING AND RUNNING SIH26055 PRODUCTION DOCKER CONTAINER"
echo "   Target Port: http://localhost:${PORT}"
echo "=========================================================================="

docker build -t "${IMAGE_NAME}" .
docker run --rm -it -p "${PORT}:8000" --name sih26055-dashboard "${IMAGE_NAME}"
