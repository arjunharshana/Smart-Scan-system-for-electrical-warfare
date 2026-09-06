from __future__ import annotations

import os
from pathlib import Path

# Project root resolution (portable, no hardcoded user paths)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Server configuration
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

# Simulation defaults
DEFAULT_SCENARIO_NAME = os.environ.get("DEFAULT_SCENARIO", "deterministic_hopping.yaml")
DEFAULT_SCENARIO_PATH = PROJECT_ROOT / "rf_environment" / "scenarios" / DEFAULT_SCENARIO_NAME

# Tactical Scheduler version: V4.0 Hybrid is frozen production tactical scheduler (Benchmark Winner: 35.19% IR)
DEFAULT_SCHEDULER = os.environ.get("DEFAULT_SCHEDULER", "hybrid_v4")
DEFAULT_V40_CHECKPOINT = PROJECT_ROOT / "models" / "v4_0" / "production_checkpoint.npz"
DEFAULT_V41_CHECKPOINT = PROJECT_ROOT / "models" / "v4_1" / "production_checkpoint.npz"
APP_VERSION = "4.0"
