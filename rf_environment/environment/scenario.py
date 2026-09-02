from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rf_environment.emitters.factory import create_emitter


def load_scenario(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Scenario file must contain a mapping")
    return data


def emitters_from_scenario(scenario: dict[str, Any]):
    seed = int(scenario.get("simulation", {}).get("seed", 0))
    return [create_emitter(cfg, seed=seed) for cfg in scenario.get("emitters", [])]
