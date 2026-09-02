from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SCENARIO_DIR = PACKAGE_ROOT / "scenarios"
DEFAULT_SCENARIO = SCENARIO_DIR / "basic.yaml"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
