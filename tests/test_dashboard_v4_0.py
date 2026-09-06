from contextlib import contextmanager
from pathlib import Path

from app.api.routes import get_benchmark_comparison, export_mission_report, health_check
from app.config import DEFAULT_V40_CHECKPOINT, DEFAULT_V41_CHECKPOINT, APP_VERSION
from app.services.simulation_service import SimulationService
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.hybrid.hybrid_scheduler import HybridScheduler
from rf_environment.scheduler.rl.checkpoint_v40 import load_v40_checkpoint


@contextmanager
def assert_raises(exc_type: type[BaseException], match: str | None = None):
    try:
        yield
    except exc_type as e:
        if match:
            assert match.lower() in str(e).lower(), f"Expected '{match}' in error message, got: {e}"
        return
    except BaseException as e:
        raise AssertionError(f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")


def test_1_v40_production_checkpoint_exists() -> None:
    """Verifies that the frozen V4.0 production checkpoint exists and is valid."""
    assert DEFAULT_V40_CHECKPOINT.exists(), f"V4.0 production checkpoint missing at {DEFAULT_V40_CHECKPOINT}"
    data = load_v40_checkpoint(DEFAULT_V40_CHECKPOINT)
    meta = data["metadata"]
    assert meta["format_version"] == "4.0"
    assert meta["model_architecture"] == "MLPQNetwork"
    assert meta["architecture"]["output_dim"] == 30
    assert meta["architecture"]["hidden_dim"] == 64
    assert len(data["weights_sha256"]) == 64


def test_2_v41_production_checkpoint_remains_untouched() -> None:
    """Verifies that V4.1 production checkpoint is intact and unchanged."""
    assert DEFAULT_V41_CHECKPOINT.exists(), f"V4.1 production checkpoint missing at {DEFAULT_V41_CHECKPOINT}"
    import hashlib
    with open(DEFAULT_V41_CHECKPOINT, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == "563dd72e7ca3b06f017c2b3cd531a4cc0ec39e45613e5a54b00d725a20afe847"


def test_3_simulation_service_initializes_with_v40_frozen() -> None:
    """Verifies that SimulationService defaults to V4.0 Hybrid with frozen inference."""
    service = SimulationService()
    sched = service.env.scheduler

    assert isinstance(sched, HybridScheduler)
    assert sched.is_pretrained is True
    assert sched.ddqn.train_mode is False
    assert sched.ddqn.epsilon == 0.0
    assert sched.checkpoint_path is not None
    assert Path(sched.checkpoint_path).resolve() == DEFAULT_V40_CHECKPOINT.resolve()

    # Verify telemetry neural model block
    telem = service.latest_telemetry
    assert "neural_model" in telem
    nmod = telem["neural_model"]
    assert nmod["status"] == "PRETRAINED"
    assert nmod["mode"] == "FROZEN_INFERENCE"
    assert nmod["runtime_training"] == "DISABLED"
    assert nmod["checkpoint_name"] == "production_checkpoint.npz"
    assert nmod["checkpoint_sha256"] == sched.checkpoint_sha256


def test_4_missing_checkpoint_fails_loudly() -> None:
    """Verifies that requiring a non-existent checkpoint raises FileNotFoundError loudly."""
    bands = [100e6 + 10e6 + i * 20e6 for i in range(30)]
    with assert_raises(FileNotFoundError, match="V4.0 CHECKPOINT NOT FOUND — INFERENCE UNAVAILABLE"):
        HybridScheduler(
            bands_hz=bands,
            checkpoint_path="/nonexistent/path/production_checkpoint.npz",
            require_checkpoint=True,
        )


def test_5_canonical_8_scenarios_available_and_switchable() -> None:
    """Verifies that list_available_scenarios contains all 8 canonical test scenarios
    and switching between them preserves pre-trained frozen state.
    """
    service = SimulationService()
    scenarios = service.list_available_scenarios()
    sc_ids = {s["id"] for s in scenarios}

    canonical_keys = [
        "1_Seen_Structure",
        "2_Unseen_Permutation",
        "3_Unseen_Phase",
        "4_Unseen_Dwell",
        "5_Unseen_Subset",
        "6_Mixed_Shift",
        "7_Random_Hopping",
        "8_Periodic_Burst",
    ]
    for ck in canonical_keys:
        assert ck in sc_ids, f"Canonical scenario '{ck}' missing from catalog"

    # Switch scenario and verify frozen state
    service.reset(seed=123, scenario_name="2_Unseen_Permutation", scheduler_name="hybrid_v4")
    sched = service.env.scheduler
    assert sched.is_pretrained is True
    assert sched.ddqn.train_mode is False
    assert sched.ddqn.epsilon == 0.0

    service.reset(seed=456, scenario_name="7_Random_Hopping", scheduler_name="hybrid_v4")
    sched2 = service.env.scheduler
    assert sched2.is_pretrained is True
    assert sched2.ddqn.train_mode is False
    assert sched2.ddqn.epsilon == 0.0


def test_6_telemetry_payload_and_latency_budget() -> None:
    """Verifies that stepping generates complete telemetry adhering to latency budget."""
    service = SimulationService()
    service.reset(seed=42, scenario_name="1_Seen_Structure", scheduler_name="hybrid_v4")

    # Step simulation
    telem = service.step(5)
    assert telem["system_status"]["step"] == 4
    assert telem["system_status"]["version"] == "4.0"

    # Latency budget verification
    assert "latency" in telem
    lat = telem["latency"]
    assert lat["budget_ms"] == 10.0
    assert lat["step_latency_ms"] < 10.0
    assert "WITHIN BUDGET" in lat["status"]

    # Explainability & arbitration verification
    assert "arbitration" in telem
    arb = telem["arbitration"]
    assert "mode" in arb
    assert "ddqn_weight_pct" in arb
    assert "ca_weight_pct" in arb
    assert "explanation" in arb
    assert len(arb["explanation"]) > 10

    # Top predictions verification
    pred = telem["primary_prediction"]
    assert len(pred["top_predictions"]) <= 5
    assert "confidence_pct" in pred


def test_7_benchmark_and_export_api_endpoints() -> None:
    """Verifies that /api/benchmark and /api/export return valid data."""
    # Benchmark endpoint
    bench_data = get_benchmark_comparison()
    assert "schedulers" in bench_data
    schedulers = bench_data["schedulers"]
    assert schedulers[0]["id"] == "hybrid_v4"
    assert schedulers[0]["overall_ir_pct"] == 35.19
    assert schedulers[0]["rank"] == 1

    whittle = next(s for s in schedulers if s["id"] == "whittle_style")
    assert whittle["overall_ir_pct"] == 34.40

    ca = next(s for s in schedulers if s["id"] == "context_aware")
    assert ca["overall_ir_pct"] == 31.15

    v41 = next(s for s in schedulers if s["id"] == "hybrid_v41")
    assert v41["overall_ir_pct"] == 26.90

    v5 = next(s for s in schedulers if s["id"] == "v5_belief")
    assert v5["overall_ir_pct"] == 15.36

    assert len(bench_data["scenario_breakdown"]) == 8

    # Export endpoint
    export_data = export_mission_report()
    assert "mission_report" in export_data
    rep = export_data["mission_report"]
    assert "performance_metrics" in rep
    assert "neural_model_audit" in rep
    assert rep["system_status"]["version"] == "4.0"

    # Health check endpoint
    h = health_check()
    assert h["status"] == "ok"
    assert h["version"] == "4.0"


def test_8_ground_truth_isolation_contract() -> None:
    """Verifies strict mathematical firewall: scheduler never receives ground-truth objects."""
    service = SimulationService()
    service.reset(seed=42, scenario_name="1_Seen_Structure", scheduler_name="hybrid_v4")
    sched = service.env.scheduler

    # Execute steps and inspect observation
    res = service.env.step()
    obs = res.observation

    assert isinstance(obs, SchedulerObservation)
    # Check that ground-truth attributes are not on SchedulerObservation
    assert not hasattr(obs, "emitters")
    assert not hasattr(obs, "carrier_frequency_hz")
    assert not hasattr(obs, "true_power")
    assert not hasattr(obs, "sinr")
