from __future__ import annotations

from fastapi.testclient import TestClient
from app.main import app


def test_1_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["version"] == "4.1"
        assert data["scheduler"] == "V4.1"
        assert "timestamp" in data


def test_2_simulation_status_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert "state" in data
        assert "time_step" in data
        assert "simulation_time_s" in data
        assert data["version"] == "4.1"
        assert "hybrid_v41" in data["scheduler_type"]


def test_3_telemetry_schema_and_non_leakage():
    with TestClient(app) as client:
        # Step once to populate telemetry
        client.post("/api/simulation/step", json={"steps": 1})
        res = client.get("/api/telemetry")
        assert res.status_code == 200
        telem = res.json()

        # Check essential blocks
        assert "system_status" in telem
        assert "primary_prediction" in telem
        assert "detector" in telem
        assert "performance" in telem
        assert "arbitration" in telem
        assert "recent_timeline" in telem

        # Primary prediction checks
        pred = telem["primary_prediction"]
        assert "current_scan_mhz" in pred
        assert "predicted_next_mhz" in pred
        assert "confidence_pct" in pred
        assert 0.0 <= pred["confidence_pct"] <= 100.0

        # Ground truth firewall in operational telemetry:
        # The operational telemetry must NOT expose emitter real identities or ground-truth frequencies to the scheduler view
        assert "associated_emitter_id" not in pred
        assert "true_emitter_frequency" not in pred


def test_4_simulation_stepping_advancement():
    with TestClient(app) as client:
        client.post("/api/simulation/reset", json={"seed": 42})
        status_0 = client.get("/api/status").json()
        t0 = status_0["time_step"]

        client.post("/api/simulation/step", json={"steps": 5})
        status_1 = client.get("/api/status").json()
        t1 = status_1["time_step"]

        assert t1 == t0 + 5


def test_5_simulation_reset_with_seed_determinism():
    with TestClient(app) as client:
        # Run seed 123
        client.post("/api/simulation/reset", json={"seed": 123})
        res_a = client.post("/api/simulation/step", json={"steps": 10}).json()
        scans_a = [e["scanned_mhz"] for e in res_a["recent_timeline"]]

        # Reset with same seed 123
        client.post("/api/simulation/reset", json={"seed": 123})
        res_b = client.post("/api/simulation/step", json={"steps": 10}).json()
        scans_b = [e["scanned_mhz"] for e in res_b["recent_timeline"]]

        assert scans_a == scans_b, "Simulation runs with identical seed must produce identical scan trajectories"


def test_6_scenario_catalog_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/scenarios")
        assert res.status_code == 200
        scenarios = res.json()
        assert len(scenarios) > 0
        filenames = [s["filename"] for s in scenarios]
        assert "frequency_agile.yaml" in filenames or "deterministic_hopping.yaml" in filenames


def test_7_speed_configuration_endpoint():
    with TestClient(app) as client:
        for speed in ["0.5x", "1x", "2x", "5x", "max"]:
            res = client.post("/api/simulation/speed", json={"speed": speed})
            assert res.status_code == 200
            assert res.json()["speed"] == speed


def test_8_static_dashboard_files_served():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/css/dashboard.css").status_code == 200
        assert client.get("/js/waterfall.js").status_code == 200
        assert client.get("/js/dashboard.js").status_code == 200
