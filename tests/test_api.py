from fastapi.testclient import TestClient

from rf_environment.app.config import DEFAULT_SCENARIO
from rf_environment.app.main import create_app


def test_rest_state_and_metrics():
    app = create_app(scenario_path=str(DEFAULT_SCENARIO), scheduler="sequential")
    with TestClient(app) as client:
        state = client.get("/simulation/state")
        assert state.status_code == 200
        assert "receiver" in state.json()

        started = client.post("/simulation/step", json={"steps": 5})
        assert started.status_code == 200

        metrics = client.get("/metrics")
        assert metrics.status_code == 200

        events = client.get("/events")
        assert events.status_code == 200

        sim_events = client.get("/simulation/events")
        assert sim_events.status_code == 200

        spectrum = client.get("/spectrum")
        assert spectrum.status_code == 200

        rx = client.get("/receiver/state")
        assert rx.status_code == 200

        sched = client.get("/scheduler/state")
        assert sched.status_code == 200

        obs = client.get("/observations")
        assert obs.status_code == 200

        opps = client.get("/opportunities")
        assert opps.status_code == 200

        bench = client.post(
            "/benchmark",
            json={
                "algorithms": ["sequential", "random"],
                "seeds": [1],
                "steps": 20,
            },
        )
        assert bench.status_code == 200
        assert "summary" in bench.json()
