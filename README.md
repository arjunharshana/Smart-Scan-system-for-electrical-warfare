# SIH26055 — Smart Scan Strategy for Electronic Warfare

Modular **Python** RF environment for SIH26055. Emitters are composed from independent type, frequency behavior, and time behavior. The environment generates ground truth; the receiver observes only its tuned band; schedulers learn from observations and rewards, never from truth.

GNU Radio is an optional future `SignalSource`. The scheduler, API, and metrics do not depend on it.

## Layout

```text
rf_environment/
  domain/                 serializable state models
  emitters/               radar / communication + factory
  frequency_behaviors/    fixed, hop, sweep, random
  time_behaviors/         continuous, periodic, burst, intermittent
  channel/                noise, path loss, interference
  signal/                 PythonSignalSource (GNU Radio later)
  receiver/               tuner, receiver, detector
  environment/            clock, scenario loader, RFEnvironment
  scheduler/              sequential, random, UCB1, Thompson Sampling
  rewards/                hit / miss / false-alarm costs
  metrics/                event-derived metrics
  api/                    REST + WebSocket
  visualization/          event stream for a future dashboard
  scenarios/              YAML worlds
  experiments/            scheduler comparison runner
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run a scenario

```bash
python -m rf_environment.app.main run --scenario rf_environment/scenarios/basic.yaml --scheduler ucb1
```

## API

```bash
python -m rf_environment.app.main serve --scenario rf_environment/scenarios/mixed_environment.yaml
```

REST:

- `GET /simulation/state`
- `POST /simulation/start|pause|stop|reset`
- `GET|POST|PUT|DELETE /emitters`
- `GET /spectrum` `/ground-truth` `/receiver` `/scheduler` `/metrics` `/events` `/waterfall`

WebSocket: `ws://localhost:8000/ws/simulation` streams simulation events.

Ground-truth endpoints are for the operator dashboard. Schedulers only receive `Observation`.

## Scenarios

YAML files under `rf_environment/scenarios/` compose emitters without code changes:

- `basic.yaml` — fixed radar + hopping burst communication
- `periodic.yaml` — sweep radar + periodic communication
- `frequency_agile.yaml` — hop / random / sweep mix
- `mixed_environment.yaml` — six emitters covering the spec combinations

## Design rule

Environment generates reality → receiver observes reality → scheduler decides → reward updates the scheduler → metrics evaluate → events feed the API/dashboard.
