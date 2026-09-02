from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rf_environment.api.routes import router, service
from rf_environment.app.config import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SCENARIO
from rf_environment.environment.scenario import load_scenario


@asynccontextmanager
async def lifespan(app: FastAPI):
    scenario_path = getattr(app.state, "scenario_path", DEFAULT_SCENARIO)
    scheduler = getattr(app.state, "scheduler", None)
    service.load_scenario(load_scenario(scenario_path), scheduler_name=scheduler)
    yield


def create_app(scenario_path: str | None = None, scheduler: str | None = None) -> FastAPI:
    app = FastAPI(title="SIH26055 RF Environment", version="0.1.0", lifespan=lifespan)
    app.state.scenario_path = scenario_path or DEFAULT_SCENARIO
    app.state.scheduler = scheduler
    app.include_router(router)
    return app


app = create_app()


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="SIH26055 RF environment")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run a scenario headlessly")
    run_p.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    run_p.add_argument("--scheduler", default="sequential")
    run_p.add_argument("--steps", type=int, default=None)
    serve_p = sub.add_parser("serve", help="Start the REST/WebSocket API")
    serve_p.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    serve_p.add_argument("--scheduler", default="sequential")
    serve_p.add_argument("--host", default=DEFAULT_HOST)
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if args.command == "run":
        from rf_environment.environment.builder import build_from_path

        env = build_from_path(args.scenario, scheduler_name=args.scheduler)
        env.run(steps=args.steps)
        print(env.metrics.snapshot(max(env.clock.time_step, 0)).model_dump_json(indent=2))
        return
    import uvicorn

    application = create_app(scenario_path=args.scenario, scheduler=args.scheduler)
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    cli_main()
