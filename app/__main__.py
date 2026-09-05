from __future__ import annotations

import argparse
import sys
import uvicorn

from app.config import DEFAULT_HOST, DEFAULT_PORT
from app.main import create_app
from app.services.simulation_service import service


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26055 Electronic Warfare Tactical Scanning Application")
    sub = parser.add_subparsers(dest="command")

    # Serve command (Default production entrypoint)
    serve_p = sub.add_parser("serve", help="Start the production API and Web Dashboard")
    serve_p.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Headless run command
    run_p = sub.add_parser("run", help="Run scenario headlessly and print final telemetry")
    run_p.add_argument("--scenario", default=None, help="Scenario filename to run")
    run_p.add_argument("--scheduler", default="hybrid_v41", help="Scheduler algorithm (default: hybrid_v41)")
    run_p.add_argument("--steps", type=int, default=300, help="Number of steps to run (default: 300)")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Default to serve if no command provided
    if args.command is None or args.command == "serve":
        host = getattr(args, "host", DEFAULT_HOST)
        port = getattr(args, "port", DEFAULT_PORT)
        reload = getattr(args, "reload", False)
        print(f"\n==========================================================================")
        print(f"📡 SIH26055 EW SMART SCAN STRATEGY — V4.1 TACTICAL DASHBOARD")
        print(f"   Server listening on: http://{host}:{port}")
        print(f"   Health Check:        http://{host}:{port}/health")
        print(f"   WebSocket Stream:    ws://{host}:{port}/ws/telemetry")
        print(f"==========================================================================\n")
        uvicorn.run("app.main:app", host=host, port=port, reload=reload)

    elif args.command == "run":
        service.reset(seed=args.seed, scenario_name=args.scenario, scheduler_name=args.scheduler)
        print(f"Running {args.steps} steps with {args.scheduler} on {service.scenario_name} (seed={args.seed})...")
        telem = service.step(count=args.steps)
        perf = telem.get("performance", {})
        print(f"\nCompleted {args.steps} steps:")
        print(f"   Interception Ratio: {perf.get('interception_ratio_pct')}%")
        print(f"   Detection Rate:     {perf.get('detection_rate_pct')}%")
        print(f"   Scan Efficiency:    {perf.get('scan_efficiency_pct')}%")


if __name__ == "__main__":
    main()
