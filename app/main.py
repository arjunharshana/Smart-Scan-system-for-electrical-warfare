from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import (
    APP_VERSION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_LOG_LEVEL,
    PROJECT_ROOT,
)
from app.services.simulation_service import service

logging.basicConfig(level=getattr(logging, DEFAULT_LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager handling graceful startup and teardown."""
    logger.info("Initializing SIH26055 V4.0 Tactical Electronic Warfare Application (V4.0 Hybrid Production)...")
    yield
    logger.info("Shutting down simulation runner...")
    await service.pause()
    if service._runner_task and not service._runner_task.done():
        service._runner_task.cancel()


def create_app() -> FastAPI:
    """Factory creating the production FastAPI application."""
    app = FastAPI(
        title="SIH26055: Smart Scan Strategy for Electronic Warfare",
        version=APP_VERSION,
        description="Tactical Cognitive Radio Scan Strategy with V4.1 LSTM-Hybrid Scheduler",
        lifespan=lifespan,
    )

    # Allow CORS for external browser connections across local networks
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API and WebSocket routes
    app.include_router(api_router)
    app.include_router(ws_router)

    # Static presentation dashboard mounting
    dashboard_dir = PROJECT_ROOT / "app" / "dashboard"
    if dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    return app


app = create_app()
