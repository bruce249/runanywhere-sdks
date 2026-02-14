"""
RunAnywhere OTA Fine-Tuning Server — Main Application

Turn your laptop into a fine-tuning cloud.
Devices upload interaction data → server fine-tunes LoRA adapters →
devices download adapters OTA for personalized inference.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from runanywhere_finetune.config import get_settings
from runanywhere_finetune.database import init_db
from runanywhere_finetune.routes_adapters import router as adapters_router
from runanywhere_finetune.routes_dashboard import router as dashboard_router
from runanywhere_finetune.routes_training import router as training_router
from runanywhere_finetune.worker import start_worker

logger = logging.getLogger("runanywhere_finetune")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RunAnywhere Fine-Tuning Server",
        description=(
            "OTA fine-tuning cloud server for RunAnywhere SDK. "
            "Upload device interaction data, fine-tune LoRA adapters, "
            "and deliver them over-the-air to your devices."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(training_router)
    app.include_router(adapters_router)
    app.include_router(dashboard_router)

    # Startup/shutdown events
    @app.on_event("startup")
    async def on_startup():
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, settings.server.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

        logger.info("=" * 60)
        logger.info("RunAnywhere OTA Fine-Tuning Server starting...")
        logger.info(f"  Server: http://{settings.server.host}:{settings.server.port}")
        logger.info(f"  Docs:   http://{settings.server.host}:{settings.server.port}/docs")
        logger.info(f"  Data:   {settings.storage.data_dir.resolve()}")
        logger.info(f"  Model:  {settings.training.base_model}")
        logger.info(f"  LoRA:   rank={settings.training.lora_rank}, "
                     f"alpha={settings.training.lora_alpha}")
        logger.info("=" * 60)

        # Initialize database
        await init_db(str(settings.storage.db_path))
        logger.info(f"Database initialized: {settings.storage.db_path}")

        # Start background worker
        asyncio.create_task(start_worker())
        logger.info("Background training worker started")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Server shutting down...")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "RunAnywhere Fine-Tuning Server",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "stats": "/api/v1/stats",
        }

    return app


# ─── Entry Point ─────────────────────────────────────────────────────────────

app = create_app()


def main():
    """Run the server."""
    settings = get_settings()
    uvicorn.run(
        "runanywhere_finetune.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        log_level=settings.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
