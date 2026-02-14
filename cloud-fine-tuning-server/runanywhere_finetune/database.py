"""
Database models and setup — SQLite via SQLAlchemy async.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ─── Enums ───────────────────────────────────────────────────────────────────


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    CONVERTING = "converting"  # GGUF conversion
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AdapterFormat(str, enum.Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    BIN = "bin"


# ─── ORM Models ──────────────────────────────────────────────────────────────


class TrainingDataUpload(Base):
    """Record of a data upload from a device."""

    __tablename__ = "training_data_uploads"

    id = Column(String, primary_key=True)
    device_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metadata_json = Column(Text, nullable=True)


class TrainingJob(Base):
    """A fine-tuning job."""

    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True)
    device_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    upload_ids = Column(Text, nullable=False)  # JSON list of upload IDs
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0.0 - 1.0
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=3)
    current_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)
    config_json = Column(Text, nullable=True)  # Training config
    error_message = Column(Text, nullable=True)
    adapter_id = Column(String, nullable=True)  # Result adapter ID
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TrainedAdapter(Base):
    """A trained LoRA adapter ready for OTA delivery."""

    __tablename__ = "trained_adapters"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    device_id = Column(String, nullable=False, index=True)
    model_id = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    format = Column(Enum(AdapterFormat), nullable=False, default=AdapterFormat.GGUF)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, default=0)
    lora_rank = Column(Integer, default=8)
    lora_alpha = Column(Integer, default=16)
    target_modules = Column(Text, nullable=True)  # JSON list
    training_samples = Column(Integer, default=0)
    final_loss = Column(Float, nullable=True)
    version = Column(Integer, default=1)
    checksum = Column(String, nullable=True)  # SHA-256
    is_published = Column(Integer, default=1)  # SQLite bool
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    downloaded_count = Column(Integer, default=0)
    last_downloaded_at = Column(DateTime, nullable=True)


# ─── Database Setup ──────────────────────────────────────────────────────────

_engine = None
_session_factory = None


async def init_db(db_path: str) -> None:
    """Initialize the database engine and create tables."""
    global _engine, _session_factory
    _engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get an async database session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        yield session


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory for manual session management."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory
