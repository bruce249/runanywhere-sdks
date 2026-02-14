"""
API Routes — Adapter management and OTA delivery.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from runanywhere_finetune.database import TrainedAdapter, get_session
from runanywhere_finetune.schemas import (
    AdapterFormatEnum,
    AdapterInfo,
    AdapterListResponse,
    AdapterPollResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/adapters", tags=["adapters"])


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _adapter_to_info(adapter: TrainedAdapter) -> AdapterInfo:
    """Convert DB model to API response."""
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if adapter.target_modules:
        try:
            target_modules = json.loads(adapter.target_modules)
        except json.JSONDecodeError:
            pass

    return AdapterInfo(
        id=adapter.id,
        name=adapter.name,
        description=adapter.description,
        device_id=adapter.device_id,
        model_id=adapter.model_id,
        base_model=adapter.base_model,
        format=adapter.format.value if adapter.format else AdapterFormatEnum.GGUF,
        file_size_bytes=adapter.file_size_bytes or 0,
        lora_rank=adapter.lora_rank or 8,
        lora_alpha=adapter.lora_alpha or 16,
        target_modules=target_modules,
        training_samples=adapter.training_samples or 0,
        final_loss=adapter.final_loss,
        version=adapter.version or 1,
        checksum=adapter.checksum,
        is_published=bool(adapter.is_published),
        created_at=adapter.created_at,
    )


# ─── Poll for New Adapters (Device calls this) ──────────────────────────────


@router.get("/poll", response_model=AdapterPollResponse)
async def poll_adapters(
    device_id: str = Query(..., description="Device ID polling for updates"),
    model_id: str | None = Query(None, description="Filter by model ID"),
    since: str | None = Query(None, description="ISO timestamp — only return adapters newer"),
    db: AsyncSession = Depends(get_session),
):
    """
    Device polls this endpoint to check for new adapters.

    This is the core OTA mechanism. The device periodically calls this
    to discover newly trained adapters ready for download.
    """
    stmt = select(TrainedAdapter).where(
        TrainedAdapter.device_id == device_id,
        TrainedAdapter.is_published == 1,
    )

    if model_id:
        stmt = stmt.where(TrainedAdapter.model_id == model_id)

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            stmt = stmt.where(TrainedAdapter.created_at > since_dt)
        except ValueError:
            pass

    stmt = stmt.order_by(TrainedAdapter.created_at.desc())
    result = await db.execute(stmt)
    adapters = result.scalars().all()

    adapter_infos = [_adapter_to_info(a) for a in adapters]

    return AdapterPollResponse(
        has_new_adapters=len(adapter_infos) > 0,
        adapters=adapter_infos,
        poll_interval_seconds=60,
    )


# ─── List All Adapters ──────────────────────────────────────────────────────


@router.get("/list", response_model=AdapterListResponse)
async def list_adapters(
    device_id: str | None = Query(None),
    model_id: str | None = Query(None),
    db: AsyncSession = Depends(get_session),
):
    """List all trained adapters."""
    stmt = select(TrainedAdapter).order_by(TrainedAdapter.created_at.desc())

    if device_id:
        stmt = stmt.where(TrainedAdapter.device_id == device_id)
    if model_id:
        stmt = stmt.where(TrainedAdapter.model_id == model_id)

    result = await db.execute(stmt)
    adapters = result.scalars().all()

    adapter_infos = [_adapter_to_info(a) for a in adapters]
    return AdapterListResponse(adapters=adapter_infos, total=len(adapter_infos))


# ─── Download Adapter File ──────────────────────────────────────────────────


@router.get("/download/{adapter_id}")
async def download_adapter(
    adapter_id: str,
    db: AsyncSession = Depends(get_session),
):
    """
    Download an adapter file.

    Returns the adapter binary (GGUF or safetensors) as a streaming file response.
    The device downloads this and loads it into llama.cpp.
    """
    stmt = select(TrainedAdapter).where(TrainedAdapter.id == adapter_id)
    result = await db.execute(stmt)
    adapter = result.scalar_one_or_none()

    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")

    file_path = Path(adapter.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Adapter file not found on disk: {file_path}",
        )

    # Update download stats
    adapter.downloaded_count = (adapter.downloaded_count or 0) + 1
    adapter.last_downloaded_at = datetime.now(timezone.utc)
    await db.commit()

    # Determine media type
    suffix = file_path.suffix.lower()
    media_type = {
        ".gguf": "application/octet-stream",
        ".safetensors": "application/octet-stream",
        ".bin": "application/octet-stream",
    }.get(suffix, "application/octet-stream")

    logger.info(
        f"Device downloading adapter {adapter_id} ({adapter.name}, "
        f"{adapter.file_size_bytes} bytes)"
    )

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
        headers={
            "X-Adapter-Id": adapter.id,
            "X-Adapter-Name": adapter.name,
            "X-Adapter-Checksum": adapter.checksum or "",
            "X-Adapter-Version": str(adapter.version or 1),
            "X-LoRA-Rank": str(adapter.lora_rank or 8),
            "X-LoRA-Alpha": str(adapter.lora_alpha or 16),
        },
    )


# ─── Get Adapter Info ────────────────────────────────────────────────────────


@router.get("/{adapter_id}", response_model=AdapterInfo)
async def get_adapter(
    adapter_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Get detailed info about a specific adapter."""
    stmt = select(TrainedAdapter).where(TrainedAdapter.id == adapter_id)
    result = await db.execute(stmt)
    adapter = result.scalar_one_or_none()

    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")

    return _adapter_to_info(adapter)


# ─── Delete Adapter ─────────────────────────────────────────────────────────


@router.delete("/{adapter_id}")
async def delete_adapter(
    adapter_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Delete an adapter (metadata + file)."""
    stmt = select(TrainedAdapter).where(TrainedAdapter.id == adapter_id)
    result = await db.execute(stmt)
    adapter = result.scalar_one_or_none()

    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")

    # Delete file
    file_path = Path(adapter.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete from DB
    await db.delete(adapter)
    await db.commit()

    logger.info(f"Deleted adapter {adapter_id} ({adapter.name})")
    return {"status": "deleted", "adapter_id": adapter_id}
