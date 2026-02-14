"""
API Routes — Training data upload and fine-tuning job management.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from runanywhere_finetune.config import get_settings
from runanywhere_finetune.database import (
    JobStatus,
    TrainingDataUpload,
    TrainingJob,
    get_session,
)
from runanywhere_finetune.schemas import (
    StartTrainingRequest,
    TrainingConfigRequest,
    TrainingDataUploadRequest,
    TrainingDataUploadResponse,
    TrainingJobResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["training"])


# ─── Upload Training Data ────────────────────────────────────────────────────


@router.post("/upload", response_model=TrainingDataUploadResponse)
async def upload_training_data(
    request: TrainingDataUploadRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    Receive training data from a device.

    The device sends collected user interactions (prompts, responses,
    corrections, ratings) which are stored for later fine-tuning.
    """
    settings = get_settings()
    upload_id = str(uuid.uuid4())

    # Filter to only include training-eligible interactions
    eligible = [ix for ix in request.interactions if ix.include_in_training]
    if not eligible:
        raise HTTPException(
            status_code=400,
            detail="No training-eligible interactions in upload",
        )

    # Save interactions to disk as JSON
    upload_dir = settings.storage.training_data_dir / request.device_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{upload_id}.json"

    upload_data = {
        "upload_id": upload_id,
        "device_id": request.device_id,
        "model_id": request.get_model_id(),
        "model_name": request.model_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "interactions": [ix.model_dump() for ix in eligible],
    }
    with open(upload_path, "w", encoding="utf-8") as f:
        json.dump(upload_data, f, ensure_ascii=False, indent=2)

    # Save metadata to database
    db_upload = TrainingDataUpload(
        id=upload_id,
        device_id=request.device_id,
        model_id=request.get_model_id(),
        sample_count=len(eligible),
        file_path=str(upload_path),
        metadata_json=json.dumps(
            {
                "model_name": request.model_name,
                "total_interactions_received": len(request.interactions),
                "eligible_interactions": len(eligible),
            }
        ),
    )
    db.add(db_upload)
    await db.commit()

    # Count total samples for this device
    stmt = select(func.sum(TrainingDataUpload.sample_count)).where(
        TrainingDataUpload.device_id == request.device_id
    )
    result = await db.execute(stmt)
    total_samples = result.scalar_one_or_none() or 0

    logger.info(
        f"Upload {upload_id}: {len(eligible)} samples from device {request.device_id} "
        f"(total for device: {total_samples})"
    )

    return TrainingDataUploadResponse(
        status="success",
        upload_id=upload_id,
        device_id=request.device_id,
        sample_count=len(eligible),
        samples_received=len(eligible),
        total_samples_for_device=total_samples,
        message=f"Successfully uploaded {len(eligible)} training samples",
    )


# ─── Start Training Job ──────────────────────────────────────────────────────


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(
    request: StartTrainingRequest,
    db: AsyncSession = Depends(get_session),
):
    """
    Start a fine-tuning job.

    Collects all uploaded data for the device (or specific uploads)
    and kicks off LoRA training in a background task.
    """
    settings = get_settings()
    job_id = str(uuid.uuid4())

    # Get upload IDs
    if request.upload_ids:
        upload_ids = request.upload_ids
    else:
        # Get all uploads for this device
        stmt = select(TrainingDataUpload.id).where(
            TrainingDataUpload.device_id == request.device_id
        )
        result = await db.execute(stmt)
        upload_ids = [row[0] for row in result.fetchall()]

    if not upload_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No training data found for device {request.device_id}",
        )

    # Merge config with defaults
    config = request.to_config()
    raw_base_model = request.base_model or settings.training.base_model

    # Resolve GGUF / local model name to HuggingFace repo ID
    from runanywhere_finetune.training_engine import resolve_model_name
    base_model = resolve_model_name(raw_base_model)
    model_id = request.model_id or base_model

    # Create job record
    job = TrainingJob(
        id=job_id,
        device_id=request.device_id,
        model_id=model_id,
        base_model=base_model,
        upload_ids=json.dumps(upload_ids),
        status=JobStatus.PENDING,
        total_epochs=config.epochs,
        config_json=config.model_dump_json(),
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()

    # Enqueue training (the worker picks this up)
    # We import here to avoid circular deps
    from runanywhere_finetune.worker import enqueue_training_job

    await enqueue_training_job(job_id)

    logger.info(
        f"Training job {job_id} created for device {request.device_id} "
        f"with {len(upload_ids)} uploads, base_model={base_model}"
    )

    return TrainingJobResponse(
        job_id=job_id,
        device_id=request.device_id,
        model_id=model_id,
        base_model=base_model,
        status="pending",
        total_epochs=config.epochs,
        created_at=job.created_at,
    )


# ─── Check Job Status ────────────────────────────────────────────────────────


@router.get("/status/{job_id}", response_model=TrainingJobResponse)
async def get_training_status(
    job_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Check the status of a training job."""
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return TrainingJobResponse(
        job_id=job.id,
        device_id=job.device_id,
        model_id=job.model_id,
        base_model=job.base_model,
        status=job.status.value,
        progress=job.progress or 0.0,
        current_epoch=job.current_epoch or 0,
        total_epochs=job.total_epochs or 3,
        current_loss=job.current_loss,
        best_loss=job.best_loss,
        error_message=job.error_message,
        adapter_id=job.adapter_id,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


# ─── List All Jobs ───────────────────────────────────────────────────────────


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    device_id: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """List training jobs, optionally filtered by device or status."""
    stmt = select(TrainingJob).order_by(TrainingJob.created_at.desc())

    if device_id:
        stmt = stmt.where(TrainingJob.device_id == device_id)
    if status:
        stmt = stmt.where(TrainingJob.status == status)

    result = await db.execute(stmt)
    jobs = result.scalars().all()

    return [
        TrainingJobResponse(
            job_id=job.id,
            device_id=job.device_id,
            model_id=job.model_id,
            base_model=job.base_model,
            status=job.status.value,
            progress=job.progress or 0.0,
            current_epoch=job.current_epoch or 0,
            total_epochs=job.total_epochs or 3,
            current_loss=job.current_loss,
            best_loss=job.best_loss,
            error_message=job.error_message,
            adapter_id=job.adapter_id,
            started_at=job.started_at,
            completed_at=job.completed_at,
            created_at=job.created_at,
        )
        for job in jobs
    ]


# ─── Retry Failed Job ────────────────────────────────────────────────────────


@router.post("/retry/{job_id}", response_model=TrainingJobResponse)
async def retry_training_job(
    job_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Retry a failed training job."""
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed/cancelled jobs (current: {job.status.value})",
        )

    # Resolve model name if it's a GGUF name
    from runanywhere_finetune.training_engine import resolve_model_name
    resolved = resolve_model_name(job.base_model)
    job.base_model = resolved
    job.model_id = resolved

    # Reset job state
    job.status = JobStatus.PENDING
    job.progress = 0.0
    job.current_epoch = 0
    job.current_loss = None
    job.best_loss = None
    job.error_message = None
    job.adapter_id = None
    job.started_at = None
    job.completed_at = None
    await db.commit()

    from runanywhere_finetune.worker import enqueue_training_job
    await enqueue_training_job(job_id)

    logger.info(f"Retrying job {job_id} with base_model={resolved}")

    return TrainingJobResponse(
        job_id=job.id,
        device_id=job.device_id,
        model_id=job.model_id,
        base_model=job.base_model,
        status="pending",
        total_epochs=job.total_epochs or 3,
        created_at=job.created_at,
    )


# ─── Retry All Failed Jobs ───────────────────────────────────────────────────


@router.post("/retry-all")
async def retry_all_failed_jobs(
    db: AsyncSession = Depends(get_session),
):
    """Retry all failed training jobs."""
    stmt = select(TrainingJob).where(TrainingJob.status == JobStatus.FAILED)
    result = await db.execute(stmt)
    failed_jobs = result.scalars().all()

    if not failed_jobs:
        return {"message": "No failed jobs to retry", "count": 0}

    from runanywhere_finetune.training_engine import resolve_model_name
    from runanywhere_finetune.worker import enqueue_training_job

    retried = []
    for job in failed_jobs:
        resolved = resolve_model_name(job.base_model)
        job.base_model = resolved
        job.model_id = resolved
        job.status = JobStatus.PENDING
        job.progress = 0.0
        job.current_epoch = 0
        job.current_loss = None
        job.best_loss = None
        job.error_message = None
        job.adapter_id = None
        job.started_at = None
        job.completed_at = None
        retried.append(job.id)

    await db.commit()

    for jid in retried:
        await enqueue_training_job(jid)

    logger.info(f"Retrying {len(retried)} failed jobs")
    return {"message": f"Retried {len(retried)} failed jobs", "count": len(retried), "job_ids": retried}
