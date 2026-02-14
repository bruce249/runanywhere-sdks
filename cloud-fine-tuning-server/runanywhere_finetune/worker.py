"""
Background Training Worker — processes fine-tuning jobs asynchronously.

Runs in a separate thread to avoid blocking the FastAPI event loop.
Jobs are picked up from the database queue and processed sequentially.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from runanywhere_finetune.config import get_settings
from runanywhere_finetune.database import (
    AdapterFormat,
    JobStatus,
    TrainedAdapter,
    TrainingDataUpload,
    TrainingJob,
    get_session_factory,
)
from runanywhere_finetune.preprocessing import DataPreprocessor
from runanywhere_finetune.training_engine import FineTuningEngine, TrainingProgress, compute_file_checksum, resolve_model_name

logger = logging.getLogger(__name__)

# Job queue
_job_queue: asyncio.Queue[str] = asyncio.Queue()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finetune-worker")


async def enqueue_training_job(job_id: str) -> None:
    """Add a job to the processing queue."""
    await _job_queue.put(job_id)
    logger.info(f"Job {job_id} enqueued for training")


async def recover_orphaned_jobs() -> None:
    """Re-queue jobs that were in-progress when the server last shut down.

    Called at startup so that pending/preprocessing/training jobs that
    never finished are automatically retried.
    """
    try:
        session_factory = get_session_factory()
    except RuntimeError:
        logger.warning("Database not ready for orphan recovery — skipping")
        return

    async with session_factory() as db:
        stmt = select(TrainingJob).where(
            TrainingJob.status.in_([
                JobStatus.PENDING,
                JobStatus.PREPROCESSING,
                JobStatus.TRAINING,
            ])
        )
        result = await db.execute(stmt)
        orphans = result.scalars().all()

        if not orphans:
            logger.info("No orphaned jobs to recover")
            return

        for job in orphans:
            # Reset status to PENDING so the worker picks them up fresh
            job.status = JobStatus.PENDING
            job.progress = 0.0
            job.current_epoch = 0
            job.current_loss = None
            job.started_at = None
            await _job_queue.put(job.id)
            logger.info(f"Recovered orphaned job {job.id} (was {job.status})")

        await db.commit()
        logger.info(f"Recovered {len(orphans)} orphaned job(s)")


async def start_worker() -> None:
    """Start the background worker loop."""
    logger.info("Fine-tuning worker started — recovering orphaned jobs...")
    await recover_orphaned_jobs()
    logger.info("Worker ready — waiting for jobs...")
    while True:
        try:
            job_id = await _job_queue.get()
            logger.info(f"Worker picked up job: {job_id}")
            await _process_job(job_id)
            _job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Worker cancelled")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)


async def _process_job(job_id: str) -> None:
    """Process a single training job."""
    settings = get_settings()
    session_factory = get_session_factory()

    async with session_factory() as db:
        # Load job
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await db.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            logger.error(f"Job {job_id} not found in database")
            return

        try:
            # Update status
            job.status = JobStatus.PREPROCESSING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            # Get upload file paths
            upload_ids = json.loads(job.upload_ids)
            stmt = select(TrainingDataUpload).where(TrainingDataUpload.id.in_(upload_ids))
            result = await db.execute(stmt)
            uploads = result.scalars().all()

            upload_paths = [u.file_path for u in uploads]
            total_samples = sum(u.sample_count for u in uploads)

            if not upload_paths:
                raise ValueError("No upload files found for this job")

            logger.info(
                f"Job {job_id}: preprocessing {len(upload_paths)} uploads, "
                f"{total_samples} samples"
            )

            # Preprocess data
            preprocessor = DataPreprocessor(settings.storage.training_data_dir)
            dataset_rows = preprocessor.prepare_dataset(
                upload_paths=upload_paths,
                template="chatml",
            )

            if not dataset_rows:
                raise ValueError("No valid training samples after preprocessing")

            # Save prepared dataset
            dataset_path = (
                settings.storage.training_data_dir / "prepared" / f"{job_id}.jsonl"
            )
            preprocessor.save_prepared_dataset(dataset_rows, dataset_path)

            # Parse training config
            config = json.loads(job.config_json) if job.config_json else {}

            # Update status to training
            job.status = JobStatus.TRAINING
            await db.commit()

            # Resolve GGUF/local model name to HuggingFace repo ID
            resolved_model = resolve_model_name(job.base_model)
            if resolved_model != job.base_model:
                logger.info(f"Job {job_id}: resolved model '{job.base_model}' → '{resolved_model}'")
                job.base_model = resolved_model
                await db.commit()

            # Run training in thread pool (blocking operation)
            engine = FineTuningEngine(
                base_model=resolved_model,
                output_dir=settings.storage.adapters_dir,
                checkpoints_dir=settings.storage.checkpoints_dir,
            )

            # Progress callback that updates DB
            async def update_progress(progress: TrainingProgress) -> None:
                async with session_factory() as progress_db:
                    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
                    r = await progress_db.execute(stmt)
                    j = r.scalar_one_or_none()
                    if j:
                        j.progress = progress.progress
                        j.current_epoch = progress.current_epoch
                        j.current_loss = progress.current_loss
                        if progress.best_loss is not None:
                            j.best_loss = progress.best_loss
                        await progress_db.commit()

            def sync_progress_callback(progress: TrainingProgress) -> None:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(update_progress(progress), loop)
                except Exception:
                    pass  # Non-critical

            engine.set_progress_callback(sync_progress_callback)

            # Run training (blocking)
            loop = asyncio.get_event_loop()
            training_result = await loop.run_in_executor(
                _executor,
                lambda: engine.train(
                    dataset_path=dataset_path,
                    job_id=job_id,
                    lora_rank=config.get("lora_rank", settings.training.lora_rank),
                    lora_alpha=config.get("lora_alpha", settings.training.lora_alpha),
                    lora_dropout=config.get("lora_dropout", settings.training.lora_dropout),
                    learning_rate=config.get("learning_rate", settings.training.learning_rate),
                    epochs=config.get("epochs", settings.training.training_epochs),
                    batch_size=config.get("batch_size", settings.training.batch_size),
                    max_seq_length=config.get("max_seq_length", settings.training.max_seq_length),
                    gradient_accumulation_steps=config.get(
                        "gradient_accumulation_steps",
                        settings.training.gradient_accumulation_steps,
                    ),
                    target_modules=config.get(
                        "target_modules", settings.training.target_modules
                    ),
                    use_4bit=config.get("use_4bit_quantization", True),
                    auto_convert_gguf=settings.training.auto_convert_gguf,
                ),
            )

            if not training_result.success:
                raise RuntimeError(training_result.error or "Training failed")

            # Determine adapter file path
            adapter_file = training_result.gguf_path or (
                training_result.adapter_dir / "adapter_model.safetensors"
                if training_result.adapter_dir
                else None
            )

            if adapter_file is None or not adapter_file.exists():
                raise RuntimeError("No adapter file produced by training")

            # Determine format
            adapter_format = AdapterFormat.GGUF
            if adapter_file.suffix == ".safetensors":
                adapter_format = AdapterFormat.SAFETENSORS
            elif adapter_file.suffix == ".bin":
                adapter_format = AdapterFormat.BIN

            # Compute checksum
            checksum = compute_file_checksum(adapter_file)
            file_size = adapter_file.stat().st_size

            # Check for existing adapter version for this device+model
            stmt = select(TrainedAdapter).where(
                TrainedAdapter.device_id == job.device_id,
                TrainedAdapter.model_id == job.model_id,
            ).order_by(TrainedAdapter.version.desc())
            result = await db.execute(stmt)
            existing = result.scalars().first()
            version = (existing.version + 1) if existing else 1

            # Create adapter record
            adapter_id = str(uuid.uuid4())
            adapter = TrainedAdapter(
                id=adapter_id,
                name=f"LoRA adapter v{version} for {job.model_id}",
                description=(
                    f"Trained from {total_samples} samples, "
                    f"loss={training_result.final_loss:.4f}"
                    if training_result.final_loss
                    else f"Trained from {total_samples} samples"
                ),
                device_id=job.device_id,
                model_id=job.model_id,
                base_model=job.base_model,
                job_id=job_id,
                format=adapter_format,
                file_path=str(adapter_file),
                file_size_bytes=file_size,
                lora_rank=config.get("lora_rank", settings.training.lora_rank),
                lora_alpha=config.get("lora_alpha", settings.training.lora_alpha),
                target_modules=json.dumps(
                    config.get("target_modules", settings.training.target_modules)
                ),
                training_samples=training_result.sample_count,
                final_loss=training_result.final_loss,
                version=version,
                checksum=checksum,
                is_published=1,
                created_at=datetime.now(timezone.utc),
            )
            db.add(adapter)

            # Update job
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.current_loss = training_result.final_loss
            job.best_loss = training_result.best_loss
            job.adapter_id = adapter_id
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(
                f"Job {job_id} COMPLETED — adapter={adapter_id}, "
                f"loss={training_result.final_loss}, "
                f"file={adapter_file} ({file_size} bytes), "
                f"time={training_result.training_time_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Job {job_id} FAILED: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
