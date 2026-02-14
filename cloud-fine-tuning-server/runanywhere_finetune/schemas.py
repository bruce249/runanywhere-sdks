"""
Pydantic schemas for API request/response models.

Mirrors the existing RunAnywhere domain models (FineTuningModels.kt)
so device-side code can serialize/deserialize seamlessly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ─── Training Data Schemas ───────────────────────────────────────────────────


class ContextMessage(BaseModel):
    """A message in conversation context."""

    role: str  # "user", "assistant", "system"
    content: str


class TrainingInteraction(BaseModel):
    """A captured user interaction — matches Kotlin TrainingInteraction."""

    id: str
    user_prompt: str
    model_response: str
    corrected_response: str | None = None
    rating: int | None = None
    is_positive_example: bool = False
    include_in_training: bool = True
    model_id: str | None = None
    model_name: str | None = None
    conversation_context: list[ContextMessage] | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: int = 0
    frequency_count: int = 1
    data_source: str = "text_chat"  # "text_chat", "voice_chat", "video_analysis"


class TrainingDataUploadRequest(BaseModel):
    """Request body for uploading training data from a device."""

    device_id: str
    model_id: str = ""
    model_name: str | None = None
    base_model: str | None = None  # Alias used by Android app
    interactions: list[TrainingInteraction]
    config: TrainingConfigRequest | None = None

    def get_model_id(self) -> str:
        """Return model_id, falling back to base_model."""
        return self.model_id or self.base_model or "unknown"


class TrainingDataUploadResponse(BaseModel):
    """Response after successful data upload."""

    status: str = "success"
    upload_id: str
    device_id: str = ""
    sample_count: int = 0
    samples_received: int = 0  # Alias for Android app compatibility
    total_samples_for_device: int = 0
    message: str = ""


# ─── Training Job Schemas ────────────────────────────────────────────────────


class TrainingConfigRequest(BaseModel):
    """Fine-tuning configuration — matches Kotlin FineTuningConfig."""

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_4bit_quantization: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class StartTrainingRequest(BaseModel):
    """Request to start a fine-tuning job."""

    device_id: str
    model_id: str = ""
    base_model: str | None = None  # Falls back to server default
    upload_ids: list[str] | None = None  # Specific uploads; None = all for device
    adapter_name: str | None = None
    config: TrainingConfigRequest | None = None
    # Android app sends these at top-level (not nested in config)
    lora_rank: int | None = None
    lora_alpha: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    output_format: str | None = None

    def to_config(self) -> TrainingConfigRequest:
        """Merge top-level Android fields into a TrainingConfigRequest."""
        base = self.config or TrainingConfigRequest()
        return base.model_copy(
            update={
                k: v
                for k, v in {
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                }.items()
                if v is not None
            }
        )


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    CONVERTING = "converting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJobResponse(BaseModel):
    """Training job status response."""

    job_id: str
    device_id: str
    model_id: str
    base_model: str
    status: JobStatusEnum
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 3
    current_loss: float | None = None
    best_loss: float | None = None
    error_message: str | None = None
    adapter_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None


# ─── Adapter Schemas ─────────────────────────────────────────────────────────


class AdapterFormatEnum(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    BIN = "bin"


class AdapterInfo(BaseModel):
    """Adapter metadata — matches Kotlin LoraAdapter."""

    id: str
    name: str
    description: str | None = None
    device_id: str
    model_id: str
    base_model: str
    format: AdapterFormatEnum = AdapterFormatEnum.GGUF
    file_size_bytes: int = 0
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    training_samples: int = 0
    final_loss: float | None = None
    version: int = 1
    checksum: str | None = None
    is_published: bool = True
    created_at: datetime | None = None


class AdapterPollResponse(BaseModel):
    """Response when device polls for new adapters."""

    has_new_adapters: bool
    adapters: list[AdapterInfo] = Field(default_factory=list)
    poll_interval_seconds: int = 60


class AdapterListResponse(BaseModel):
    """List of all adapters."""

    adapters: list[AdapterInfo]
    total: int


# ─── Dashboard / Stats ──────────────────────────────────────────────────────


class ServerStatsResponse(BaseModel):
    """Server statistics for dashboard."""

    total_uploads: int = 0
    total_samples: int = 0
    total_jobs: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_adapters: int = 0
    total_adapter_downloads: int = 0
    devices: list[str] = Field(default_factory=list)
