"""
RunAnywhere OTA Fine-Tuning Server — Configuration

All settings with sensible defaults for running on your laptop.
Override via environment variables or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback: use pydantic BaseModel with manual env loading
    from pydantic import BaseModel as BaseSettings


class ServerConfig(BaseSettings):
    """Core server settings."""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    device_api_key: str = Field(
        default="runanywhere-dev-key",
        description="Shared secret for device auth (local dev)",
    )

    model_config = {"env_prefix": "SERVER_", "env_file": ".env", "extra": "ignore"}


class StorageConfig(BaseSettings):
    """Data storage paths."""

    data_dir: Path = Field(default=Path("./data"), description="Root data directory")

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def training_data_dir(self) -> Path:
        return self.data_dir / "training_data"

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "finetune.db"

    def ensure_dirs(self) -> None:
        """Create all storage directories."""
        for d in [
            self.training_data_dir,
            self.adapters_dir,
            self.checkpoints_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


class TrainingConfig(BaseSettings):
    """Fine-tuning hyperparameters."""

    base_model: str = Field(
        default="unsloth/Llama-3.2-1B",
        description="HuggingFace model ID for fine-tuning",
    )
    lora_rank: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    training_epochs: int = Field(default=3, description="Training epochs")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation")
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA",
    )
    auto_convert_gguf: bool = Field(
        default=True,
        description="Auto-convert adapters to GGUF after training",
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


class Settings:
    """Aggregate settings container."""

    def __init__(self) -> None:
        self.server = ServerConfig()
        self.storage = StorageConfig()
        self.training = TrainingConfig()
        self.storage.ensure_dirs()

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  server={self.server.host}:{self.server.port}\n"
            f"  data_dir={self.storage.data_dir}\n"
            f"  base_model={self.training.base_model}\n"
            f"  lora_rank={self.training.lora_rank}\n"
            f")"
        )


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
