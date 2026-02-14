"""
LoRA Fine-Tuning Engine — wraps PEFT/Unsloth for cloud-side training.

Trains LoRA adapters from preprocessed device data,
then optionally converts to GGUF for llama.cpp consumption on-device.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─── GGUF / Local Model Name → HuggingFace Repo ID Resolver ────────────────

# Maps common GGUF quantization filenames and short names to their
# canonical HuggingFace model repository IDs so that the training engine
# can download the full-precision model for LoRA fine-tuning.
_GGUF_TO_HF_MODEL: dict[str, str] = {
    # Qwen 2.5 family
    "qwen2.5-0.5b-instruct-q6_k": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct-q4_k_m": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct-q8_0": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct-q4_k_m": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-instruct-q6_k": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-instruct-q4_k_m": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct-q4_k_m": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # Qwen 3 family
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    # LFM family (Liquid Foundation Models)
    "lfm2-350m-q4_k_m": "LiquidAI/LFM2-350M",
    "lfm2-350m-q6_k": "LiquidAI/LFM2-350M",
    "lfm2-350m": "LiquidAI/LFM2-350M",
    "lfm2-0.5b-q4_k_m": "LiquidAI/LFM2-0.5B",
    "lfm2-0.5b": "LiquidAI/LFM2-0.5B",
    "lfm2-1.2b-q4_k_m": "LiquidAI/LFM2-1.2B",
    "lfm2-1.2b": "LiquidAI/LFM2-1.2B",
    # Llama family
    "llama-3.2-1b": "unsloth/Llama-3.2-1B",
    "llama-3.2-1b-instruct": "unsloth/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # Gemma family
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    # Phi family
    "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    # SmolLM family
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm2-360m-instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # TinyLlama
    "tinyllama-1.1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def resolve_model_name(name: str) -> str:
    """Resolve a GGUF filename or short name to a HuggingFace model repo ID.

    If the name already looks like a valid HF repo (contains '/'), it is
    returned as-is.  Otherwise we strip common GGUF suffixes and look it up
    in our mapping table.
    """
    if not name:
        return name

    # Already a valid HF repo ID (e.g. "unsloth/Llama-3.2-1B")
    if "/" in name:
        return name

    key = name.lower().strip()

    # Direct match
    if key in _GGUF_TO_HF_MODEL:
        resolved = _GGUF_TO_HF_MODEL[key]
        logger.info(f"Model name resolved: '{name}' → '{resolved}'")
        return resolved

    # Try stripping .gguf extension
    if key.endswith(".gguf"):
        key = key[:-5]
        if key in _GGUF_TO_HF_MODEL:
            resolved = _GGUF_TO_HF_MODEL[key]
            logger.info(f"Model name resolved: '{name}' → '{resolved}'")
            return resolved

    # Try stripping trailing quant suffix (e.g. "-q4_k_m", "-q6_k", "-q8_0", "-f16")
    import re
    stripped = re.sub(r'-(?:q[0-9]+_[a-z0-9]+|q[0-9]+|f16|f32)$', '', key)
    if stripped != key and stripped in _GGUF_TO_HF_MODEL:
        resolved = _GGUF_TO_HF_MODEL[stripped]
        logger.info(f"Model name resolved: '{name}' → '{resolved}' (after stripping quant suffix)")
        return resolved

    logger.warning(
        f"Could not resolve model name '{name}' to a HuggingFace repo ID. "
        f"Will attempt to use it directly. If this fails, add a mapping to "
        f"_GGUF_TO_HF_MODEL in training_engine.py"
    )
    return name


@dataclass
class TrainingProgress:
    """Live training progress."""

    status: str = "pending"
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float | None = None
    best_loss: float | None = None
    learning_rate: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None


@dataclass
class TrainingResult:
    """Result of a completed training run."""

    success: bool
    adapter_dir: Path | None = None
    gguf_path: Path | None = None
    final_loss: float | None = None
    best_loss: float | None = None
    total_steps: int = 0
    training_time_seconds: float = 0.0
    sample_count: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FineTuningEngine:
    """
    Cloud-side LoRA fine-tuning engine.

    Supports two backends:
    1. PEFT + Transformers (default, works on any GPU/CPU)
    2. Unsloth (optional, 2x faster on supported GPUs)
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Path,
        checkpoints_dir: Path,
        use_unsloth: bool = False,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.use_unsloth = use_unsloth
        self._progress = TrainingProgress()
        self._progress_callback: Callable[[TrainingProgress], None] | None = None

    @property
    def progress(self) -> TrainingProgress:
        return self._progress

    def set_progress_callback(self, callback: Callable[[TrainingProgress], None]) -> None:
        self._progress_callback = callback

    def _update_progress(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self._progress, k):
                setattr(self._progress, k, v)
        if self._progress_callback:
            self._progress_callback(self._progress)

    # ─── Main Training Entry Point ───────────────────────────────────────

    def train(
        self,
        dataset_path: Path,
        job_id: str,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        epochs: int = 3,
        batch_size: int = 4,
        max_seq_length: int = 512,
        gradient_accumulation_steps: int = 4,
        target_modules: list[str] | None = None,
        use_4bit: bool = True,
        auto_convert_gguf: bool = True,
    ) -> TrainingResult:
        """
        Run LoRA fine-tuning on the prepared dataset.

        Returns TrainingResult with adapter paths and metrics.
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        adapter_output_dir = self.output_dir / job_id
        adapter_output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        self._update_progress(status="training", progress=0.0, total_epochs=epochs)

        try:
            if self.use_unsloth:
                result = self._train_with_unsloth(
                    dataset_path=dataset_path,
                    output_dir=adapter_output_dir,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    target_modules=target_modules,
                    use_4bit=use_4bit,
                )
            else:
                result = self._train_with_peft(
                    dataset_path=dataset_path,
                    output_dir=adapter_output_dir,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    target_modules=target_modules,
                    use_4bit=use_4bit,
                )

            # Convert to GGUF if requested
            gguf_path = None
            if auto_convert_gguf and result.success:
                self._update_progress(status="converting", progress=0.9)
                gguf_path = self._convert_to_gguf(adapter_output_dir, job_id)

            elapsed = time.time() - start_time
            result.training_time_seconds = elapsed
            result.gguf_path = gguf_path

            self._update_progress(status="completed", progress=1.0)
            logger.info(
                f"Training completed in {elapsed:.1f}s — "
                f"loss={result.final_loss:.4f}, adapter={adapter_output_dir}"
            )
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Training failed: {error_msg}")
            self._update_progress(status="failed", error=error_msg)
            return TrainingResult(
                success=False,
                error=error_msg,
                training_time_seconds=elapsed,
            )

    # ─── PEFT Backend ────────────────────────────────────────────────────

    def _train_with_peft(
        self,
        dataset_path: Path,
        output_dir: Path,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        max_seq_length: int,
        gradient_accumulation_steps: int,
        target_modules: list[str],
        use_4bit: bool,
    ) -> TrainingResult:
        """Fine-tune using PEFT + Transformers."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        # Resolve GGUF filename to HuggingFace repo ID
        hf_model_id = resolve_model_name(self.base_model)
        logger.info(f"Loading base model: {hf_model_id} (requested: {self.base_model})")
        self._update_progress(status="preprocessing", progress=0.05)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization config
        bnb_config = None
        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # Load dataset
        self._update_progress(status="preprocessing", progress=0.1)
        raw_data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        dataset = Dataset.from_list(raw_data)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        # Training progress callback
        engine = self

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    progress = min(state.global_step / max(state.max_steps, 1), 0.89)
                    engine._update_progress(
                        progress=0.1 + progress * 0.8,
                        current_step=state.global_step,
                        total_steps=state.max_steps,
                        current_loss=logs.get("loss"),
                        learning_rate=logs.get("learning_rate"),
                    )

            def on_epoch_end(self, args, state, control, **kwargs):
                engine._update_progress(current_epoch=int(state.epoch))

        # Training arguments
        checkpoint_dir = self.checkpoints_dir / f"job_{output_dir.name}"
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            report_to="none",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            dataloader_pin_memory=False,
        )

        # Train
        self._update_progress(status="training", progress=0.1, current_epoch=0)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[ProgressCallback()],
        )

        train_result = trainer.train()

        # Save adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Extract metrics
        final_loss = train_result.metrics.get("train_loss", None)

        return TrainingResult(
            success=True,
            adapter_dir=output_dir,
            final_loss=final_loss,
            best_loss=final_loss,
            total_steps=train_result.global_step,
            sample_count=len(raw_data),
            metadata={
                "trainable_params": trainable_params,
                "total_params": total_params,
                "base_model": self.base_model,
            },
        )

    # ─── Unsloth Backend (Optional, Faster) ──────────────────────────────

    def _train_with_unsloth(
        self,
        dataset_path: Path,
        output_dir: Path,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        max_seq_length: int,
        gradient_accumulation_steps: int,
        target_modules: list[str],
        use_4bit: bool,
    ) -> TrainingResult:
        """Fine-tune using Unsloth (2x faster training)."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.warning("Unsloth not installed, falling back to PEFT")
            return self._train_with_peft(
                dataset_path=dataset_path,
                output_dir=output_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                target_modules=target_modules,
                use_4bit=use_4bit,
            )

        import json

        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, TrainingArguments
        from trl import SFTTrainer

        self._update_progress(status="preprocessing", progress=0.05)

        # Load model with Unsloth (patched for 2x speed)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
        )

        # Apply LoRA via Unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Load dataset
        self._update_progress(status="preprocessing", progress=0.1)
        raw_data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

        dataset = Dataset.from_list(raw_data)

        # Training
        training_args = TrainingArguments(
            output_dir=str(self.checkpoints_dir / f"job_{output_dir.name}"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            optim="adamw_8bit",
        )

        self._update_progress(status="training", progress=0.1, current_epoch=0)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
        )

        train_result = trainer.train()

        # Save
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        final_loss = train_result.metrics.get("train_loss", None)

        return TrainingResult(
            success=True,
            adapter_dir=output_dir,
            final_loss=final_loss,
            best_loss=final_loss,
            total_steps=train_result.global_step,
            sample_count=len(raw_data),
            metadata={"base_model": self.base_model, "backend": "unsloth"},
        )

    # ─── GGUF Conversion ─────────────────────────────────────────────────

    def _convert_to_gguf(self, adapter_dir: Path, job_id: str) -> Path | None:
        """
        Convert LoRA adapter to GGUF format for llama.cpp.

        Uses llama-cpp-python's conversion utilities or the llama.cpp
        convert script if available.
        """
        gguf_path = adapter_dir / f"adapter-{job_id}.gguf"

        try:
            # Try using the llama.cpp convert-lora-to-ggml script
            # This requires llama.cpp to be installed or available
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "llama_cpp.convert",
                    "--outfile",
                    str(gguf_path),
                    "--outtype",
                    "f16",
                    str(adapter_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info(f"Converted adapter to GGUF: {gguf_path}")
                return gguf_path
            else:
                logger.warning(f"GGUF conversion via llama_cpp failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"GGUF conversion tool not available: {e}")

        # Fallback: save the safetensors adapter as-is and note that
        # the device should use safetensors loading path
        safetensors_path = adapter_dir / "adapter_model.safetensors"
        if safetensors_path.exists():
            logger.info(
                f"GGUF conversion not available. Adapter available as safetensors: "
                f"{safetensors_path}"
            )
            return safetensors_path

        logger.warning("No adapter file found after training")
        return None


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
