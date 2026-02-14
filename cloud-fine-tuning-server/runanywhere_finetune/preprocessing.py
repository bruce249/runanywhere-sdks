"""
Data preprocessing — convert raw device interactions into training-ready datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from runanywhere_finetune.schemas import TrainingInteraction

logger = logging.getLogger(__name__)


# ─── Chat Template Formatters ────────────────────────────────────────────────


def format_chatml(interaction: TrainingInteraction) -> str:
    """Format interaction as ChatML (Llama 3, Mistral, etc.)."""
    parts = []

    # System message
    parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>")

    # Add conversation context if available
    if interaction.conversation_context:
        for msg in interaction.conversation_context:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    # User prompt
    parts.append(f"<|im_start|>user\n{interaction.user_prompt}<|im_end|>")

    # Target response (corrected if available, otherwise original)
    target = interaction.corrected_response or interaction.model_response
    parts.append(f"<|im_start|>assistant\n{target}<|im_end|>")

    return "\n".join(parts)


def format_llama3(interaction: TrainingInteraction) -> str:
    """Format interaction as Llama 3 chat template."""
    parts = []

    parts.append(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
    )

    if interaction.conversation_context:
        for msg in interaction.conversation_context:
            parts.append(
                f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n"
                f"{msg.content}<|eot_id|>"
            )

    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{interaction.user_prompt}<|eot_id|>"
    )

    target = interaction.corrected_response or interaction.model_response
    parts.append(
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{target}<|eot_id|>"
    )

    return "".join(parts)


def format_alpaca(interaction: TrainingInteraction) -> dict[str, str]:
    """Format as Alpaca instruction-input-output."""
    target = interaction.corrected_response or interaction.model_response
    return {
        "instruction": "You are a helpful assistant.",
        "input": interaction.user_prompt,
        "output": target,
    }


# ─── Template Registry ───────────────────────────────────────────────────────

FORMATTERS = {
    "chatml": format_chatml,
    "llama3": format_llama3,
    "alpaca": format_alpaca,
}


# ─── Preprocessor ────────────────────────────────────────────────────────────


class DataPreprocessor:
    """Converts raw device interactions into training-ready format."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_interactions_from_upload(self, upload_path: str) -> list[TrainingInteraction]:
        """Load interactions from a stored upload file."""
        path = Path(upload_path)
        if not path.exists():
            logger.error(f"Upload file not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = []
        for item in data.get("interactions", []):
            try:
                interactions.append(TrainingInteraction(**item))
            except Exception as e:
                logger.warning(f"Skipping invalid interaction: {e}")

        return interactions

    def filter_interactions(
        self,
        interactions: list[TrainingInteraction],
        min_rating: int | None = None,
        require_correction: bool = False,
        categories: list[str] | None = None,
    ) -> list[TrainingInteraction]:
        """Filter interactions based on quality criteria."""
        filtered = []
        for ix in interactions:
            if not ix.include_in_training:
                continue
            if min_rating is not None and (ix.rating is None or ix.rating < min_rating):
                continue
            if require_correction and ix.corrected_response is None:
                continue
            if categories and ix.category not in categories:
                continue
            filtered.append(ix)

        logger.info(
            f"Filtered {len(interactions)} → {len(filtered)} interactions "
            f"(min_rating={min_rating}, require_correction={require_correction})"
        )
        return filtered

    def apply_sample_weights(
        self, interactions: list[TrainingInteraction]
    ) -> list[TrainingInteraction]:
        """Duplicate or weight samples based on quality signals."""
        weighted = []
        for ix in interactions:
            repeat = 1

            # Corrected responses are high quality — repeat them
            if ix.corrected_response:
                repeat += 2

            # Highly rated interactions
            if ix.rating and ix.rating >= 4:
                repeat += 1

            # Positive examples explicitly marked
            if ix.is_positive_example:
                repeat += 1

            # Frequent queries — user asks this a lot, model should learn it
            if ix.frequency_count and ix.frequency_count > 3:
                repeat += 1

            for _ in range(min(repeat, 5)):  # Cap at 5x
                weighted.append(ix)

        logger.info(f"Weighted samples: {len(interactions)} → {len(weighted)}")
        return weighted

    def prepare_dataset(
        self,
        upload_paths: list[str],
        template: str = "chatml",
        min_rating: int | None = None,
    ) -> list[dict]:
        """
        Full pipeline: load → filter → weight → format → return as dataset rows.
        Returns list of dicts with 'text' key (for HuggingFace datasets).
        """
        all_interactions: list[TrainingInteraction] = []
        for path in upload_paths:
            all_interactions.extend(self.load_interactions_from_upload(path))

        if not all_interactions:
            logger.warning("No interactions loaded from uploads")
            return []

        # Filter and weight
        filtered = self.filter_interactions(all_interactions, min_rating=min_rating)
        weighted = self.apply_sample_weights(filtered)

        # Format
        formatter = FORMATTERS.get(template)
        if formatter is None:
            logger.warning(f"Unknown template '{template}', falling back to chatml")
            formatter = FORMATTERS["chatml"]

        dataset_rows = []
        for ix in weighted:
            formatted = formatter(ix)
            if isinstance(formatted, str):
                dataset_rows.append({"text": formatted})
            elif isinstance(formatted, dict):
                dataset_rows.append(formatted)

        logger.info(f"Prepared {len(dataset_rows)} training samples with template '{template}'")
        return dataset_rows

    def save_prepared_dataset(
        self,
        dataset_rows: list[dict],
        output_path: Path,
    ) -> Path:
        """Save prepared dataset as JSONL for training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(dataset_rows)} samples to {output_path}")
        return output_path
