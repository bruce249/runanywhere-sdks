package com.runanywhere.runanywhereai.domain.models

import kotlinx.serialization.Serializable
import java.util.UUID

// =============================================================================
// On-Device Fine-Tuning Domain Models
// =============================================================================

/**
 * Represents a single user interaction captured for training data.
 * Each interaction records the user's input, the model's response,
 * and optional user feedback/corrections.
 */
@Serializable
data class TrainingInteraction(
    val id: String = UUID.randomUUID().toString(),
    /** The user's original prompt/query */
    val userPrompt: String,
    /** The model's generated response */
    val modelResponse: String,
    /** User-corrected/preferred response (if user provided feedback) */
    val correctedResponse: String? = null,
    /** User rating: 1-5 stars, null if not rated */
    val rating: Int? = null,
    /** Whether the user explicitly marked this as a good example */
    val isPositiveExample: Boolean = false,
    /** Whether to include this sample in training */
    val includeInTraining: Boolean = true,
    /** The model that generated the response */
    val modelId: String? = null,
    val modelName: String? = null,
    /** Conversation context (previous messages for multi-turn) */
    val conversationContext: List<ContextMessage>? = null,
    /** Interaction category/topic (auto-detected or user-tagged) */
    val category: String? = null,
    /** Custom tags for organizing training data */
    val tags: List<String> = emptyList(),
    /** Timestamp of the interaction */
    val timestamp: Long = System.currentTimeMillis(),
    /** Number of times similar queries were asked */
    val frequencyCount: Int = 1,
    /** Data source: "text_chat", "voice_chat", "video_analysis" */
    val dataSource: String = "text_chat",
)

/**
 * Simplified context message for training data.
 */
@Serializable
data class ContextMessage(
    val role: String, // "user", "assistant", "system"
    val content: String,
)

/**
 * A prepared training sample ready for the fine-tuning engine.
 * Follows the instruction-input-output format compatible with LoRA training.
 */
@Serializable
data class TrainingSample(
    val id: String = UUID.randomUUID().toString(),
    /** System instruction/persona */
    val instruction: String = "You are a helpful assistant.",
    /** User input/prompt */
    val input: String,
    /** Expected/desired output */
    val output: String,
    /** Weight/importance of this sample (higher = more emphasis during training) */
    val weight: Float = 1.0f,
    /** Source interaction ID */
    val sourceInteractionId: String? = null,
)

/**
 * Configuration for a LoRA adapter training session.
 */
@Serializable
data class FineTuningConfig(
    /** LoRA rank (lower = fewer params, faster; higher = more expressive) */
    val loraRank: Int = 8,
    /** LoRA alpha (scaling factor, typically 2x rank) */
    val loraAlpha: Int = 16,
    /** LoRA dropout rate */
    val loraDropout: Float = 0.05f,
    /** Learning rate */
    val learningRate: Float = 2e-4f,
    /** Number of training epochs */
    val epochs: Int = 3,
    /** Batch size (keep small for mobile) */
    val batchSize: Int = 1,
    /** Maximum sequence length */
    val maxSeqLength: Int = 512,
    /** Gradient accumulation steps (simulates larger batch) */
    val gradientAccumulationSteps: Int = 4,
    /** Weight decay for regularization */
    val weightDecay: Float = 0.01f,
    /** Warmup ratio for learning rate scheduler */
    val warmupRatio: Float = 0.1f,
    /** Use 4-bit quantization (QLoRA) */
    val use4BitQuantization: Boolean = true,
    /** Target modules to apply LoRA (e.g., ["q_proj", "v_proj"]) */
    val targetModules: List<String> = listOf("q_proj", "v_proj", "k_proj", "o_proj"),
    /** Minimum training samples required before training can start */
    val minTrainingSamples: Int = 10,
    /** Save checkpoint every N steps */
    val saveCheckpointEverySteps: Int = 50,
    /** Maximum memory usage in MB for training */
    val maxMemoryMB: Int = 1024,
    /** Enable gradient checkpointing to save memory */
    val gradientCheckpointing: Boolean = true,
)

/**
 * Represents a trained LoRA adapter that can be applied to a model.
 */
@Serializable
data class LoraAdapter(
    val id: String = UUID.randomUUID().toString(),
    /** Human-readable name for this adapter */
    val name: String,
    /** Description of what this adapter specializes in */
    val description: String = "",
    /** The base model this adapter was trained on */
    val baseModelId: String,
    val baseModelName: String,
    /** Training configuration used */
    val config: FineTuningConfig,
    /** Number of training samples used */
    val trainingSampleCount: Int,
    /** Training metrics */
    val trainingMetrics: TrainingMetrics? = null,
    /** File path to the adapter weights */
    val adapterFilePath: String? = null,
    /** Size of adapter file in bytes */
    val adapterSizeBytes: Long = 0,
    /** Whether this adapter is currently active/loaded */
    val isActive: Boolean = false,
    /** Creation timestamp */
    val createdAt: Long = System.currentTimeMillis(),
    /** Last used timestamp */
    val lastUsedAt: Long? = null,
    /** Version number (increments with re-training) */
    val version: Int = 1,
    /** Tags/categories this adapter is good at */
    val specializations: List<String> = emptyList(),
)

/**
 * Metrics from a training session.
 */
@Serializable
data class TrainingMetrics(
    /** Final training loss */
    val finalLoss: Float = 0f,
    /** Initial training loss */
    val initialLoss: Float = 0f,
    /** Loss history per step */
    val lossHistory: List<Float> = emptyList(),
    /** Training duration in milliseconds */
    val trainingDurationMs: Long = 0,
    /** Total training steps completed */
    val totalSteps: Int = 0,
    /** Tokens processed per second */
    val tokensPerSecond: Float = 0f,
    /** Peak memory usage in MB */
    val peakMemoryMB: Int = 0,
    /** Number of epochs completed */
    val epochsCompleted: Int = 0,
    /** Best validation loss (if validation data available) */
    val bestValidationLoss: Float? = null,
    /** Average improvement per epoch */
    val improvementPerEpoch: Float = 0f,
)

/**
 * Current state of a fine-tuning job.
 */
@Serializable
sealed class FineTuningStatus {
    @Serializable
    data object Idle : FineTuningStatus()

    @Serializable
    data class CollectingData(val samplesCollected: Int, val minRequired: Int) : FineTuningStatus()

    @Serializable
    data class PreparingData(val progress: Float) : FineTuningStatus()

    @Serializable
    data class Training(
        val currentEpoch: Int,
        val totalEpochs: Int,
        val currentStep: Int,
        val totalSteps: Int,
        val currentLoss: Float,
        val estimatedTimeRemainingMs: Long,
        val tokensPerSecond: Float,
        val memoryUsageMB: Int,
    ) : FineTuningStatus()

    @Serializable
    data class SavingAdapter(val progress: Float) : FineTuningStatus()

    @Serializable
    data class Completed(
        val adapter: LoraAdapter,
        val metrics: TrainingMetrics,
    ) : FineTuningStatus()

    @Serializable
    data class Failed(val error: String, val partialMetrics: TrainingMetrics? = null) : FineTuningStatus()

    @Serializable
    data object Cancelled : FineTuningStatus()
}

/**
 * Summary statistics for the training data collection.
 */
@Serializable
data class TrainingDataStats(
    val totalInteractions: Int = 0,
    val ratedInteractions: Int = 0,
    val correctedInteractions: Int = 0,
    val positiveExamples: Int = 0,
    val readyForTraining: Int = 0,
    val categories: Map<String, Int> = emptyMap(),
    val averageRating: Float = 0f,
    val oldestInteraction: Long? = null,
    val newestInteraction: Long? = null,
    val topQueries: List<String> = emptyList(),
    /** Count of interactions per data source (text_chat, voice_chat, video_analysis) */
    val dataSourceCounts: Map<String, Int> = emptyMap(),
)

/**
 * Event types emitted during the fine-tuning process.
 */
sealed class FineTuningEvent {
    data class DataCollected(val interaction: TrainingInteraction) : FineTuningEvent()
    data class TrainingStarted(val config: FineTuningConfig, val sampleCount: Int) : FineTuningEvent()
    data class TrainingProgress(val status: FineTuningStatus.Training) : FineTuningEvent()
    data class TrainingCompleted(val adapter: LoraAdapter) : FineTuningEvent()
    data class TrainingFailed(val error: String) : FineTuningEvent()
    data class AdapterLoaded(val adapter: LoraAdapter) : FineTuningEvent()
    data class AdapterUnloaded(val adapterId: String) : FineTuningEvent()
}
