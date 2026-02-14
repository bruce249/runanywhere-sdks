package com.runanywhere.sdk.features.ota

import kotlinx.serialization.Serializable

/**
 * Domain models for the OTA fine-tuning feature.
 *
 * These mirror the cloud server's API schemas and the existing
 * FineTuningModels.kt from the example app.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Training Data Models
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A conversation message for context.
 */
@Serializable
data class OtaContextMessage(
    val role: String,
    val content: String,
)

/**
 * A user interaction captured for training data upload.
 */
@Serializable
data class OtaTrainingInteraction(
    val id: String,
    val userPrompt: String,
    val modelResponse: String,
    val correctedResponse: String? = null,
    val rating: Int? = null,
    val isPositiveExample: Boolean = false,
    val includeInTraining: Boolean = true,
    val modelId: String? = null,
    val modelName: String? = null,
    val conversationContext: List<OtaContextMessage>? = null,
    val category: String? = null,
    val tags: List<String> = emptyList(),
    val timestamp: Long = System.currentTimeMillis(),
    val frequencyCount: Int = 1,
)

/**
 * Request to upload training data to the cloud server.
 */
@Serializable
data class TrainingDataUploadRequest(
    val deviceId: String,
    val modelId: String,
    val modelName: String? = null,
    val interactions: List<OtaTrainingInteraction>,
)

/**
 * Response from a successful data upload.
 */
@Serializable
data class TrainingDataUploadResponse(
    val uploadId: String,
    val deviceId: String,
    val sampleCount: Int,
    val totalSamplesForDevice: Int,
    val message: String,
)

// ═══════════════════════════════════════════════════════════════════════════
// Adapter Models
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format of a LoRA adapter file.
 */
@Serializable
enum class AdapterFormat {
    GGUF,
    SAFETENSORS,
    BIN,
}

/**
 * Metadata for a trained LoRA adapter available for download.
 */
@Serializable
data class OtaAdapterInfo(
    val id: String,
    val name: String,
    val description: String? = null,
    val deviceId: String,
    val modelId: String,
    val baseModel: String,
    val format: String = "gguf",
    val fileSizeBytes: Long = 0,
    val loraRank: Int = 8,
    val loraAlpha: Int = 16,
    val targetModules: List<String> = listOf("q_proj", "v_proj", "k_proj", "o_proj"),
    val trainingSamples: Int = 0,
    val finalLoss: Float? = null,
    val version: Int = 1,
    val checksum: String? = null,
    val isPublished: Boolean = true,
    val createdAt: String? = null,
) {
    /** Local file path — set after download */
    @kotlinx.serialization.Transient
    var localFilePath: String? = null

    /** Whether this adapter is currently active for inference */
    @kotlinx.serialization.Transient
    var isActive: Boolean = false
}

/**
 * Response from polling for new adapters.
 */
@Serializable
data class AdapterPollResponse(
    val hasNewAdapters: Boolean,
    val adapters: List<OtaAdapterInfo> = emptyList(),
    val pollIntervalSeconds: Int = 60,
)

// ═══════════════════════════════════════════════════════════════════════════
// Training Job Models
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Request to start a fine-tuning job on the cloud server.
 */
@Serializable
data class StartTrainingRequest(
    val deviceId: String,
    val modelId: String,
    val baseModel: String? = null,
    val uploadIds: List<String>? = null,
    val adapterName: String? = null,
)

/**
 * Status of a cloud-side training job.
 */
@Serializable
enum class TrainingJobStatus {
    PENDING,
    PREPROCESSING,
    TRAINING,
    CONVERTING,
    COMPLETED,
    FAILED,
    CANCELLED,
}

/**
 * Training job status response from the cloud.
 */
@Serializable
data class TrainingJobResponse(
    val jobId: String,
    val deviceId: String,
    val modelId: String,
    val baseModel: String,
    val status: String,
    val progress: Float = 0f,
    val currentEpoch: Int = 0,
    val totalEpochs: Int = 3,
    val currentLoss: Float? = null,
    val bestLoss: Float? = null,
    val errorMessage: String? = null,
    val adapterId: String? = null,
    val startedAt: String? = null,
    val completedAt: String? = null,
    val createdAt: String? = null,
)

// ═══════════════════════════════════════════════════════════════════════════
// OTA Events (for the SDK event bus)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Events emitted by the OTA fine-tuning system.
 */
sealed class OtaFineTuningEvent {
    /** Training data uploaded successfully */
    data class DataUploaded(
        val uploadId: String,
        val sampleCount: Int,
        val totalSamples: Int,
    ) : OtaFineTuningEvent()

    /** New adapter available for download */
    data class AdapterAvailable(
        val adapter: OtaAdapterInfo,
    ) : OtaFineTuningEvent()

    /** Adapter download progress */
    data class DownloadProgress(
        val adapterId: String,
        val progress: Float,
    ) : OtaFineTuningEvent()

    /** Adapter downloaded and ready */
    data class AdapterDownloaded(
        val adapter: OtaAdapterInfo,
        val localPath: String,
    ) : OtaFineTuningEvent()

    /** Adapter applied to the model */
    data class AdapterApplied(
        val adapter: OtaAdapterInfo,
    ) : OtaFineTuningEvent()

    /** An error occurred */
    data class Error(
        val operation: String,
        val message: String,
        val cause: Throwable? = null,
    ) : OtaFineTuningEvent()
}
