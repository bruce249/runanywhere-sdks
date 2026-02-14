package com.runanywhere.runanywhereai.data.cloud

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

// =============================================================================
// Cloud Fine-Tuning — Data Transfer Objects & State Models
// =============================================================================

// ─── Server Configuration ───────────────────────────────────────────────────

/**
 * Configuration for the cloud fine-tuning server connection.
 * Persisted as JSON in the app's files directory.
 */
@Serializable
data class CloudServerConfig(
    /** Full URL of the cloud server (e.g., "http://192.168.1.100:8000") */
    val serverUrl: String = "http://192.168.1.100:8000",
    /** API key (optional) */
    val apiKey: String = "",
    /** Unique device identifier for this phone */
    val deviceId: String = "",
    /** Auto-sync OTA adapters in background */
    val autoSync: Boolean = false,
)

// ─── Upload Request / Response ──────────────────────────────────────────────

@Serializable
data class CloudUploadRequest(
    @SerialName("device_id") val deviceId: String,
    @SerialName("base_model") val baseModel: String = "",
    val interactions: List<CloudInteraction>,
)

@Serializable
data class CloudInteraction(
    @SerialName("user_prompt") val userPrompt: String,
    @SerialName("model_response") val modelResponse: String,
    @SerialName("corrected_response") val correctedResponse: String? = null,
    val rating: Int? = null,
    @SerialName("is_positive_example") val isPositive: Boolean = false,
    val category: String? = null,
    @SerialName("data_source") val dataSource: String = "text_chat",
    @SerialName("conversation_context") val conversationContext: List<CloudContextMessage>? = null,
    // Match the server TrainingInteraction fields
    val id: String = "",
    @SerialName("include_in_training") val includeInTraining: Boolean = true,
    val tags: List<String> = emptyList(),
    val timestamp: Long = System.currentTimeMillis(),
    @SerialName("frequency_count") val frequencyCount: Int = 1,
)

@Serializable
data class CloudContextMessage(
    val role: String,
    val content: String,
)

@Serializable
data class CloudUploadResponse(
    val status: String = "",
    @SerialName("upload_id") val uploadId: String = "",
    @SerialName("samples_received") val samplesReceived: Int = 0,
    @SerialName("sample_count") val sampleCount: Int = 0,
    val message: String = "",
)

// ─── Train Request / Response ───────────────────────────────────────────────

@Serializable
data class CloudTrainRequest(
    @SerialName("device_id") val deviceId: String,
    @SerialName("base_model") val baseModel: String = "unsloth/Llama-3.2-1B",
    @SerialName("adapter_name") val adapterName: String = "device-adapter",
    @SerialName("lora_rank") val loraRank: Int = 8,
    @SerialName("lora_alpha") val loraAlpha: Int = 16,
    @SerialName("learning_rate") val learningRate: Float = 2e-4f,
    val epochs: Int = 3,
    @SerialName("output_format") val outputFormat: String = "gguf",
)

@Serializable
data class CloudTrainResponse(
    @SerialName("job_id") val jobId: String = "",
    val status: String = "",
    val message: String = "",
)

// ─── Job Status ─────────────────────────────────────────────────────────────

@Serializable
data class CloudJobStatus(
    @SerialName("job_id") val jobId: String = "",
    val status: String = "", // "queued", "processing", "completed", "failed"
    val progress: Float = 0f,
    @SerialName("current_epoch") val currentEpoch: Int? = null,
    @SerialName("total_epochs") val totalEpochs: Int? = null,
    @SerialName("current_loss") val currentLoss: Float? = null,
    @SerialName("error_message") val errorMessage: String? = null,
    @SerialName("adapter_id") val adapterId: String? = null,
)

// ─── Adapter Poll / List ────────────────────────────────────────────────────

@Serializable
data class CloudAdapterPollResponse(
    @SerialName("has_new_adapters") val hasNewAdapter: Boolean = false,
    val adapters: List<CloudAdapterInfo> = emptyList(),
)

@Serializable
data class CloudAdapterListResponse(
    val adapters: List<CloudAdapterInfo> = emptyList(),
    val total: Int = 0,
)

@Serializable
data class CloudAdapterInfo(
    val id: String = "",
    val name: String = "",
    @SerialName("base_model") val baseModel: String = "",
    @SerialName("training_samples") val trainingSamples: Int = 0,
    @SerialName("file_size_bytes") val sizeBytes: Long = 0,
    @SerialName("final_loss") val finalLoss: Float? = null,
    val format: String = "gguf",
    @SerialName("created_at") val createdAt: String? = null,
)

// ─── Sync Status (Sealed for UI) ────────────────────────────────────────────

sealed class CloudSyncStatus {
    data object Disconnected : CloudSyncStatus()
    data object Connecting : CloudSyncStatus()
    data object Connected : CloudSyncStatus()
    data class Uploading(val progress: Float, val message: String = "") : CloudSyncStatus()
    data class Training(
        val jobId: String,
        val progress: Float,
        val status: String,
    ) : CloudSyncStatus()
    data class DownloadingAdapter(val progress: Float) : CloudSyncStatus()
    data class Error(val message: String) : CloudSyncStatus()
    data class Completed(val message: String) : CloudSyncStatus()
}

// ─── Cloud UI State ─────────────────────────────────────────────────────────

/**
 * UI state for the Cloud Fine-Tuning tab.
 * Observed by FineTuneViewModel and rendered by CloudFineTuneTab.
 */
data class CloudUiState(
    val config: CloudServerConfig = CloudServerConfig(),
    val isServerReachable: Boolean = false,
    val syncStatus: CloudSyncStatus = CloudSyncStatus.Disconnected,

    // Multi-modal data counts
    val dataReadyForUpload: Int = 0,
    val textChatCount: Int = 0,
    val voiceChatCount: Int = 0,
    val videoAnalysisCount: Int = 0,

    // Training
    val lastUploadId: String? = null,
    val activeJobId: String? = null,
    val activeJobStatus: CloudJobStatus? = null,

    // OTA Adapters
    val availableAdapters: List<CloudAdapterInfo> = emptyList(),
    val downloadedAdapterIds: Set<String> = emptySet(),

    // Sync
    val lastSyncTime: Long? = null,
    val errorMessage: String? = null,
) {
    val isBusy: Boolean
        get() = syncStatus is CloudSyncStatus.Uploading ||
            syncStatus is CloudSyncStatus.Training ||
            syncStatus is CloudSyncStatus.DownloadingAdapter ||
            syncStatus is CloudSyncStatus.Connecting

    val hasServerUrl: Boolean
        get() = config.serverUrl.isNotBlank()
}
