package com.runanywhere.runanywhereai.presentation.finetune

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.runanywhere.runanywhereai.data.AdapterStore
import com.runanywhere.runanywhereai.data.TrainingDataStore
import com.runanywhere.runanywhereai.data.cloud.CloudFineTuningService
import com.runanywhere.runanywhereai.data.cloud.CloudSyncStatus
import com.runanywhere.runanywhereai.data.cloud.CloudUiState
import com.runanywhere.runanywhereai.data.cloud.OtaAdapterReceiver
import com.runanywhere.runanywhereai.domain.models.FineTuningConfig
import com.runanywhere.runanywhereai.domain.models.FineTuningStatus
import com.runanywhere.runanywhereai.domain.models.LoraAdapter
import com.runanywhere.runanywhereai.domain.models.TrainingDataStats
import com.runanywhere.runanywhereai.domain.models.TrainingInteraction
import com.runanywhere.runanywhereai.domain.services.FineTuningEngine
import com.runanywhere.sdk.public.RunAnywhere
import com.runanywhere.sdk.public.extensions.availableModels
import com.runanywhere.sdk.public.extensions.currentLLMModelId
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * UI state for the Fine-Tuning screen.
 */
data class FineTuneUiState(
    // Current tab
    val selectedTab: FineTuneTab = FineTuneTab.OVERVIEW,

    // Data collection stats
    val dataStats: TrainingDataStats = TrainingDataStats(),
    val recentInteractions: List<TrainingInteraction> = emptyList(),

    // Training configuration
    val config: FineTuningConfig = FineTuningConfig(),
    val adapterName: String = "",

    // Training status
    val trainingStatus: FineTuningStatus = FineTuningStatus.Idle,

    // Adapter management
    val adapters: List<LoraAdapter> = emptyList(),
    val activeAdapter: LoraAdapter? = null,

    // Model info
    val currentModelId: String? = null,
    val currentModelName: String? = null,

    // UI states
    val showConfigSheet: Boolean = false,
    val showDeleteConfirm: String? = null, // adapter ID to confirm delete
    val showInteractionDetail: TrainingInteraction? = null,
    val errorMessage: String? = null,

    // Device readiness
    val isDeviceReady: Boolean = true,
    val estimatedTrainingTimeMs: Long = 0,
) {
    val canStartTraining: Boolean
        get() = dataStats.readyForTraining >= config.minTrainingSamples &&
            !isTraining &&
            currentModelId != null &&
            isDeviceReady &&
            adapterName.isNotBlank()

    val isTraining: Boolean
        get() = trainingStatus is FineTuningStatus.Training ||
            trainingStatus is FineTuningStatus.PreparingData ||
            trainingStatus is FineTuningStatus.SavingAdapter

    val trainingProgress: Float
        get() = when (val status = trainingStatus) {
            is FineTuningStatus.PreparingData -> status.progress * 0.1f
            is FineTuningStatus.Training -> {
                val stepProgress = if (status.totalSteps > 0) {
                    status.currentStep.toFloat() / status.totalSteps
                } else 0f
                0.1f + stepProgress * 0.8f
            }
            is FineTuningStatus.SavingAdapter -> 0.9f + status.progress * 0.1f
            is FineTuningStatus.Completed -> 1f
            else -> 0f
        }
}

enum class FineTuneTab(val label: String) {
    OVERVIEW("Overview"),
    DATA("Data"),
    TRAIN("Train"),
    ADAPTERS("Adapters"),
    CLOUD("Cloud"),
}

/**
 * ViewModel for the Fine-Tuning screen.
 *
 * Coordinates between:
 * - TrainingDataStore (collected interaction data)
 * - FineTuningEngine (training execution)
 * - AdapterStore (adapter management)
 */
class FineTuneViewModel(application: Application) : AndroidViewModel(application) {
    private val trainingDataStore = TrainingDataStore.getInstance(application)
    private val fineTuningEngine = FineTuningEngine.getInstance(application)
    private val adapterStore = AdapterStore.getInstance(application)
    private val cloudService = CloudFineTuningService.getInstance(application)

    private val _uiState = MutableStateFlow(FineTuneUiState())
    val uiState: StateFlow<FineTuneUiState> = _uiState.asStateFlow()

    private val _cloudUiState = MutableStateFlow(CloudUiState())
    val cloudUiState: StateFlow<CloudUiState> = _cloudUiState.asStateFlow()

    init {
        // Observe training data stats
        viewModelScope.launch {
            trainingDataStore.stats.collect { stats ->
                _uiState.update { it.copy(dataStats = stats) }
                updateEstimatedTime()
            }
        }

        // Observe interactions
        viewModelScope.launch {
            trainingDataStore.interactions.collect { interactions ->
                _uiState.update { it.copy(recentInteractions = interactions.take(50)) }
            }
        }

        // Observe training status
        viewModelScope.launch {
            fineTuningEngine.status.collect { status ->
                _uiState.update { it.copy(trainingStatus = status) }

                // Auto-save completed adapter
                if (status is FineTuningStatus.Completed) {
                    adapterStore.saveAdapter(status.adapter)
                }
            }
        }

        // Observe adapters
        viewModelScope.launch {
            adapterStore.adapters.collect { adapters ->
                _uiState.update { it.copy(adapters = adapters) }
            }
        }

        // Observe active adapter
        viewModelScope.launch {
            adapterStore.activeAdapter.collect { active ->
                _uiState.update { it.copy(activeAdapter = active) }
            }
        }

        // Initialize model info
        refreshModelInfo()
        checkDeviceReadiness()

        // ── Cloud Observers ─────────────────────────────────────────────
        // Observe cloud config
        viewModelScope.launch {
            cloudService.config.collect { config ->
                _cloudUiState.update { it.copy(config = config) }
            }
        }

        // Observe cloud sync status
        viewModelScope.launch {
            cloudService.syncStatus.collect { status ->
                _cloudUiState.update { it.copy(syncStatus = status) }
            }
        }

        // Observe training data stats for cloud data summary
        viewModelScope.launch {
            trainingDataStore.stats.collect { stats ->
                _cloudUiState.update {
                    it.copy(
                        dataReadyForUpload = stats.readyForTraining,
                        textChatCount = stats.dataSourceCounts["text_chat"] ?: 0,
                        voiceChatCount = stats.dataSourceCounts["voice_chat"] ?: 0,
                        videoAnalysisCount = stats.dataSourceCounts["video_analysis"] ?: 0,
                    )
                }
            }
        }
    }

    // ========================================================================
    // Tab Navigation
    // ========================================================================

    fun selectTab(tab: FineTuneTab) {
        _uiState.update { it.copy(selectedTab = tab) }
    }

    // ========================================================================
    // Training Data Management
    // ========================================================================

    fun rateInteraction(interactionId: String, rating: Int) {
        trainingDataStore.addFeedback(interactionId, rating = rating)
    }

    fun correctInteraction(interactionId: String, correctedResponse: String) {
        trainingDataStore.addFeedback(interactionId, correctedResponse = correctedResponse)
    }

    fun markAsPositive(interactionId: String) {
        val interaction = _uiState.value.recentInteractions.find { it.id == interactionId }
        val newValue = !(interaction?.isPositiveExample ?: false)
        trainingDataStore.addFeedback(interactionId, isPositiveExample = newValue)
    }

    fun toggleTrainingInclusion(interactionId: String) {
        trainingDataStore.toggleTrainingInclusion(interactionId)
    }

    fun deleteInteraction(interactionId: String) {
        trainingDataStore.deleteInteraction(interactionId)
    }

    fun clearAllData() {
        trainingDataStore.clearAllData()
    }

    fun showInteractionDetail(interaction: TrainingInteraction?) {
        _uiState.update { it.copy(showInteractionDetail = interaction) }
    }

    // ========================================================================
    // Training Configuration
    // ========================================================================

    fun updateAdapterName(name: String) {
        _uiState.update { it.copy(adapterName = name) }
    }

    fun updateConfig(config: FineTuningConfig) {
        _uiState.update { it.copy(config = config) }
        updateEstimatedTime()
    }

    fun updateLoraRank(rank: Int) {
        updateConfig(_uiState.value.config.copy(loraRank = rank))
    }

    fun updateLearningRate(lr: Float) {
        updateConfig(_uiState.value.config.copy(learningRate = lr))
    }

    fun updateEpochs(epochs: Int) {
        updateConfig(_uiState.value.config.copy(epochs = epochs))
    }

    fun updateMaxSeqLength(length: Int) {
        updateConfig(_uiState.value.config.copy(maxSeqLength = length))
    }

    fun toggleQuantization() {
        val config = _uiState.value.config
        updateConfig(config.copy(use4BitQuantization = !config.use4BitQuantization))
    }

    fun toggleGradientCheckpointing() {
        val config = _uiState.value.config
        updateConfig(config.copy(gradientCheckpointing = !config.gradientCheckpointing))
    }

    fun showConfigSheet(show: Boolean) {
        _uiState.update { it.copy(showConfigSheet = show) }
    }

    // ========================================================================
    // Training Execution
    // ========================================================================

    fun startTraining() {
        val state = _uiState.value
        if (!state.canStartTraining) {
            Log.w(TAG, "Cannot start training: conditions not met")
            return
        }

        viewModelScope.launch {
            try {
                // Prepare training samples
                val samples = trainingDataStore.prepareTrainingSamples()
                if (samples.isEmpty()) {
                    _uiState.update { it.copy(errorMessage = "No training samples available") }
                    return@launch
                }

                // Default adapter name if empty
                val name = state.adapterName.ifBlank {
                    "Adapter ${state.adapters.size + 1}"
                }

                // Start training
                val adapter = fineTuningEngine.startTraining(
                    samples = samples,
                    config = state.config,
                    baseModelId = state.currentModelId ?: "unknown",
                    baseModelName = state.currentModelName ?: "Unknown Model",
                    adapterName = name,
                )

                if (adapter != null) {
                    Log.i(TAG, "Training completed successfully: ${adapter.name}")
                    // Auto-switch to adapters tab to show the result
                    _uiState.update { it.copy(selectedTab = FineTuneTab.ADAPTERS) }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Training failed: ${e.message}", e)
                _uiState.update { it.copy(errorMessage = "Training failed: ${e.message}") }
            }
        }
    }

    fun cancelTraining() {
        fineTuningEngine.cancelTraining()
    }

    fun prepareForNewTraining() {
        fineTuningEngine.resetToIdle()
        _uiState.update {
            it.copy(
                trainingStatus = FineTuningStatus.Idle,
                selectedTab = FineTuneTab.TRAIN,
                adapterName = "",
                errorMessage = null,
            )
        }
    }

    // ========================================================================
    // Adapter Management
    // ========================================================================

    fun activateAdapter(adapterId: String) {
        adapterStore.activateAdapter(adapterId)
    }

    fun deactivateAdapter() {
        adapterStore.deactivateAdapter()
    }

    fun confirmDeleteAdapter(adapterId: String?) {
        _uiState.update { it.copy(showDeleteConfirm = adapterId) }
    }

    fun deleteAdapter(adapterId: String) {
        adapterStore.deleteAdapter(adapterId)
        _uiState.update { it.copy(showDeleteConfirm = null) }
    }

    // ========================================================================
    // UI Helpers
    // ========================================================================

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    private fun refreshModelInfo() {
        viewModelScope.launch {
            try {
                val modelId = RunAnywhere.currentLLMModelId
                val models = RunAnywhere.availableModels()
                val currentModel = models.find { it.id == modelId }

                _uiState.update {
                    it.copy(
                        currentModelId = modelId,
                        currentModelName = currentModel?.name ?: modelId,
                    )
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not get model info: ${e.message}")
            }
        }
    }

    private fun checkDeviceReadiness() {
        val ready = fineTuningEngine.isDeviceReady()
        _uiState.update { it.copy(isDeviceReady = ready) }
    }

    private fun updateEstimatedTime() {
        val state = _uiState.value
        val sampleCount = state.dataStats.readyForTraining
        if (sampleCount > 0) {
            val estimate = fineTuningEngine.estimateTrainingTime(sampleCount, state.config)
            _uiState.update { it.copy(estimatedTrainingTimeMs = estimate) }
        }
    }

    // ========================================================================
    // Cloud Fine-Tuning Operations
    // ========================================================================

    fun updateCloudServerUrl(url: String) {
        val trimmed = url.trim()
        val normalized = when {
            trimmed.isBlank() -> ""
            trimmed.startsWith("http://") || trimmed.startsWith("https://") -> trimmed
            else -> "http://$trimmed"
        }
        cloudService.updateServerUrl(normalized)
        _cloudUiState.update {
            it.copy(
                isServerReachable = false,
                syncStatus = CloudSyncStatus.Disconnected,
                errorMessage = null,
            )
        }
    }

    fun saveCloudConfig() {
        // Config is auto-saved by CloudFineTuningService
        _cloudUiState.update {
            it.copy(
                syncStatus = CloudSyncStatus.Completed("Cloud server URL saved"),
                errorMessage = null,
            )
        }
    }

    fun checkCloudConnection() {
        viewModelScope.launch {
            val url = cloudService.config.value.serverUrl.trim()
            if (url.isBlank()) {
                _cloudUiState.update {
                    it.copy(
                        isServerReachable = false,
                        syncStatus = CloudSyncStatus.Error("Please enter a server URL"),
                        errorMessage = "Please enter a server URL",
                    )
                }
                return@launch
            }

            _cloudUiState.update {
                it.copy(
                    syncStatus = CloudSyncStatus.Connecting,
                    errorMessage = null,
                )
            }

            val reachable = cloudService.checkServerHealth()
            _cloudUiState.update {
                if (reachable) {
                    it.copy(
                        isServerReachable = true,
                        syncStatus = CloudSyncStatus.Connected,
                        errorMessage = null,
                    )
                } else {
                    val msg = "Cannot reach cloud server at $url"
                    it.copy(
                        isServerReachable = false,
                        syncStatus = CloudSyncStatus.Error(msg),
                        errorMessage = msg,
                    )
                }
            }
        }
    }

    /**
     * Upload all training data (text, voice, video) to the cloud server.
     */
    fun uploadTrainingDataToCloud() {
        viewModelScope.launch {
            try {
                val interactions = trainingDataStore.interactions.value
                    .filter { it.includeInTraining }

                if (interactions.isEmpty()) {
                    _cloudUiState.update { it.copy(errorMessage = "No training data to upload") }
                    return@launch
                }

                val baseModel = _uiState.value.currentModelId ?: ""
                val result = cloudService.uploadTrainingData(interactions, baseModel)

                result.onSuccess { response ->
                    _cloudUiState.update {
                        it.copy(
                            lastUploadId = response.uploadId,
                            lastSyncTime = System.currentTimeMillis(),
                        )
                    }
                    Log.i(TAG, "Cloud upload success: ${response.samplesReceived} samples")
                }

                result.onFailure { error ->
                    _cloudUiState.update { it.copy(errorMessage = error.message) }
                }
            } catch (e: Exception) {
                _cloudUiState.update { it.copy(errorMessage = "Upload failed: ${e.message}") }
            }
        }
    }

    /**
     * Start a training job on the cloud.
     */
    fun startCloudTraining() {
        viewModelScope.launch {
            try {
                val config = _uiState.value.config
                val name = _uiState.value.adapterName.ifBlank { "device-adapter-${System.currentTimeMillis()}" }

                val result = cloudService.startTraining(
                    adapterName = name,
                    baseModel = _uiState.value.currentModelId ?: "unsloth/Llama-3.2-1B",
                    loraRank = config.loraRank,
                    loraAlpha = config.loraAlpha,
                    learningRate = config.learningRate,
                    epochs = config.epochs,
                    outputFormat = "gguf",
                )

                result.onSuccess { response ->
                    _cloudUiState.update { it.copy(activeJobId = response.jobId) }
                    // Start polling for status
                    pollCloudJobStatus()
                }

                result.onFailure { error ->
                    _cloudUiState.update { it.copy(errorMessage = error.message) }
                }
            } catch (e: Exception) {
                _cloudUiState.update { it.copy(errorMessage = "Start training failed: ${e.message}") }
            }
        }
    }

    /**
     * Poll for the status of the active cloud training job.
     */
    fun pollCloudJobStatus() {
        val jobId = _cloudUiState.value.activeJobId ?: return

        viewModelScope.launch {
            val result = cloudService.getJobStatus(jobId)
            result.onSuccess { status ->
                _cloudUiState.update { it.copy(activeJobStatus = status) }

                // If still in progress, poll again after a delay
                if (status.status == "queued" || status.status == "processing") {
                    delay(5000)
                    pollCloudJobStatus()
                }

                // If completed, auto-check for adapters
                if (status.status == "completed") {
                    delay(2000)
                    pollCloudAdapters()
                }
            }
            result.onFailure { error ->
                Log.w(TAG, "Job status poll failed: ${error.message}")
            }
        }
    }

    /**
     * Poll the cloud for available OTA adapters.
     */
    fun pollCloudAdapters() {
        viewModelScope.launch {
            val result = cloudService.listCloudAdapters()
            result.onSuccess { response ->
                val downloadedFiles = cloudService.getLocalAdapterFiles()
                val downloadedIds = downloadedFiles.map {
                    it.nameWithoutExtension.removePrefix("adapter_")
                }.toSet()

                _cloudUiState.update {
                    it.copy(
                        availableAdapters = response.adapters,
                        downloadedAdapterIds = downloadedIds,
                    )
                }
            }
            result.onFailure { error ->
                Log.w(TAG, "Adapter poll failed: ${error.message}")
            }
        }
    }

    /**
     * Download a specific adapter from the cloud (OTA delivery).
     */
    fun downloadCloudAdapter(adapterId: String) {
        viewModelScope.launch {
            val result = cloudService.downloadAdapter(adapterId)
            result.onSuccess { file ->
                // Find adapter info
                val adapterInfo = _cloudUiState.value.availableAdapters.find { it.id == adapterId }

                // Register in the local AdapterStore
                val loraAdapter = LoraAdapter(
                    id = adapterId,
                    name = adapterInfo?.name ?: "Cloud Adapter",
                    description = "Cloud-trained OTA adapter",
                    baseModelId = adapterInfo?.baseModel ?: "",
                    baseModelName = adapterInfo?.baseModel ?: "",
                    config = _uiState.value.config,
                    trainingSampleCount = adapterInfo?.trainingSamples ?: 0,
                    adapterFilePath = file.absolutePath,
                    adapterSizeBytes = file.length(),
                    specializations = listOf("cloud-trained", "ota"),
                )

                adapterStore.saveAdapter(loraAdapter)

                // Update downloaded IDs
                _cloudUiState.update {
                    it.copy(downloadedAdapterIds = it.downloadedAdapterIds + adapterId)
                }

                Log.i(TAG, "Cloud adapter downloaded and registered: ${loraAdapter.name}")
            }
            result.onFailure { error ->
                _cloudUiState.update { it.copy(errorMessage = "Download failed: ${error.message}") }
            }
        }
    }

    /**
     * Toggle background auto-sync for OTA adapters.
     */
    fun toggleAutoSync(enabled: Boolean) {
        cloudService.updateConfig(cloudService.config.value.copy(autoSync = enabled))
        if (enabled) {
            OtaAdapterReceiver.schedule(getApplication(), intervalMinutes = 15)
        } else {
            OtaAdapterReceiver.cancel(getApplication())
        }
    }

    fun clearCloudError() {
        _cloudUiState.update { it.copy(errorMessage = null) }
        cloudService.resetSyncStatus()
    }

    companion object {
        private const val TAG = "FineTuneViewModel"
    }
}
