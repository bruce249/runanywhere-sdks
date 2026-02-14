package com.runanywhere.sdk.features.ota

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

/**
 * RunAnywhere OTA Fine-Tuning Client
 *
 * The device-side client for the OTA fine-tuning system.
 * Handles:
 *   1. Uploading training data (interactions) to the cloud server
 *   2. Polling for newly trained LoRA adapters
 *   3. Downloading adapters
 *   4. Managing local adapter storage
 *
 * Usage:
 * ```kotlin
 * val client = OtaFineTuningClient(
 *     config = OtaFineTuningConfig(
 *         cloudEndpoint = "http://192.168.1.100:8000",
 *     ),
 *     adapterStorageDir = context.filesDir.resolve("OtaAdapters"),
 * )
 *
 * // Upload training data
 * client.uploadTrainingData(interactions, modelId = "llama-3.2-1b")
 *
 * // Poll for new adapters
 * client.pollForAdapters()
 *
 * // Events
 * client.events.collect { event ->
 *     when (event) {
 *         is OtaFineTuningEvent.AdapterDownloaded -> applyAdapter(event.localPath)
 *     }
 * }
 * ```
 */
class OtaFineTuningClient(
    val config: OtaFineTuningConfig,
    private val adapterStorageDir: File,
    private val deviceId: String = config.deviceId ?: generateDeviceId(),
) {
    companion object {
        private const val TAG = "OtaFineTuningClient"

        private fun generateDeviceId(): String {
            return "device-${java.util.UUID.randomUUID().toString().take(8)}"
        }
    }

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        isLenient = true
        prettyPrint = false
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Events
    private val _events = MutableSharedFlow<OtaFineTuningEvent>(replay = 0)
    val events: SharedFlow<OtaFineTuningEvent> = _events.asSharedFlow()

    // State
    private val _status = MutableStateFlow(OtaStatus.IDLE)
    val status: StateFlow<OtaStatus> = _status.asStateFlow()

    // Local adapter registry
    private val localAdapters = ConcurrentHashMap<String, OtaAdapterInfo>()
    private var activeAdapterId: String? = null

    // Polling
    private var pollJob: Job? = null
    private var lastPollTimestamp: String? = null

    // Interaction buffer for auto-upload
    private val pendingInteractions = mutableListOf<OtaTrainingInteraction>()

    init {
        adapterStorageDir.mkdirs()
        loadLocalAdapterRegistry()
    }

    enum class OtaStatus {
        IDLE, UPLOADING, POLLING, DOWNLOADING, ERROR
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Training Data Upload
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Buffer an interaction for later upload.
     * If auto-upload is enabled and threshold is reached, triggers upload.
     */
    fun recordInteraction(interaction: OtaTrainingInteraction) {
        synchronized(pendingInteractions) {
            pendingInteractions.add(interaction)

            if (config.autoUpload && pendingInteractions.size >= config.autoUploadThreshold) {
                val toUpload = pendingInteractions.toList()
                pendingInteractions.clear()
                scope.launch { uploadTrainingData(toUpload, interaction.modelId ?: "") }
            }
        }
    }

    /**
     * Upload training interactions to the cloud server.
     */
    suspend fun uploadTrainingData(
        interactions: List<OtaTrainingInteraction>,
        modelId: String,
        modelName: String? = null,
    ): Result<TrainingDataUploadResponse> {
        if (interactions.isEmpty()) {
            return Result.failure(IllegalArgumentException("No interactions to upload"))
        }

        _status.value = OtaStatus.UPLOADING

        return try {
            val request = TrainingDataUploadRequest(
                deviceId = deviceId,
                modelId = modelId,
                modelName = modelName,
                interactions = interactions,
            )

            val response = httpPost<TrainingDataUploadResponse>(
                path = "/api/v1/training/upload",
                body = json.encodeToString(request),
            )

            _events.emit(
                OtaFineTuningEvent.DataUploaded(
                    uploadId = response.uploadId,
                    sampleCount = response.sampleCount,
                    totalSamples = response.totalSamplesForDevice,
                )
            )

            _status.value = OtaStatus.IDLE
            Result.success(response)

        } catch (e: Exception) {
            _status.value = OtaStatus.ERROR
            _events.emit(
                OtaFineTuningEvent.Error("upload", e.message ?: "Upload failed", e)
            )
            Result.failure(e)
        }
    }

    /**
     * Flush any pending interactions (force upload even if below threshold).
     */
    suspend fun flushPendingInteractions(modelId: String): Result<TrainingDataUploadResponse>? {
        val toUpload: List<OtaTrainingInteraction>
        synchronized(pendingInteractions) {
            if (pendingInteractions.isEmpty()) return null
            toUpload = pendingInteractions.toList()
            pendingInteractions.clear()
        }
        return uploadTrainingData(toUpload, modelId)
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Training Job Management
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Request the cloud server to start a fine-tuning job.
     */
    suspend fun startTrainingJob(
        modelId: String,
        baseModel: String? = null,
        uploadIds: List<String>? = null,
        adapterName: String? = null,
    ): Result<TrainingJobResponse> {
        return try {
            val request = StartTrainingRequest(
                deviceId = deviceId,
                modelId = modelId,
                baseModel = baseModel,
                uploadIds = uploadIds,
                adapterName = adapterName,
            )

            val response = httpPost<TrainingJobResponse>(
                path = "/api/v1/training/start",
                body = json.encodeToString(request),
            )

            Result.success(response)
        } catch (e: Exception) {
            _events.emit(
                OtaFineTuningEvent.Error("start_training", e.message ?: "Failed", e)
            )
            Result.failure(e)
        }
    }

    /**
     * Check the status of a training job.
     */
    suspend fun getTrainingStatus(jobId: String): Result<TrainingJobResponse> {
        return try {
            Result.success(httpGet("/api/v1/training/status/$jobId"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Observe training job progress as a Flow.
     */
    fun observeTrainingJob(
        jobId: String,
        pollIntervalMs: Long = 5000,
    ): Flow<TrainingJobResponse> = flow {
        while (true) {
            val result = getTrainingStatus(jobId)
            result.onSuccess { job ->
                emit(job)
                if (job.status in listOf("completed", "failed", "cancelled")) {
                    return@flow
                }
            }
            result.onFailure {
                throw it
            }
            delay(pollIntervalMs)
        }
    }.flowOn(Dispatchers.IO)

    // ═════════════════════════════════════════════════════════════════════════
    // Adapter Polling & Download
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Poll the cloud server for new adapters.
     */
    suspend fun pollForAdapters(
        modelId: String? = null,
    ): Result<AdapterPollResponse> {
        _status.value = OtaStatus.POLLING

        return try {
            var path = "/api/v1/adapters/poll?device_id=$deviceId"
            if (modelId != null) path += "&model_id=$modelId"
            if (lastPollTimestamp != null) path += "&since=$lastPollTimestamp"

            val response: AdapterPollResponse = httpGet(path)

            if (response.hasNewAdapters) {
                for (adapter in response.adapters) {
                    _events.emit(OtaFineTuningEvent.AdapterAvailable(adapter))

                    // Auto-download if configured
                    if (config.autoApplyAdapter) {
                        downloadAdapter(adapter.id)
                    }
                }
            }

            // Update poll timestamp
            lastPollTimestamp = java.time.Instant.now().toString()

            _status.value = OtaStatus.IDLE
            Result.success(response)

        } catch (e: Exception) {
            _status.value = OtaStatus.ERROR
            _events.emit(
                OtaFineTuningEvent.Error("poll", e.message ?: "Poll failed", e)
            )
            Result.failure(e)
        }
    }

    /**
     * Start automatic polling at the configured interval.
     */
    fun startAutoPolling(modelId: String? = null) {
        stopAutoPolling()
        pollJob = scope.launch {
            while (isActive) {
                pollForAdapters(modelId)
                delay(config.pollIntervalSeconds * 1000L)
            }
        }
    }

    /**
     * Stop automatic polling.
     */
    fun stopAutoPolling() {
        pollJob?.cancel()
        pollJob = null
    }

    /**
     * Download an adapter from the cloud server.
     */
    suspend fun downloadAdapter(adapterId: String): Result<OtaAdapterInfo> {
        _status.value = OtaStatus.DOWNLOADING

        return try {
            // Get adapter info first
            val adapterInfo: OtaAdapterInfo = httpGet("/api/v1/adapters/$adapterId")

            // Download the file
            val adapterDir = File(adapterStorageDir, adapterId)
            adapterDir.mkdirs()

            val fileName = when (adapterInfo.format.lowercase()) {
                "gguf" -> "adapter.gguf"
                "safetensors" -> "adapter_model.safetensors"
                else -> "adapter.bin"
            }
            val localFile = File(adapterDir, fileName)

            downloadFile(
                path = "/api/v1/adapters/download/$adapterId",
                destination = localFile,
                onProgress = { progress ->
                    scope.launch {
                        _events.emit(OtaFineTuningEvent.DownloadProgress(adapterId, progress))
                    }
                },
            )

            // Verify checksum
            if (adapterInfo.checksum != null) {
                val localChecksum = computeSha256(localFile)
                if (localChecksum != adapterInfo.checksum) {
                    localFile.delete()
                    throw SecurityException(
                        "Checksum mismatch: expected ${adapterInfo.checksum}, got $localChecksum"
                    )
                }
            }

            // Update local registry
            adapterInfo.localFilePath = localFile.absolutePath
            localAdapters[adapterId] = adapterInfo
            saveLocalAdapterRegistry()

            _events.emit(
                OtaFineTuningEvent.AdapterDownloaded(adapterInfo, localFile.absolutePath)
            )

            // Auto-apply if configured
            if (config.autoApplyAdapter) {
                applyAdapter(adapterId)
            }

            _status.value = OtaStatus.IDLE
            Result.success(adapterInfo)

        } catch (e: Exception) {
            _status.value = OtaStatus.ERROR
            _events.emit(
                OtaFineTuningEvent.Error("download", e.message ?: "Download failed", e)
            )
            Result.failure(e)
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Local Adapter Management
    // ═════════════════════════════════════════════════════════════════════════

    /**
     * Apply a downloaded adapter to the active LLM model.
     *
     * This notifies the LLM service to load the LoRA adapter weights.
     * The actual application happens via the llama.cpp backend's LoRA loading.
     */
    fun applyAdapter(adapterId: String): Boolean {
        val adapter = localAdapters[adapterId] ?: return false
        val filePath = adapter.localFilePath ?: return false

        if (!File(filePath).exists()) return false

        // Deactivate current adapter
        activeAdapterId?.let { currentId ->
            localAdapters[currentId]?.isActive = false
        }

        // Activate new adapter
        adapter.isActive = true
        activeAdapterId = adapterId
        saveLocalAdapterRegistry()

        scope.launch {
            _events.emit(OtaFineTuningEvent.AdapterApplied(adapter))
        }

        return true
    }

    /**
     * Remove the currently active adapter.
     */
    fun removeActiveAdapter() {
        activeAdapterId?.let { id ->
            localAdapters[id]?.isActive = false
        }
        activeAdapterId = null
        saveLocalAdapterRegistry()
    }

    /**
     * Get all locally available adapters.
     */
    fun getLocalAdapters(): List<OtaAdapterInfo> = localAdapters.values.toList()

    /**
     * Get the currently active adapter.
     */
    fun getActiveAdapter(): OtaAdapterInfo? = activeAdapterId?.let { localAdapters[it] }

    /**
     * Delete a locally stored adapter.
     */
    fun deleteLocalAdapter(adapterId: String) {
        if (activeAdapterId == adapterId) removeActiveAdapter()

        localAdapters.remove(adapterId)
        val adapterDir = File(adapterStorageDir, adapterId)
        adapterDir.deleteRecursively()
        saveLocalAdapterRegistry()
    }

    /**
     * Shutdown the client and release resources.
     */
    fun shutdown() {
        stopAutoPolling()
        scope.cancel()
    }

    // ═════════════════════════════════════════════════════════════════════════
    // HTTP Helpers
    // ═════════════════════════════════════════════════════════════════════════

    private inline fun <reified T> httpPost(path: String, body: String): T {
        val url = URL("${config.cloudEndpoint}$path")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("X-API-Key", config.apiKey)
            setRequestProperty("X-Device-Id", deviceId)
            doOutput = true
            connectTimeout = 30_000
            readTimeout = 60_000
        }

        conn.outputStream.use { os ->
            os.write(body.toByteArray(Charsets.UTF_8))
        }

        if (conn.responseCode !in 200..299) {
            val error = conn.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
            throw RuntimeException("HTTP ${conn.responseCode}: $error")
        }

        val responseBody = conn.inputStream.bufferedReader().readText()
        return json.decodeFromString(responseBody)
    }

    private inline fun <reified T> httpGet(path: String): T {
        val url = URL("${config.cloudEndpoint}$path")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            setRequestProperty("X-API-Key", config.apiKey)
            setRequestProperty("X-Device-Id", deviceId)
            connectTimeout = 30_000
            readTimeout = 60_000
        }

        if (conn.responseCode !in 200..299) {
            val error = conn.errorStream?.bufferedReader()?.readText() ?: "Unknown error"
            throw RuntimeException("HTTP ${conn.responseCode}: $error")
        }

        val responseBody = conn.inputStream.bufferedReader().readText()
        return json.decodeFromString(responseBody)
    }

    private fun downloadFile(
        path: String,
        destination: File,
        onProgress: (Float) -> Unit,
    ) {
        val url = URL("${config.cloudEndpoint}$path")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            setRequestProperty("X-API-Key", config.apiKey)
            connectTimeout = 30_000
            readTimeout = 300_000 // 5 min for large adapters
        }

        if (conn.responseCode !in 200..299) {
            throw RuntimeException("Download failed: HTTP ${conn.responseCode}")
        }

        val totalBytes = conn.contentLengthLong
        var downloadedBytes = 0L
        val buffer = ByteArray(8192)

        conn.inputStream.use { input ->
            FileOutputStream(destination).use { output ->
                while (true) {
                    val bytesRead = input.read(buffer)
                    if (bytesRead == -1) break

                    output.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead

                    if (totalBytes > 0) {
                        onProgress(downloadedBytes.toFloat() / totalBytes)
                    }
                }
            }
        }

        onProgress(1.0f)
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Local Registry Persistence
    // ═════════════════════════════════════════════════════════════════════════

    private val registryFile = File(adapterStorageDir, "adapter_registry.json")

    private fun loadLocalAdapterRegistry() {
        if (!registryFile.exists()) return

        try {
            val data = registryFile.readText()
            val adapters: List<OtaAdapterInfo> = json.decodeFromString(data)
            for (adapter in adapters) {
                localAdapters[adapter.id] = adapter
                if (adapter.isActive) {
                    activeAdapterId = adapter.id
                }
            }
        } catch (e: Exception) {
            // Corrupted registry — start fresh
            registryFile.delete()
        }
    }

    private fun saveLocalAdapterRegistry() {
        try {
            val adapters = localAdapters.values.toList()
            registryFile.writeText(json.encodeToString(adapters))
        } catch (e: Exception) {
            // Non-critical
        }
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val bytesRead = input.read(buffer)
                if (bytesRead == -1) break
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
