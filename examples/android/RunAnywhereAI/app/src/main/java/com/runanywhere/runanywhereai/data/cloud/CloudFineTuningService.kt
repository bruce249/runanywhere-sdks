package com.runanywhere.runanywhereai.data.cloud

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import com.runanywhere.runanywhereai.domain.models.TrainingInteraction
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * CloudFineTuningService — Singleton HTTP client that communicates
 * with the cloud fine-tuning server (FastAPI on laptop).
 *
 * Responsibilities:
 * - Upload multi-modal training data (text, voice, video analysis)
 * - Start cloud training jobs
 * - Poll for training job status
 * - Download OTA adapters
 * - Health check / connectivity
 *
 * Follows the same getInstance(context) singleton pattern as other stores.
 */
class CloudFineTuningService private constructor(context: Context) {

    companion object {
        private const val TAG = "CloudFineTuneService"
        private const val CONFIG_FILE = "cloud_config.json"
        private const val ADAPTERS_DIR = "CloudAdapters"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: CloudFineTuningService? = null

        fun getInstance(context: Context): CloudFineTuningService {
            return instance ?: synchronized(this) {
                instance ?: CloudFineTuningService(context.applicationContext).also { instance = it }
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val json = Json {
        prettyPrint = false
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    private val adaptersDirectory: File = File(appContext.filesDir, ADAPTERS_DIR).also { it.mkdirs() }
    private val configFile: File = File(appContext.filesDir, CONFIG_FILE)

    // State
    private val _config = MutableStateFlow(CloudServerConfig())
    val config: StateFlow<CloudServerConfig> = _config.asStateFlow()

    private val _syncStatus = MutableStateFlow<CloudSyncStatus>(CloudSyncStatus.Disconnected)
    val syncStatus: StateFlow<CloudSyncStatus> = _syncStatus.asStateFlow()

    init {
        loadConfig()
        if (_config.value.deviceId.isBlank()) {
            updateConfig(_config.value.copy(deviceId = UUID.randomUUID().toString().take(8)))
        }
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    fun updateConfig(newConfig: CloudServerConfig) {
        _config.value = newConfig
        saveConfig()
    }

    fun updateServerUrl(url: String) {
        updateConfig(_config.value.copy(serverUrl = url.trimEnd('/')))
    }

    private fun loadConfig() {
        try {
            if (configFile.exists()) {
                _config.value = json.decodeFromString<CloudServerConfig>(configFile.readText())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load config: ${e.message}")
        }
    }

    private fun saveConfig() {
        try {
            configFile.writeText(json.encodeToString(_config.value))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save config: ${e.message}")
        }
    }

    private fun baseUrl(): String = _config.value.serverUrl

    // ========================================================================
    // Health Check
    // ========================================================================

    /**
     * Check if the cloud server is reachable.
     */
    suspend fun checkServerHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl()}/")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                response.isSuccessful
            }
        } catch (e: Exception) {
            Log.w(TAG, "Server health check failed: ${e.message}")
            false
        }
    }

    // ========================================================================
    // Upload Training Data
    // ========================================================================

    /**
     * Upload all training interactions to the cloud server.
     * Converts the app's TrainingInteraction format to the server's expected format.
     */
    suspend fun uploadTrainingData(
        interactions: List<TrainingInteraction>,
        baseModel: String = "",
    ): Result<CloudUploadResponse> = withContext(Dispatchers.IO) {
        try {
            _syncStatus.value = CloudSyncStatus.Uploading(0f, "Preparing data...")

            val cloudInteractions = interactions.map { it.toCloudInteraction() }
            val uploadRequest = CloudUploadRequest(
                deviceId = _config.value.deviceId,
                baseModel = baseModel,
                interactions = cloudInteractions,
            )

            _syncStatus.value = CloudSyncStatus.Uploading(0.3f, "Uploading ${interactions.size} interactions...")

            val requestBody = json.encodeToString(uploadRequest)
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/training/upload")
                .post(requestBody)
                .apply {
                    if (_config.value.apiKey.isNotBlank()) {
                        addHeader("X-API-Key", _config.value.apiKey)
                    }
                }
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val result = json.decodeFromString<CloudUploadResponse>(body)
                    _syncStatus.value = CloudSyncStatus.Completed(
                        "Uploaded ${result.samplesReceived} samples successfully"
                    )
                    Log.i(TAG, "Upload successful: ${result.samplesReceived} samples, upload_id=${result.uploadId}")
                    Result.success(result)
                } else {
                    val error = "Upload failed (${response.code}): $body"
                    _syncStatus.value = CloudSyncStatus.Error(error)
                    Log.e(TAG, error)
                    Result.failure(Exception(error))
                }
            }
        } catch (e: Exception) {
            val error = "Upload failed: ${e.message}"
            _syncStatus.value = CloudSyncStatus.Error(error)
            Log.e(TAG, error, e)
            Result.failure(e)
        }
    }

    // ========================================================================
    // Start Training
    // ========================================================================

    /**
     * Start a cloud training job.
     */
    suspend fun startTraining(
        adapterName: String = "device-adapter",
        baseModel: String = "unsloth/Llama-3.2-1B",
        loraRank: Int = 8,
        loraAlpha: Int = 16,
        learningRate: Float = 2e-4f,
        epochs: Int = 3,
        outputFormat: String = "gguf",
    ): Result<CloudTrainResponse> = withContext(Dispatchers.IO) {
        try {
            _syncStatus.value = CloudSyncStatus.Uploading(0.5f, "Starting cloud training...")

            val trainRequest = CloudTrainRequest(
                deviceId = _config.value.deviceId,
                baseModel = baseModel,
                adapterName = adapterName,
                loraRank = loraRank,
                loraAlpha = loraAlpha,
                learningRate = learningRate,
                epochs = epochs,
                outputFormat = outputFormat,
            )

            val requestBody = json.encodeToString(trainRequest)
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/training/start")
                .post(requestBody)
                .apply {
                    if (_config.value.apiKey.isNotBlank()) {
                        addHeader("X-API-Key", _config.value.apiKey)
                    }
                }
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val result = json.decodeFromString<CloudTrainResponse>(body)
                    _syncStatus.value = CloudSyncStatus.Training(
                        jobId = result.jobId,
                        progress = 0f,
                        status = "Training started"
                    )
                    Log.i(TAG, "Training started: job_id=${result.jobId}")
                    Result.success(result)
                } else {
                    val error = "Start training failed (${response.code}): $body"
                    _syncStatus.value = CloudSyncStatus.Error(error)
                    Result.failure(Exception(error))
                }
            }
        } catch (e: Exception) {
            val error = "Start training failed: ${e.message}"
            _syncStatus.value = CloudSyncStatus.Error(error)
            Result.failure(e)
        }
    }

    // ========================================================================
    // Poll Training Status
    // ========================================================================

    /**
     * Check the status of a training job.
     */
    suspend fun getJobStatus(jobId: String): Result<CloudJobStatus> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/training/status/$jobId")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val status = json.decodeFromString<CloudJobStatus>(body)

                    // Update sync status based on job progress
                    when (status.status) {
                        "queued", "processing" -> {
                            _syncStatus.value = CloudSyncStatus.Training(
                                jobId = jobId,
                                progress = status.progress,
                                status = "Training in progress..."
                            )
                        }
                        "completed" -> {
                            _syncStatus.value = CloudSyncStatus.Completed(
                                "Training completed! Adapter ready for download."
                            )
                        }
                        "failed" -> {
                            _syncStatus.value = CloudSyncStatus.Error(
                                "Training failed: ${status.errorMessage ?: "Unknown error"}"
                            )
                        }
                    }

                    Result.success(status)
                } else {
                    Result.failure(Exception("Status check failed (${response.code}): $body"))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ========================================================================
    // Poll for Adapters
    // ========================================================================

    /**
     * Poll the cloud for new adapters available for this device.
     */
    suspend fun pollForAdapters(): Result<CloudAdapterPollResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/adapters/poll?device_id=${_config.value.deviceId}")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val result = json.decodeFromString<CloudAdapterPollResponse>(body)
                    Result.success(result)
                } else {
                    Result.failure(Exception("Poll failed (${response.code}): $body"))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * List all adapters on the cloud.
     */
    suspend fun listCloudAdapters(): Result<CloudAdapterListResponse> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/adapters/list")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val result = json.decodeFromString<CloudAdapterListResponse>(body)
                    Result.success(result)
                } else {
                    Result.failure(Exception("List adapters failed (${response.code}): $body"))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ========================================================================
    // Download Adapter (OTA)
    // ========================================================================

    /**
     * Download an adapter from the cloud to local device storage.
     * This is the OTA delivery mechanism.
     */
    suspend fun downloadAdapter(adapterId: String): Result<File> = withContext(Dispatchers.IO) {
        try {
            _syncStatus.value = CloudSyncStatus.DownloadingAdapter(0f)

            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/adapters/download/$adapterId")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val error = "Download failed (${response.code}): ${response.body?.string()}"
                    _syncStatus.value = CloudSyncStatus.Error(error)
                    return@withContext Result.failure(Exception(error))
                }

                val body = response.body ?: run {
                    _syncStatus.value = CloudSyncStatus.Error("Empty response body")
                    return@withContext Result.failure(Exception("Empty response body"))
                }

                val contentLength = body.contentLength()
                val adapterFile = File(adaptersDirectory, "adapter_$adapterId.gguf")

                body.byteStream().use { inputStream ->
                    FileOutputStream(adapterFile).use { outputStream ->
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        var totalBytesRead = 0L

                        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                            outputStream.write(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead

                            if (contentLength > 0) {
                                val progress = totalBytesRead.toFloat() / contentLength
                                _syncStatus.value = CloudSyncStatus.DownloadingAdapter(progress)
                            }
                        }
                    }
                }

                _syncStatus.value = CloudSyncStatus.Completed(
                    "Adapter downloaded (${adapterFile.length() / 1024}KB)"
                )
                Log.i(TAG, "Adapter downloaded: ${adapterFile.absolutePath} (${adapterFile.length()} bytes)")
                Result.success(adapterFile)
            }
        } catch (e: Exception) {
            val error = "Download failed: ${e.message}"
            _syncStatus.value = CloudSyncStatus.Error(error)
            Log.e(TAG, error, e)
            Result.failure(e)
        }
    }

    // ========================================================================
    // List Jobs
    // ========================================================================

    /**
     * Get all training jobs from the cloud.
     */
    suspend fun listJobs(): Result<List<CloudJobStatus>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${baseUrl()}/api/v1/training/jobs?device_id=${_config.value.deviceId}")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val body = response.body?.string() ?: ""
                if (response.isSuccessful) {
                    val jobs = json.decodeFromString<List<CloudJobStatus>>(body)
                    Result.success(jobs)
                } else {
                    Result.failure(Exception("List jobs failed (${response.code}): $body"))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ========================================================================
    // Reset
    // ========================================================================

    fun resetSyncStatus() {
        _syncStatus.value = CloudSyncStatus.Disconnected
    }

    /**
     * Get local adapter files that have been downloaded.
     */
    fun getLocalAdapterFiles(): List<File> {
        return adaptersDirectory.listFiles()?.filter { it.extension == "gguf" } ?: emptyList()
    }

    // ========================================================================
    // Helper — Convert domain model to cloud format
    // ========================================================================

    private fun TrainingInteraction.toCloudInteraction(): CloudInteraction {
        return CloudInteraction(
            id = id,
            userPrompt = userPrompt,
            modelResponse = modelResponse,
            correctedResponse = correctedResponse,
            rating = rating,
            isPositive = isPositiveExample,
            category = category,
            dataSource = dataSource,
            conversationContext = conversationContext?.map { ctx ->
                CloudContextMessage(role = ctx.role, content = ctx.content)
            },
            includeInTraining = includeInTraining,
            tags = tags,
            timestamp = timestamp,
            frequencyCount = frequencyCount,
        )
    }
}
