package com.runanywhere.runanywhereai.data.cloud

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.runanywhere.runanywhereai.data.AdapterStore
import com.runanywhere.runanywhereai.domain.models.FineTuningConfig
import com.runanywhere.runanywhereai.domain.models.LoraAdapter
import java.util.concurrent.TimeUnit

/**
 * OtaAdapterReceiver — WorkManager periodic worker that polls the cloud server
 * for new adapters and downloads them OTA (Over-The-Air) to the device.
 *
 * This runs in the background even when the app is closed, checking for newly
 * trained adapters at a configurable interval.
 *
 * Flow:
 * 1. Poll cloud server for new adapters (GET /api/v1/adapters/poll)
 * 2. If new adapters available, download them (GET /api/v1/adapters/download/{id})
 * 3. Register downloaded adapter in AdapterStore for on-device use
 * 4. Optionally auto-apply the adapter
 */
class OtaAdapterReceiver(
    context: Context,
    workerParams: WorkerParameters,
) : CoroutineWorker(context, workerParams) {

    companion object {
        private const val TAG = "OtaAdapterReceiver"
        private const val WORK_NAME = "ota_adapter_poll"

        /**
         * Schedule periodic adapter polling.
         *
         * @param context Application context
         * @param intervalMinutes How often to poll (minimum 15 min per WorkManager)
         */
        fun schedule(context: Context, intervalMinutes: Long = 15) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build()

            val workRequest = PeriodicWorkRequestBuilder<OtaAdapterReceiver>(
                intervalMinutes, TimeUnit.MINUTES,
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    1, TimeUnit.MINUTES,
                )
                .build()

            WorkManager.getInstance(context)
                .enqueueUniquePeriodicWork(
                    WORK_NAME,
                    ExistingPeriodicWorkPolicy.UPDATE,
                    workRequest,
                )

            Log.i(TAG, "OTA adapter polling scheduled every $intervalMinutes minutes")
        }

        /**
         * Cancel periodic polling.
         */
        fun cancel(context: Context) {
            WorkManager.getInstance(context).cancelUniqueWork(WORK_NAME)
            Log.i(TAG, "OTA adapter polling cancelled")
        }
    }

    override suspend fun doWork(): Result {
        Log.d(TAG, "Starting OTA adapter poll...")

        val cloudService = CloudFineTuningService.getInstance(applicationContext)
        val adapterStore = AdapterStore.getInstance(applicationContext)
        val config = cloudService.config.value

        // Skip if no server URL configured
        if (config.serverUrl.isBlank()) {
            Log.d(TAG, "No cloud server configured, skipping poll")
            return Result.success()
        }

        // Check server health
        val isHealthy = cloudService.checkServerHealth()
        if (!isHealthy) {
            Log.w(TAG, "Cloud server not reachable, will retry")
            return Result.retry()
        }

        // Poll for new adapters
        val pollResult = cloudService.pollForAdapters()
        pollResult.onSuccess { poll ->
            if (!poll.hasNewAdapter || poll.adapters.isEmpty()) {
                Log.d(TAG, "No new adapters available")
                return Result.success()
            }

            Log.i(TAG, "Found ${poll.adapters.size} new adapter(s)!")

            // Download each new adapter
            for (adapterInfo in poll.adapters) {
                val alreadyDownloaded = adapterStore.adapters.value
                    .any { it.id == adapterInfo.id || it.name == adapterInfo.name }

                if (alreadyDownloaded) {
                    Log.d(TAG, "Adapter ${adapterInfo.name} already exists, skipping")
                    continue
                }

                Log.i(TAG, "Downloading adapter: ${adapterInfo.name} (${adapterInfo.sizeBytes} bytes)")
                val downloadResult = cloudService.downloadAdapter(adapterInfo.id)

                downloadResult.onSuccess { file ->
                    // Register the downloaded adapter in the local AdapterStore
                    val loraAdapter = LoraAdapter(
                        id = adapterInfo.id,
                        name = adapterInfo.name,
                        description = "Cloud-trained adapter (${adapterInfo.trainingSamples} samples)",
                        baseModelId = adapterInfo.baseModel,
                        baseModelName = adapterInfo.baseModel,
                        config = FineTuningConfig(), // Default config
                        trainingSampleCount = adapterInfo.trainingSamples,
                        adapterFilePath = file.absolutePath,
                        adapterSizeBytes = file.length(),
                        specializations = listOf("cloud-trained"),
                    )

                    adapterStore.saveAdapter(loraAdapter)
                    Log.i(TAG, "Adapter registered: ${loraAdapter.name} at ${file.absolutePath}")

                    // Auto-apply if configured
                    if (config.autoSync) {
                        adapterStore.activateAdapter(loraAdapter.id)
                        Log.i(TAG, "Auto-applied adapter: ${loraAdapter.name}")
                    }
                }

                downloadResult.onFailure { error ->
                    Log.e(TAG, "Failed to download adapter ${adapterInfo.name}: ${error.message}")
                }
            }
        }

        pollResult.onFailure { error ->
            Log.e(TAG, "Adapter poll failed: ${error.message}")
            return Result.retry()
        }

        return Result.success()
    }
}
