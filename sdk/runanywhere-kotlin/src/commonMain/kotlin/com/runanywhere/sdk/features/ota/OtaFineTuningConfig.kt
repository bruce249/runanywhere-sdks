package com.runanywhere.sdk.features.ota

/**
 * RunAnywhere OTA Fine-Tuning Client Configuration.
 *
 * Configure the connection to your fine-tuning cloud server
 * (laptop or remote) for uploading training data and receiving
 * LoRA adapters over-the-air.
 */
data class OtaFineTuningConfig(
    /**
     * Cloud server endpoint URL.
     * Example: "http://192.168.1.100:8000" (your laptop on WiFi)
     */
    val cloudEndpoint: String,

    /**
     * API key for authentication.
     * For local dev, use the key from your server's .env file.
     */
    val apiKey: String = "runanywhere-dev-key",

    /**
     * Unique device identifier.
     * Auto-generated if not provided.
     */
    val deviceId: String? = null,

    /**
     * Automatically upload interactions when threshold is reached.
     */
    val autoUpload: Boolean = true,

    /**
     * Minimum number of interactions before auto-upload triggers.
     */
    val autoUploadThreshold: Int = 20,

    /**
     * Poll interval for checking new adapters.
     * Set to null to disable auto-polling.
     */
    val pollIntervalSeconds: Int = 60,

    /**
     * Automatically apply downloaded adapters to the active model.
     */
    val autoApplyAdapter: Boolean = true,

    /**
     * Maximum local adapter cache size in bytes (0 = unlimited).
     */
    val maxCacheSizeBytes: Long = 0,
)
