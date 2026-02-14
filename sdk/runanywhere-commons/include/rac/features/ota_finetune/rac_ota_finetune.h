/**
 * @file rac_ota_finetune.h
 * @brief RunAnywhere OTA Fine-Tuning — C API
 *
 * Provides the device-side interface for OTA fine-tuning:
 *   1. Upload training data (interactions) to a cloud endpoint
 *   2. Poll for newly trained LoRA adapters
 *   3. Download and manage adapters locally
 *   4. Apply adapters to the LLM inference engine
 *
 * This integrates with the cloud-fine-tuning-server running on
 * the user's laptop (or any cloud host).
 *
 * Thread Safety: All functions are thread-safe. Callbacks are invoked
 * on background threads — callers must dispatch to main if needed.
 */

#ifndef RAC_OTA_FINETUNE_H
#define RAC_OTA_FINETUNE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Types & Enums
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Status of an OTA fine-tuning operation.
 */
typedef enum {
    RAC_OTA_STATUS_IDLE         = 0,
    RAC_OTA_STATUS_UPLOADING    = 1,
    RAC_OTA_STATUS_POLLING      = 2,
    RAC_OTA_STATUS_DOWNLOADING  = 3,
    RAC_OTA_STATUS_APPLYING     = 4,
    RAC_OTA_STATUS_COMPLETED    = 5,
    RAC_OTA_STATUS_FAILED       = 6,
} rac_ota_status_t;

/**
 * Format of a LoRA adapter file.
 */
typedef enum {
    RAC_ADAPTER_FORMAT_GGUF         = 0,
    RAC_ADAPTER_FORMAT_SAFETENSORS  = 1,
    RAC_ADAPTER_FORMAT_BIN          = 2,
} rac_adapter_format_t;

/**
 * Configuration for the OTA fine-tuning client.
 */
typedef struct {
    /** Cloud server endpoint URL (e.g., "http://192.168.1.100:8000") */
    const char* cloud_endpoint;

    /** Unique device identifier */
    const char* device_id;

    /** API key for authentication */
    const char* api_key;

    /** Auto-upload interactions when threshold is reached */
    int auto_upload;

    /** Minimum interactions before auto-upload triggers */
    int auto_upload_threshold;

    /** Poll interval in seconds (0 = manual poll only) */
    int poll_interval_seconds;

    /** Auto-apply downloaded adapters */
    int auto_apply_adapter;

    /** Local directory for adapter storage */
    const char* adapter_storage_dir;

    /** Maximum adapter cache size in bytes (0 = unlimited) */
    uint64_t max_cache_size_bytes;
} rac_ota_config_t;

/**
 * Metadata for a trained LoRA adapter.
 */
typedef struct {
    const char*         id;
    const char*         name;
    const char*         description;
    const char*         model_id;
    const char*         base_model;
    rac_adapter_format_t format;
    uint64_t            file_size_bytes;
    int                 lora_rank;
    int                 lora_alpha;
    int                 version;
    const char*         checksum;
    int                 training_samples;
    float               final_loss;
    const char*         created_at;     /* ISO 8601 */
    const char*         local_file_path; /* NULL if not downloaded */
    int                 is_active;
} rac_adapter_info_t;

/**
 * Training data interaction to upload.
 */
typedef struct {
    const char* id;
    const char* user_prompt;
    const char* model_response;
    const char* corrected_response;  /* NULL if no correction */
    int         rating;              /* 0 if not rated */
    int         is_positive_example;
    int         include_in_training;
    const char* model_id;
    const char* category;
    int64_t     timestamp;
} rac_training_interaction_t;

/**
 * Result of a data upload.
 */
typedef struct {
    const char* upload_id;
    int         sample_count;
    int         total_samples_for_device;
    int         success;
    const char* error_message;
} rac_upload_result_t;

/**
 * Result of a poll operation.
 */
typedef struct {
    int                     has_new_adapters;
    rac_adapter_info_t*     adapters;
    int                     adapter_count;
    int                     poll_interval_seconds;
} rac_poll_result_t;


/* ═══════════════════════════════════════════════════════════════════════════
 * Callbacks
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Progress callback for uploads/downloads (progress: 0.0 - 1.0) */
typedef void (*rac_ota_progress_callback_t)(
    float progress,
    const char* status_message,
    void* user_data
);

/** Completion callback for upload */
typedef void (*rac_upload_complete_callback_t)(
    const rac_upload_result_t* result,
    void* user_data
);

/** Callback when new adapters are discovered */
typedef void (*rac_adapter_available_callback_t)(
    const rac_adapter_info_t* adapter,
    void* user_data
);

/** Callback when adapter download completes */
typedef void (*rac_adapter_downloaded_callback_t)(
    const rac_adapter_info_t* adapter,
    int success,
    const char* error_message,
    void* user_data
);


/* ═══════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Initialize the OTA fine-tuning module.
 *
 * @param config   Configuration (copied internally)
 * @return 0 on success, error code on failure
 */
int rac_ota_init(const rac_ota_config_t* config);

/**
 * Shutdown the OTA fine-tuning module and free resources.
 */
void rac_ota_shutdown(void);

/**
 * Get the current OTA status.
 */
rac_ota_status_t rac_ota_get_status(void);


/* ═══════════════════════════════════════════════════════════════════════════
 * Training Data Upload
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Upload training interactions to the cloud server.
 *
 * @param interactions   Array of interactions
 * @param count          Number of interactions
 * @param model_id       The model these interactions are for
 * @param callback        Completion callback (called on background thread)
 * @param progress_cb    Progress callback (optional, may be NULL)
 * @param user_data      Opaque pointer passed to callbacks
 * @return 0 on success (async operation started), error code on failure
 */
int rac_ota_upload_training_data(
    const rac_training_interaction_t* interactions,
    int count,
    const char* model_id,
    rac_upload_complete_callback_t callback,
    rac_ota_progress_callback_t progress_cb,
    void* user_data
);


/* ═══════════════════════════════════════════════════════════════════════════
 * Adapter Polling & Download
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Poll the cloud server for new adapters.
 * If auto_apply_adapter is enabled, automatically downloads and applies.
 *
 * @param model_id     Filter by model ID (optional, may be NULL for all)
 * @param callback     Called for each new adapter found
 * @param user_data    Opaque pointer
 * @return 0 on success (async operation started)
 */
int rac_ota_poll_adapters(
    const char* model_id,
    rac_adapter_available_callback_t callback,
    void* user_data
);

/**
 * Start automatic polling on the configured interval.
 * Calls the registered adapter_available callback when new adapters appear.
 */
int rac_ota_start_auto_poll(
    rac_adapter_available_callback_t callback,
    void* user_data
);

/**
 * Stop automatic polling.
 */
void rac_ota_stop_auto_poll(void);

/**
 * Download a specific adapter by ID.
 *
 * @param adapter_id   The adapter to download
 * @param callback     Called on completion
 * @param progress_cb  Progress callback (optional)
 * @param user_data    Opaque pointer
 * @return 0 on success (async download started)
 */
int rac_ota_download_adapter(
    const char* adapter_id,
    rac_adapter_downloaded_callback_t callback,
    rac_ota_progress_callback_t progress_cb,
    void* user_data
);


/* ═══════════════════════════════════════════════════════════════════════════
 * Adapter Management
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * List locally available adapters.
 *
 * @param out_adapters  Pointer to receive array (caller must free with rac_ota_free_adapters)
 * @param out_count     Number of adapters returned
 * @return 0 on success
 */
int rac_ota_list_local_adapters(
    rac_adapter_info_t** out_adapters,
    int* out_count
);

/**
 * Apply a downloaded adapter to the active LLM model.
 *
 * @param adapter_id  ID of the adapter to apply
 * @return 0 on success, error code on failure
 */
int rac_ota_apply_adapter(const char* adapter_id);

/**
 * Remove (unapply) the currently active adapter.
 *
 * @return 0 on success
 */
int rac_ota_remove_active_adapter(void);

/**
 * Get info about the currently active adapter.
 *
 * @return Pointer to adapter info (NULL if no adapter active).
 *         Caller must NOT free this pointer.
 */
const rac_adapter_info_t* rac_ota_get_active_adapter(void);

/**
 * Delete a locally stored adapter.
 *
 * @param adapter_id  ID of the adapter to delete
 * @return 0 on success
 */
int rac_ota_delete_local_adapter(const char* adapter_id);

/**
 * Free an adapter info array returned by rac_ota_list_local_adapters.
 */
void rac_ota_free_adapters(rac_adapter_info_t* adapters, int count);


#ifdef __cplusplus
}
#endif

#endif /* RAC_OTA_FINETUNE_H */
