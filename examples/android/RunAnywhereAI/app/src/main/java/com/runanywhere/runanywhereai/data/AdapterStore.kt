package com.runanywhere.runanywhereai.data

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import com.runanywhere.runanywhereai.domain.models.LoraAdapter
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

/**
 * AdapterStore - Manages the lifecycle and persistence of trained LoRA adapters.
 *
 * Responsibilities:
 * - Persists adapter metadata as JSON files
 * - Tracks which adapter is currently active
 * - Provides adapter listing, loading, and deletion
 * - Manages adapter versioning (when re-training on more data)
 *
 * Adapter metadata is stored in {filesDir}/LoraAdapters/meta/
 * Adapter weights are stored in {filesDir}/LoraAdapters/ (managed by FineTuningEngine)
 */
class AdapterStore private constructor(context: Context) {
    companion object {
        private const val TAG = "AdapterStore"
        private const val ADAPTERS_DIR = "LoraAdapters"
        private const val META_DIR = "meta"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: AdapterStore? = null

        fun getInstance(context: Context): AdapterStore {
            return instance ?: synchronized(this) {
                instance ?: AdapterStore(context.applicationContext).also { instance = it }
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val metaDirectory = File(File(appContext.filesDir, ADAPTERS_DIR), META_DIR)

    private val _adapters = MutableStateFlow<List<LoraAdapter>>(emptyList())
    val adapters: StateFlow<List<LoraAdapter>> = _adapters.asStateFlow()

    private val _activeAdapter = MutableStateFlow<LoraAdapter?>(null)
    val activeAdapter: StateFlow<LoraAdapter?> = _activeAdapter.asStateFlow()

    init {
        metaDirectory.mkdirs()
        loadAdapters()
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /**
     * Save a newly trained adapter's metadata.
     */
    fun saveAdapter(adapter: LoraAdapter) {
        val existing = _adapters.value.toMutableList()
        val existingIndex = existing.indexOfFirst { it.id == adapter.id }

        if (existingIndex >= 0) {
            existing[existingIndex] = adapter
        } else {
            existing.add(0, adapter)
        }

        _adapters.value = existing
        persistAdapter(adapter)
        Log.i(TAG, "Saved adapter: ${adapter.name} (${adapter.id})")
    }

    /**
     * Activate an adapter for use during inference.
     * Only one adapter can be active at a time.
     */
    fun activateAdapter(adapterId: String): Boolean {
        val adapter = _adapters.value.find { it.id == adapterId }
        if (adapter == null) {
            Log.w(TAG, "Adapter not found: $adapterId")
            return false
        }

        // Check adapter file exists
        if (adapter.adapterFilePath != null && !File(adapter.adapterFilePath).exists()) {
            Log.e(TAG, "Adapter file missing: ${adapter.adapterFilePath}")
            return false
        }

        // Deactivate current adapter
        _activeAdapter.value?.let { current ->
            val updated = current.copy(isActive = false)
            updateAdapter(updated)
        }

        // Activate new adapter
        val activated = adapter.copy(
            isActive = true,
            lastUsedAt = System.currentTimeMillis(),
        )
        updateAdapter(activated)
        _activeAdapter.value = activated

        Log.i(TAG, "Activated adapter: ${adapter.name}")
        return true
    }

    /**
     * Deactivate the currently active adapter.
     */
    fun deactivateAdapter() {
        _activeAdapter.value?.let { current ->
            val updated = current.copy(isActive = false)
            updateAdapter(updated)
        }
        _activeAdapter.value = null
        Log.i(TAG, "All adapters deactivated")
    }

    /**
     * Delete an adapter (metadata + weights file).
     */
    fun deleteAdapter(adapterId: String) {
        val adapter = _adapters.value.find { it.id == adapterId }

        // If this is the active adapter, deactivate first
        if (_activeAdapter.value?.id == adapterId) {
            _activeAdapter.value = null
        }

        // Remove from list
        _adapters.value = _adapters.value.filter { it.id != adapterId }

        // Delete metadata file
        val metaFile = File(metaDirectory, "$adapterId.json")
        if (metaFile.exists()) metaFile.delete()

        // Delete weights file
        adapter?.adapterFilePath?.let { path ->
            val weightsFile = File(path)
            if (weightsFile.exists()) weightsFile.delete()
        }

        Log.i(TAG, "Deleted adapter: $adapterId")
    }

    /**
     * Get an adapter by ID.
     */
    fun getAdapter(adapterId: String): LoraAdapter? {
        return _adapters.value.find { it.id == adapterId }
    }

    /**
     * Get all adapters for a specific base model.
     */
    fun getAdaptersForModel(modelId: String): List<LoraAdapter> {
        return _adapters.value.filter { it.baseModelId == modelId }
    }

    /**
     * Get total storage used by all adapters (metadata + weights).
     */
    fun getTotalStorageBytes(): Long {
        return _adapters.value.sumOf { it.adapterSizeBytes }
    }

    /**
     * Increment adapter version (for re-training with more data).
     */
    fun createNewVersion(adapter: LoraAdapter): LoraAdapter {
        return adapter.copy(
            version = adapter.version + 1,
            createdAt = System.currentTimeMillis(),
        )
    }

    // ========================================================================
    // Private
    // ========================================================================

    private fun updateAdapter(adapter: LoraAdapter) {
        val list = _adapters.value.toMutableList()
        val index = list.indexOfFirst { it.id == adapter.id }
        if (index >= 0) {
            list[index] = adapter
            _adapters.value = list
            persistAdapter(adapter)
        }
    }

    private fun persistAdapter(adapter: LoraAdapter) {
        try {
            val file = File(metaDirectory, "${adapter.id}.json")
            file.writeText(json.encodeToString(adapter))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to persist adapter: ${e.message}", e)
        }
    }

    private fun loadAdapters() {
        try {
            val files = metaDirectory.listFiles { f -> f.extension == "json" } ?: emptyArray()
            val loaded = files.mapNotNull { file ->
                try {
                    json.decodeFromString<LoraAdapter>(file.readText())
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load adapter: ${file.name}", e)
                    null
                }
            }
            _adapters.value = loaded.sortedByDescending { it.createdAt }

            // Restore active adapter
            val active = loaded.find { it.isActive }
            _activeAdapter.value = active

            Log.i(TAG, "Loaded ${loaded.size} adapters, active: ${active?.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load adapters", e)
        }
    }
}
