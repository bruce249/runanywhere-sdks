package com.runanywhere.runanywhereai.data

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

// =============================================================================
// Personalization — Data Model & Persistence
// =============================================================================

/**
 * Tone presets for the AI's response style.
 */
enum class PersonalizationTone(val label: String, val emoji: String, val promptFragment: String) {
    FRIENDLY("Friendly", "😊", "Respond in a warm, friendly, and approachable tone."),
    PROFESSIONAL("Professional", "💼", "Respond in a professional, clear, and structured tone."),
    CASUAL("Casual", "✌️", "Respond in a relaxed, casual, and conversational tone."),
    TECHNICAL("Technical", "🔧", "Respond with technical precision, include relevant details and code examples when appropriate."),
    CREATIVE("Creative", "🎨", "Respond creatively with vivid language, analogies, and imaginative explanations."),
    CONCISE("Concise", "⚡", "Be extremely concise and to the point. Use short sentences and bullet points."),
}

/**
 * User's personalization profile — persisted as JSON.
 */
@Serializable
data class PersonalizationProfile(
    /** User's name (optional, used for personalized greetings) */
    val displayName: String = "",

    /** What should the AI know about the user? (free-text, multi-line) */
    val aboutUser: String = "",

    /** How should the AI respond? (free-text, multi-line) */
    val responseStyle: String = "",

    /** Selected tone preset */
    val tone: String = PersonalizationTone.FRIENDLY.name,

    /** Advanced: raw system prompt that overrides everything when non-empty */
    val customSystemPrompt: String = "",

    /** Master toggle — when false, no personalization is applied */
    val isEnabled: Boolean = true,
) {
    /** Resolved tone enum (safe default if stored value is invalid) */
    val resolvedTone: PersonalizationTone
        get() = PersonalizationTone.entries.firstOrNull { it.name == tone }
            ?: PersonalizationTone.FRIENDLY

    /** True if the user has entered any personalization content */
    val hasContent: Boolean
        get() = displayName.isNotBlank() ||
            aboutUser.isNotBlank() ||
            responseStyle.isNotBlank() ||
            customSystemPrompt.isNotBlank()
}

/**
 * PersonalizationStore — Manages persistence and composition of
 * the user's personalization profile.
 *
 * Usage:
 * ```
 * val store = PersonalizationStore.getInstance(context)
 *
 * // Read current profile reactively
 * store.profile.collect { profile -> ... }
 *
 * // Build the system prompt for LLM calls
 * val systemPrompt: String? = store.buildSystemPrompt()
 *
 * // Update fields
 * store.updateAboutUser("I'm a Kotlin developer")
 * store.updateTone(PersonalizationTone.TECHNICAL)
 * ```
 */
class PersonalizationStore private constructor(context: Context) {

    companion object {
        private const val TAG = "PersonalizationStore"
        private const val PREFS_NAME = "runanywhere_personalization"
        private const val KEY_PROFILE = "profile_json"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: PersonalizationStore? = null

        fun getInstance(context: Context): PersonalizationStore {
            return instance ?: synchronized(this) {
                instance ?: PersonalizationStore(context.applicationContext).also { instance = it }
            }
        }
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val _profile = MutableStateFlow(PersonalizationProfile())
    val profile: StateFlow<PersonalizationProfile> = _profile.asStateFlow()

    init {
        loadProfile()
    }

    // ========================================================================
    // System Prompt Composition
    // ========================================================================

    /**
     * Build the final system prompt from the profile.
     *
     * Returns `null` when personalization is disabled or empty,
     * so callers can skip injecting a system prompt entirely.
     *
     * Priority:
     * 1. If `customSystemPrompt` is set → use it verbatim
     * 2. Otherwise → compose from aboutUser + responseStyle + tone
     */
    fun buildSystemPrompt(): String? {
        val p = _profile.value
        if (!p.isEnabled || !p.hasContent) return null

        // Custom prompt takes full priority
        if (p.customSystemPrompt.isNotBlank()) {
            return p.customSystemPrompt.trim()
        }

        // Compose from structured fields
        val parts = mutableListOf<String>()

        // Base identity
        parts.add("You are a helpful AI assistant.")

        // User context
        if (p.displayName.isNotBlank()) {
            parts.add("The user's name is ${p.displayName.trim()}.")
        }
        if (p.aboutUser.isNotBlank()) {
            parts.add("")
            parts.add("About the user:")
            parts.add(p.aboutUser.trim())
        }

        // Response style
        if (p.responseStyle.isNotBlank()) {
            parts.add("")
            parts.add("Response style instructions:")
            parts.add(p.responseStyle.trim())
        }

        // Tone
        parts.add("")
        parts.add(p.resolvedTone.promptFragment)

        return parts.joinToString("\n")
    }

    // ========================================================================
    // Update Methods
    // ========================================================================

    fun updateDisplayName(value: String) {
        updateProfile { it.copy(displayName = value) }
    }

    fun updateAboutUser(value: String) {
        updateProfile { it.copy(aboutUser = value) }
    }

    fun updateResponseStyle(value: String) {
        updateProfile { it.copy(responseStyle = value) }
    }

    fun updateTone(tone: PersonalizationTone) {
        updateProfile { it.copy(tone = tone.name) }
    }

    fun updateCustomSystemPrompt(value: String) {
        updateProfile { it.copy(customSystemPrompt = value) }
    }

    fun updateEnabled(enabled: Boolean) {
        updateProfile { it.copy(isEnabled = enabled) }
    }

    fun resetToDefaults() {
        _profile.value = PersonalizationProfile()
        persistProfile()
        Log.i(TAG, "Profile reset to defaults")
    }

    // ========================================================================
    // Private
    // ========================================================================

    private inline fun updateProfile(transform: (PersonalizationProfile) -> PersonalizationProfile) {
        _profile.value = transform(_profile.value)
        persistProfile()
    }

    private fun persistProfile() {
        try {
            val encoded = json.encodeToString(_profile.value)
            prefs.edit().putString(KEY_PROFILE, encoded).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to persist profile: ${e.message}", e)
        }
    }

    private fun loadProfile() {
        try {
            val stored = prefs.getString(KEY_PROFILE, null)
            if (stored != null) {
                _profile.value = json.decodeFromString<PersonalizationProfile>(stored)
                Log.i(TAG, "Loaded personalization profile (enabled=${_profile.value.isEnabled})")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load profile, using defaults: ${e.message}", e)
            _profile.value = PersonalizationProfile()
        }
    }
}
