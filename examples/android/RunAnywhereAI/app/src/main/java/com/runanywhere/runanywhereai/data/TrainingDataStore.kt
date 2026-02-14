package com.runanywhere.runanywhereai.data

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import com.runanywhere.runanywhereai.domain.models.ContextMessage
import com.runanywhere.runanywhereai.domain.models.TrainingDataStats
import com.runanywhere.runanywhereai.domain.models.TrainingInteraction
import com.runanywhere.runanywhereai.domain.models.TrainingSample
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

/**
 * TrainingDataStore - Collects, persists, and manages user interaction data
 * for on-device fine-tuning.
 *
 * Responsibilities:
 * - Captures user interactions automatically during chat
 * - Stores user corrections and ratings
 * - Detects frequently asked queries and patterns
 * - Prepares training samples from raw interactions
 * - Provides statistics about collected data
 *
 * Data is stored as JSON files in {filesDir}/TrainingData/
 */
class TrainingDataStore private constructor(context: Context) {
    companion object {
        private const val TAG = "TrainingDataStore"
        private const val TRAINING_DATA_DIR = "TrainingData"
        private const val INTERACTIONS_DIR = "interactions"
        private const val SAMPLES_DIR = "samples"
        private const val QUERY_FREQUENCY_FILE = "query_frequencies.json"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: TrainingDataStore? = null

        fun getInstance(context: Context): TrainingDataStore {
            return instance ?: synchronized(this) {
                instance ?: TrainingDataStore(context.applicationContext).also { instance = it }
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val baseDirectory: File = File(appContext.filesDir, TRAINING_DATA_DIR)
    private val interactionsDirectory: File = File(baseDirectory, INTERACTIONS_DIR)
    private val samplesDirectory: File = File(baseDirectory, SAMPLES_DIR)
    private val queryFrequencyFile: File = File(baseDirectory, QUERY_FREQUENCY_FILE)

    // In-memory cache of interactions
    private val _interactions = MutableStateFlow<List<TrainingInteraction>>(emptyList())
    val interactions: StateFlow<List<TrainingInteraction>> = _interactions.asStateFlow()

    private val _stats = MutableStateFlow(TrainingDataStats())
    val stats: StateFlow<TrainingDataStats> = _stats.asStateFlow()

    // Query frequency tracking
    private val queryFrequencies = mutableMapOf<String, Int>()

    init {
        baseDirectory.mkdirs()
        interactionsDirectory.mkdirs()
        samplesDirectory.mkdirs()
        loadInteractions()
        loadQueryFrequencies()
        updateStats()
    }

    // ========================================================================
    // Public API - Data Collection
    // ========================================================================

    /**
     * Record a user interaction from a chat session.
     * Called automatically after each model response.
     */
    fun recordInteraction(
        userPrompt: String,
        modelResponse: String,
        modelId: String? = null,
        modelName: String? = null,
        conversationContext: List<ContextMessage>? = null,
        dataSource: String = "text_chat",
    ): TrainingInteraction {
        // Normalize the query for frequency tracking
        val normalizedQuery = normalizeQuery(userPrompt)
        val frequency = incrementQueryFrequency(normalizedQuery)

        // Auto-detect category from the prompt
        val category = detectCategory(userPrompt)

        val interaction = TrainingInteraction(
            userPrompt = userPrompt,
            modelResponse = modelResponse,
            modelId = modelId,
            modelName = modelName,
            conversationContext = conversationContext,
            category = category,
            frequencyCount = frequency,
            // Include all interactions by default so they appear in training immediately
            includeInTraining = true,
            dataSource = dataSource,
        )

        saveInteraction(interaction)
        val updated = _interactions.value.toMutableList()
        updated.add(0, interaction)
        _interactions.value = updated
        updateStats()

        Log.d(TAG, "Recorded interaction: ${userPrompt.take(50)}... (freq: $frequency, cat: $category)")
        return interaction
    }

    /**
     * Add user feedback to an existing interaction.
     */
    fun addFeedback(
        interactionId: String,
        rating: Int? = null,
        correctedResponse: String? = null,
        isPositiveExample: Boolean? = null,
        tags: List<String>? = null,
    ) {
        val index = _interactions.value.indexOfFirst { it.id == interactionId }
        if (index == -1) {
            Log.w(TAG, "Interaction not found: $interactionId")
            return
        }

        val interaction = _interactions.value[index]
        val updated = interaction.copy(
            rating = rating ?: interaction.rating,
            correctedResponse = correctedResponse ?: interaction.correctedResponse,
            isPositiveExample = isPositiveExample ?: interaction.isPositiveExample,
            tags = tags ?: interaction.tags,
            // Always include rated/corrected interactions in training
            includeInTraining = true,
        )

        val list = _interactions.value.toMutableList()
        list[index] = updated
        _interactions.value = list

        saveInteraction(updated)
        updateStats()

        Log.d(TAG, "Updated feedback for interaction: $interactionId")
    }

    /**
     * Toggle whether an interaction is included in training.
     */
    fun toggleTrainingInclusion(interactionId: String) {
        val index = _interactions.value.indexOfFirst { it.id == interactionId }
        if (index == -1) return

        val interaction = _interactions.value[index]
        val updated = interaction.copy(includeInTraining = !interaction.includeInTraining)

        val list = _interactions.value.toMutableList()
        list[index] = updated
        _interactions.value = list

        saveInteraction(updated)
        updateStats()
    }

    /**
     * Delete an interaction.
     */
    fun deleteInteraction(interactionId: String) {
        _interactions.value = _interactions.value.filter { it.id != interactionId }
        val file = File(interactionsDirectory, "$interactionId.json")
        if (file.exists()) file.delete()
        updateStats()
    }

    /**
     * Clear all training data.
     */
    fun clearAllData() {
        _interactions.value = emptyList()
        queryFrequencies.clear()
        interactionsDirectory.listFiles()?.forEach { it.delete() }
        samplesDirectory.listFiles()?.forEach { it.delete() }
        if (queryFrequencyFile.exists()) queryFrequencyFile.delete()
        updateStats()
        Log.i(TAG, "All training data cleared")
    }

    // ========================================================================
    // Public API - Training Sample Preparation
    // ========================================================================

    /**
     * Convert collected interactions into training samples.
     * Applies filtering, weighting, and formatting.
     */
    fun prepareTrainingSamples(): List<TrainingSample> {
        val eligibleInteractions = _interactions.value.filter { it.includeInTraining }

        return eligibleInteractions.map { interaction ->
            // Use corrected response if available, otherwise use original
            val targetResponse = interaction.correctedResponse ?: interaction.modelResponse

            // Calculate weight based on quality signals
            val weight = calculateSampleWeight(interaction)

            // Build instruction from context
            val instruction = buildInstruction(interaction)

            TrainingSample(
                instruction = instruction,
                input = interaction.userPrompt,
                output = targetResponse,
                weight = weight,
                sourceInteractionId = interaction.id,
            )
        }.also { samples ->
            // Save prepared samples
            savePreparedSamples(samples)
            Log.i(TAG, "Prepared ${samples.size} training samples from ${eligibleInteractions.size} interactions")
        }
    }

    /**
     * Get interactions filtered by various criteria.
     */
    fun getFilteredInteractions(
        category: String? = null,
        minRating: Int? = null,
        onlyWithCorrections: Boolean = false,
        onlyFrequent: Boolean = false,
        onlyIncludedInTraining: Boolean = false,
    ): List<TrainingInteraction> {
        return _interactions.value.filter { interaction ->
            (category == null || interaction.category == category) &&
                (minRating == null || (interaction.rating ?: 0) >= minRating) &&
                (!onlyWithCorrections || interaction.correctedResponse != null) &&
                (!onlyFrequent || interaction.frequencyCount >= 3) &&
                (!onlyIncludedInTraining || interaction.includeInTraining)
        }
    }

    /**
     * Get the top N most frequently asked queries.
     */
    fun getTopQueries(limit: Int = 10): List<Pair<String, Int>> {
        return queryFrequencies.entries
            .sortedByDescending { it.value }
            .take(limit)
            .map { it.key to it.value }
    }

    /**
     * Get interactions most relevant to a prompt for adapter-guided inference.
     */
    fun getRelevantInteractions(
        prompt: String,
        modelId: String? = null,
        limit: Int = 3,
    ): List<TrainingInteraction> {
        val promptWords = prompt.lowercase()
            .split(Regex("\\W+"))
            .filter { it.length >= 3 }
            .toSet()
        val promptCategory = detectCategory(prompt)

        return _interactions.value
            .asSequence()
            .filter { it.includeInTraining }
            .map { interaction ->
                val interactionWords = interaction.userPrompt.lowercase()
                    .split(Regex("\\W+"))
                    .filter { it.length >= 3 }
                    .toSet()
                val overlap = promptWords.intersect(interactionWords).size
                val categoryBonus = if (interaction.category == promptCategory) 3 else 0
                val modelBonus = if (modelId != null && interaction.modelId == modelId) 2 else 0
                val qualityBonus = (interaction.rating ?: 3) + if (interaction.isPositiveExample) 2 else 0
                val recencyBonus = if ((System.currentTimeMillis() - interaction.timestamp) < 7L * 24 * 60 * 60 * 1000) 1 else 0

                interaction to (overlap * 4 + categoryBonus + modelBonus + qualityBonus + recencyBonus)
            }
            .filter { it.second > 0 }
            .sortedWith(compareByDescending<Pair<TrainingInteraction, Int>> { it.second }
                .thenByDescending { it.first.timestamp })
            .take(limit)
            .map { it.first }
            .toList()
    }

    // ========================================================================
    // Private - Persistence
    // ========================================================================

    private fun saveInteraction(interaction: TrainingInteraction) {
        try {
            val file = File(interactionsDirectory, "${interaction.id}.json")
            file.writeText(json.encodeToString(interaction))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save interaction: ${e.message}", e)
        }
    }

    private fun loadInteractions() {
        try {
            val files = interactionsDirectory.listFiles { f -> f.extension == "json" } ?: emptyArray()
            val loaded = files.mapNotNull { file ->
                try {
                    json.decodeFromString<TrainingInteraction>(file.readText())
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load interaction: ${file.name}", e)
                    null
                }
            }
            _interactions.value = loaded.sortedByDescending { it.timestamp }
            Log.i(TAG, "Loaded ${loaded.size} interactions from disk")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load interactions", e)
        }
    }

    private fun savePreparedSamples(samples: List<TrainingSample>) {
        try {
            // Save as JSONL (one JSON object per line) - standard training format
            val file = File(samplesDirectory, "training_samples.jsonl")
            file.writeText(samples.joinToString("\n") { json.encodeToString(it) })

            // Also save as a single JSON array for easier inspection
            val jsonFile = File(samplesDirectory, "training_samples.json")
            jsonFile.writeText(json.encodeToString(samples))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save prepared samples: ${e.message}", e)
        }
    }

    private fun loadQueryFrequencies() {
        try {
            if (queryFrequencyFile.exists()) {
                val data = json.decodeFromString<Map<String, Int>>(queryFrequencyFile.readText())
                queryFrequencies.putAll(data)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load query frequencies", e)
        }
    }

    private fun saveQueryFrequencies() {
        try {
            queryFrequencyFile.writeText(json.encodeToString(queryFrequencies as Map<String, Int>))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save query frequencies", e)
        }
    }

    // ========================================================================
    // Private - Data Processing
    // ========================================================================

    /**
     * Normalize a query for frequency counting.
     * Lowercases, trims, removes extra whitespace.
     */
    private fun normalizeQuery(query: String): String {
        return query.trim().lowercase()
            .replace(Regex("\\s+"), " ")
            .take(200) // Limit length for grouping
    }

    /**
     * Increment query frequency counter and persist.
     */
    private fun incrementQueryFrequency(normalizedQuery: String): Int {
        val count = (queryFrequencies[normalizedQuery] ?: 0) + 1
        queryFrequencies[normalizedQuery] = count
        saveQueryFrequencies()

        // Also update matching interactions' frequency counts
        val updated = _interactions.value.map { interaction ->
            if (normalizeQuery(interaction.userPrompt) == normalizedQuery) {
                interaction.copy(frequencyCount = count)
            } else {
                interaction
            }
        }
        _interactions.value = updated

        return count
    }

    /**
     * Auto-detect the category of a user prompt.
     */
    private fun detectCategory(prompt: String): String {
        val lower = prompt.lowercase()
        return when {
            lower.containsAny("code", "function", "class", "program", "debug", "error", "bug", "compile") -> "coding"
            lower.containsAny("write", "essay", "story", "poem", "letter", "email", "article") -> "writing"
            lower.containsAny("explain", "what is", "how does", "why", "define", "meaning") -> "explanation"
            lower.containsAny("translate", "language", "spanish", "french", "german", "chinese") -> "translation"
            lower.containsAny("math", "calculate", "equation", "formula", "solve", "algebra") -> "math"
            lower.containsAny("summarize", "summary", "tldr", "key points", "overview") -> "summarization"
            lower.containsAny("recipe", "cook", "food", "meal", "ingredient") -> "cooking"
            lower.containsAny("health", "medical", "symptom", "exercise", "diet", "fitness") -> "health"
            lower.containsAny("create", "generate", "make", "build", "design") -> "creative"
            lower.containsAny("analyze", "compare", "contrast", "evaluate", "review") -> "analysis"
            else -> "general"
        }
    }

    /**
     * Calculate the training weight for a sample based on quality signals.
     */
    private fun calculateSampleWeight(interaction: TrainingInteraction): Float {
        var weight = 1.0f

        // Corrected responses are highest quality
        if (interaction.correctedResponse != null) weight += 2.0f

        // Highly rated responses are valuable
        when (interaction.rating) {
            5 -> weight += 1.5f
            4 -> weight += 1.0f
            3 -> weight += 0.5f
            2 -> weight -= 0.5f
            1 -> weight -= 1.0f
        }

        // Explicitly marked positive examples
        if (interaction.isPositiveExample) weight += 1.0f

        // Frequently asked queries deserve more attention
        if (interaction.frequencyCount >= 5) weight += 1.0f
        else if (interaction.frequencyCount >= 3) weight += 0.5f

        return weight.coerceIn(0.1f, 5.0f)
    }

    /**
     * Build a training instruction from interaction context.
     */
    private fun buildInstruction(interaction: TrainingInteraction): String {
        val base = "You are a helpful AI assistant."
        val categoryHint = when (interaction.category) {
            "coding" -> " You are skilled at programming and technical explanations."
            "writing" -> " You are a creative and articulate writer."
            "explanation" -> " You provide clear and thorough explanations."
            "translation" -> " You are a skilled multilingual translator."
            "math" -> " You are excellent at mathematics and logical reasoning."
            "summarization" -> " You create concise and accurate summaries."
            else -> ""
        }
        return base + categoryHint
    }

    /**
     * Update aggregate statistics.
     */
    private fun updateStats() {
        val all = _interactions.value
        val rated = all.filter { it.rating != null }
        val topQueries = getTopQueries(5).map { it.first }

        _stats.value = TrainingDataStats(
            totalInteractions = all.size,
            ratedInteractions = rated.size,
            correctedInteractions = all.count { it.correctedResponse != null },
            positiveExamples = all.count { it.isPositiveExample },
            readyForTraining = all.count { it.includeInTraining },
            categories = all.groupBy { it.category ?: "uncategorized" }.mapValues { it.value.size },
            averageRating = if (rated.isNotEmpty()) rated.mapNotNull { it.rating }.average().toFloat() else 0f,
            oldestInteraction = all.minByOrNull { it.timestamp }?.timestamp,
            newestInteraction = all.maxByOrNull { it.timestamp }?.timestamp,
            topQueries = topQueries,
            dataSourceCounts = all.groupBy { it.dataSource }.mapValues { it.value.size },
        )
    }
}

/** Extension to check if string contains any of the given substrings. */
private fun String.containsAny(vararg terms: String): Boolean =
    terms.any { this.contains(it) }
