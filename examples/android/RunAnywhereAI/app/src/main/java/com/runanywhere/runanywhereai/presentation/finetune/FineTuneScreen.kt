package com.runanywhere.runanywhereai.presentation.finetune

import android.text.format.DateUtils
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.DataUsage
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Extension
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.StarBorder
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.ThumbUp
import androidx.compose.material.icons.filled.ThumbUpOffAlt
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material.icons.filled.TrendingDown
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.material.icons.outlined.Cloud
import androidx.compose.material.icons.outlined.DataUsage
import androidx.compose.material.icons.outlined.Extension
import androidx.compose.material.icons.outlined.ModelTraining
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.runanywhere.runanywhereai.domain.models.FineTuningConfig
import com.runanywhere.runanywhereai.domain.models.FineTuningStatus
import com.runanywhere.runanywhereai.domain.models.LoraAdapter
import com.runanywhere.runanywhereai.domain.models.TrainingInteraction
import com.runanywhere.runanywhereai.domain.models.TrainingMetrics
import com.runanywhere.runanywhereai.ui.theme.AppColors

// =============================================================================
// Fine-Tuning Screen - Main Composable
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FineTuneScreen(viewModel: FineTuneViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize()) {
        // Header
        FineTuneHeader(uiState)

        // Tab Navigation
        FineTuneTabBar(
            selectedTab = uiState.selectedTab,
            onTabSelected = viewModel::selectTab,
            isTraining = uiState.isTraining,
        )

        // Content based on selected tab
        when (uiState.selectedTab) {
            FineTuneTab.OVERVIEW -> OverviewTab(uiState, viewModel)
            FineTuneTab.DATA -> DataTab(uiState, viewModel)
            FineTuneTab.TRAIN -> TrainTab(uiState, viewModel)
            FineTuneTab.ADAPTERS -> AdaptersTab(uiState, viewModel)
            FineTuneTab.CLOUD -> {
                val cloudState by viewModel.cloudUiState.collectAsStateWithLifecycle()
                CloudFineTuneTab(
                    cloudState = cloudState,
                    onServerUrlChanged = viewModel::updateCloudServerUrl,
                    onSaveConfig = viewModel::saveCloudConfig,
                    onCheckConnection = viewModel::checkCloudConnection,
                    onUploadData = viewModel::uploadTrainingDataToCloud,
                    onStartTraining = viewModel::startCloudTraining,
                    onPollStatus = viewModel::pollCloudJobStatus,
                    onPollAdapters = viewModel::pollCloudAdapters,
                    onDownloadAdapter = viewModel::downloadCloudAdapter,
                    onToggleAutoSync = viewModel::toggleAutoSync,
                    onClearError = viewModel::clearCloudError,
                )
            }
        }
    }

    // Config bottom sheet
    if (uiState.showConfigSheet) {
        ConfigBottomSheet(
            config = uiState.config,
            onDismiss = { viewModel.showConfigSheet(false) },
            onSave = { config ->
                viewModel.updateConfig(config)
                viewModel.showConfigSheet(false)
            },
        )
    }

    // Delete confirmation dialog
    uiState.showDeleteConfirm?.let { adapterId ->
        AlertDialog(
            onDismissRequest = { viewModel.confirmDeleteAdapter(null) },
            title = { Text("Delete Adapter") },
            text = { Text("This will permanently delete this adapter and its weights. Continue?") },
            confirmButton = {
                TextButton(
                    onClick = { viewModel.deleteAdapter(adapterId) },
                    colors = ButtonDefaults.textButtonColors(contentColor = AppColors.primaryRed),
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { viewModel.confirmDeleteAdapter(null) }) { Text("Cancel") }
            },
        )
    }

    // Error snackbar
    uiState.errorMessage?.let { error ->
        AlertDialog(
            onDismissRequest = viewModel::clearError,
            title = { Text("Error") },
            text = { Text(error) },
            confirmButton = {
                TextButton(onClick = viewModel::clearError) { Text("OK") }
            },
        )
    }
}

// =============================================================================
// Header
// =============================================================================

@Composable
private fun FineTuneHeader(uiState: FineTuneUiState) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = Icons.Default.Psychology,
            contentDescription = null,
            tint = AppColors.primaryPurple,
            modifier = Modifier.size(28.dp),
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Fine-Tuning Studio",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
            if (uiState.currentModelName != null) {
                Text(
                    text = "Model: ${uiState.currentModelName}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        if (uiState.activeAdapter != null) {
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = AppColors.primaryGreen.copy(alpha = 0.15f),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.Extension,
                        contentDescription = null,
                        tint = AppColors.primaryGreen,
                        modifier = Modifier.size(14.dp),
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Adapter Active",
                        style = MaterialTheme.typography.labelSmall,
                        color = AppColors.primaryGreen,
                    )
                }
            }
        }
    }
}

// =============================================================================
// Tab Navigation
// =============================================================================

@Composable
private fun FineTuneTabBar(
    selectedTab: FineTuneTab,
    onTabSelected: (FineTuneTab) -> Unit,
    isTraining: Boolean,
) {
    val tabs = listOf(
        Triple(FineTuneTab.OVERVIEW, Icons.Outlined.AutoFixHigh, Icons.Filled.AutoFixHigh),
        Triple(FineTuneTab.DATA, Icons.Outlined.DataUsage, Icons.Filled.DataUsage),
        Triple(FineTuneTab.TRAIN, Icons.Outlined.ModelTraining, Icons.Filled.ModelTraining),
        Triple(FineTuneTab.ADAPTERS, Icons.Outlined.Extension, Icons.Filled.Extension),
        Triple(FineTuneTab.CLOUD, Icons.Outlined.Cloud, Icons.Filled.Cloud),
    )

    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 0.dp,
    ) {
        tabs.forEach { (tab, icon, selectedIcon) ->
            val isSelected = selectedTab == tab
            NavigationBarItem(
                icon = {
                    if (tab == FineTuneTab.TRAIN && isTraining) {
                        // Show pulsing indicator when training
                        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                        val alpha by infiniteTransition.animateFloat(
                            initialValue = 0.4f,
                            targetValue = 1f,
                            animationSpec = infiniteRepeatable(tween(800), RepeatMode.Reverse),
                            label = "pulse",
                        )
                        Icon(
                            selectedIcon,
                            contentDescription = tab.label,
                            tint = AppColors.primaryOrange.copy(alpha = alpha),
                        )
                    } else {
                        Icon(
                            if (isSelected) selectedIcon else icon,
                            contentDescription = tab.label,
                        )
                    }
                },
                label = { Text(tab.label) },
                selected = isSelected,
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = AppColors.primaryPurple,
                    selectedTextColor = AppColors.primaryPurple,
                    indicatorColor = AppColors.primaryPurple.copy(alpha = 0.12f),
                ),
                onClick = { onTabSelected(tab) },
            )
        }
    }
}

// =============================================================================
// Overview Tab
// =============================================================================

@Composable
private fun OverviewTab(uiState: FineTuneUiState, viewModel: FineTuneViewModel) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // Training Pipeline Status Card
        PipelineStatusCard(uiState)

        // Quick Stats
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            StatCard(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.DataUsage,
                label = "Samples",
                value = "${uiState.dataStats.readyForTraining}",
                subtitle = "ready for training",
                color = AppColors.primaryBlue,
            )
            StatCard(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.Extension,
                label = "Adapters",
                value = "${uiState.adapters.size}",
                subtitle = if (uiState.activeAdapter != null) "1 active" else "none active",
                color = AppColors.primaryPurple,
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            StatCard(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.Star,
                label = "Avg Rating",
                value = if (uiState.dataStats.averageRating > 0) {
                    "${"%.1f".format(uiState.dataStats.averageRating)}"
                } else "—",
                subtitle = "${uiState.dataStats.ratedInteractions} rated",
                color = AppColors.primaryYellow,
            )
            StatCard(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.ThumbUp,
                label = "Corrected",
                value = "${uiState.dataStats.correctedInteractions}",
                subtitle = "improved samples",
                color = AppColors.primaryGreen,
            )
        }

        // How It Works
        HowItWorksCard()

        // Category Distribution
        if (uiState.dataStats.categories.isNotEmpty()) {
            CategoryDistributionCard(uiState.dataStats.categories)
        }
    }
}

@Composable
private fun PipelineStatusCard(uiState: FineTuneUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Training Pipeline",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(16.dp))

            val steps = listOf(
                PipelineStep(
                    "Collect Data",
                    "Chat with the model, rate responses",
                    uiState.dataStats.totalInteractions > 0,
                ),
                PipelineStep(
                    "Prepare Samples",
                    "${uiState.dataStats.readyForTraining} samples ready",
                    uiState.dataStats.readyForTraining >= uiState.config.minTrainingSamples,
                ),
                PipelineStep(
                    "Train Adapter",
                    when (uiState.trainingStatus) {
                        is FineTuningStatus.Training -> "Training in progress..."
                        is FineTuningStatus.Completed -> "Training complete!"
                        else -> "Configure and start"
                    },
                    uiState.trainingStatus is FineTuningStatus.Completed || uiState.adapters.isNotEmpty(),
                ),
                PipelineStep(
                    "Apply Adapter",
                    if (uiState.activeAdapter != null) "\"${uiState.activeAdapter.name}\" active" else "Load a trained adapter",
                    uiState.activeAdapter != null,
                ),
            )

            steps.forEachIndexed { index, step ->
                PipelineStepRow(step = step, isLast = index == steps.lastIndex)
            }
        }
    }
}

private data class PipelineStep(val title: String, val subtitle: String, val isComplete: Boolean)

@Composable
private fun PipelineStepRow(step: PipelineStep, isLast: Boolean) {
    Row(
        verticalAlignment = Alignment.Top,
        modifier = Modifier.padding(vertical = 4.dp),
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                imageVector = if (step.isComplete) Icons.Default.CheckCircle else Icons.Default.Info,
                contentDescription = null,
                tint = if (step.isComplete) AppColors.primaryGreen else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(24.dp),
            )
            if (!isLast) {
                Box(
                    modifier = Modifier
                        .width(2.dp)
                        .height(28.dp)
                        .background(
                            if (step.isComplete) AppColors.primaryGreen.copy(alpha = 0.5f)
                            else MaterialTheme.colorScheme.outlineVariant,
                        ),
                )
            }
        }
        Spacer(modifier = Modifier.width(12.dp))
        Column {
            Text(
                text = step.title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = if (step.isComplete) AppColors.primaryGreen else MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = step.subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun StatCard(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    label: String,
    value: String,
    subtitle: String,
    color: Color,
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.08f),
        ),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text(label, style = MaterialTheme.typography.labelMedium, color = color)
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun HowItWorksCard() {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "How On-Device Fine-Tuning Works",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(12.dp))

            val steps = listOf(
                "Use the app normally — chat, ask questions, explore",
                "Rate responses or provide corrections for better training data",
                "When enough data is collected, start LoRA adapter training",
                "The adapter learns your preferences and style on-device",
                "Activate the adapter to personalize future responses",
                "All data stays on your device — fully private",
            )

            steps.forEachIndexed { index, step ->
                Row(
                    modifier = Modifier.padding(vertical = 3.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Surface(
                        shape = CircleShape,
                        color = AppColors.primaryPurple.copy(alpha = 0.15f),
                        modifier = Modifier.size(22.dp),
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Text(
                                text = "${index + 1}",
                                style = MaterialTheme.typography.labelSmall,
                                color = AppColors.primaryPurple,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = step,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun CategoryDistributionCard(categories: Map<String, Int>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Query Categories",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                categories.entries.sortedByDescending { it.value }.forEach { (category, count) ->
                    AssistChip(
                        onClick = {},
                        label = { Text("$category ($count)") },
                    )
                }
            }
        }
    }
}

// =============================================================================
// Data Tab
// =============================================================================

@Composable
private fun DataTab(uiState: FineTuneUiState, viewModel: FineTuneViewModel) {
    Column(modifier = Modifier.fillMaxSize()) {
        // Data controls bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "${uiState.dataStats.totalInteractions} interactions",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            if (uiState.recentInteractions.isNotEmpty()) {
                TextButton(
                    onClick = viewModel::clearAllData,
                    colors = ButtonDefaults.textButtonColors(contentColor = AppColors.primaryRed),
                ) {
                    Icon(Icons.Default.DeleteSweep, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Clear All")
                }
            }
        }

        if (uiState.recentInteractions.isEmpty()) {
            // Empty state
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        Icons.Default.DataUsage,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                        modifier = Modifier.size(64.dp),
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "No Training Data Yet",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Start chatting with the model to collect\ninteraction data for fine-tuning",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                    )
                }
            }
        } else {
            LazyColumn(
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(uiState.recentInteractions, key = { it.id }) { interaction ->
                    InteractionCard(
                        interaction = interaction,
                        onRate = { rating -> viewModel.rateInteraction(interaction.id, rating) },
                        onTogglePositive = { viewModel.markAsPositive(interaction.id) },
                        onToggleInclude = { viewModel.toggleTrainingInclusion(interaction.id) },
                        onDelete = { viewModel.deleteInteraction(interaction.id) },
                        onClick = { viewModel.showInteractionDetail(interaction) },
                    )
                }
            }
        }
    }
}

@Composable
private fun InteractionCard(
    interaction: TrainingInteraction,
    onRate: (Int) -> Unit,
    onTogglePositive: () -> Unit,
    onToggleInclude: () -> Unit,
    onDelete: () -> Unit,
    onClick: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (interaction.includeInTraining) {
                MaterialTheme.colorScheme.surface
            } else {
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
            },
        ),
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            // User prompt
            Row(verticalAlignment = Alignment.Top) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "User:",
                        style = MaterialTheme.typography.labelSmall,
                        color = AppColors.primaryBlue,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = interaction.userPrompt,
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                // Category chip
                interaction.category?.let { cat ->
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = AppColors.primaryPurple.copy(alpha = 0.1f),
                    ) {
                        Text(
                            text = cat,
                            style = MaterialTheme.typography.labelSmall,
                            color = AppColors.primaryPurple,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Model response (truncated)
            Text(
                text = "Assistant:",
                style = MaterialTheme.typography.labelSmall,
                color = AppColors.primaryGreen,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = interaction.correctedResponse ?: interaction.modelResponse,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                color = if (interaction.correctedResponse != null) {
                    AppColors.primaryGreen
                } else {
                    MaterialTheme.colorScheme.onSurface
                },
            )

            if (interaction.correctedResponse != null) {
                Text(
                    text = "(corrected)",
                    style = MaterialTheme.typography.labelSmall,
                    color = AppColors.primaryGreen,
                )
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Action row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Star rating
                Row {
                    for (i in 1..5) {
                        IconButton(
                            onClick = { onRate(i) },
                            modifier = Modifier.size(28.dp),
                        ) {
                            Icon(
                                imageVector = if ((interaction.rating ?: 0) >= i) {
                                    Icons.Filled.Star
                                } else {
                                    Icons.Filled.StarBorder
                                },
                                contentDescription = "Rate $i",
                                tint = if ((interaction.rating ?: 0) >= i) {
                                    AppColors.primaryYellow
                                } else {
                                    MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                                },
                                modifier = Modifier.size(18.dp),
                            )
                        }
                    }
                }

                Row {
                    // Positive example toggle
                    IconButton(onClick = onTogglePositive, modifier = Modifier.size(32.dp)) {
                        Icon(
                            imageVector = if (interaction.isPositiveExample) {
                                Icons.Filled.ThumbUp
                            } else {
                                Icons.Filled.ThumbUpOffAlt
                            },
                            contentDescription = "Mark as positive",
                            tint = if (interaction.isPositiveExample) AppColors.primaryGreen else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(18.dp),
                        )
                    }

                    // Include in training toggle
                    IconButton(onClick = onToggleInclude, modifier = Modifier.size(32.dp)) {
                        Icon(
                            imageVector = if (interaction.includeInTraining) Icons.Default.CheckCircle else Icons.Default.Cancel,
                            contentDescription = "Toggle include",
                            tint = if (interaction.includeInTraining) AppColors.primaryBlue else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                            modifier = Modifier.size(18.dp),
                        )
                    }

                    // Delete
                    IconButton(onClick = onDelete, modifier = Modifier.size(32.dp)) {
                        Icon(
                            Icons.Default.Delete,
                            contentDescription = "Delete",
                            tint = AppColors.primaryRed.copy(alpha = 0.6f),
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }

            // Metadata row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = DateUtils.getRelativeTimeSpanString(interaction.timestamp).toString(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
                if (interaction.frequencyCount > 1) {
                    Text(
                        text = "Asked ${interaction.frequencyCount}x",
                        style = MaterialTheme.typography.labelSmall,
                        color = AppColors.primaryOrange,
                    )
                }
            }
        }
    }
}

// =============================================================================
// Train Tab
// =============================================================================

@Composable
private fun TrainTab(uiState: FineTuneUiState, viewModel: FineTuneViewModel) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // Training Status
        when (val status = uiState.trainingStatus) {
            is FineTuningStatus.Idle -> {
                TrainingSetupCard(uiState, viewModel)
            }
            is FineTuningStatus.PreparingData -> {
                TrainingProgressCard("Preparing Data...", status.progress, null, viewModel)
            }
            is FineTuningStatus.Training -> {
                ActiveTrainingCard(status, uiState.trainingProgress, viewModel)
            }
            is FineTuningStatus.SavingAdapter -> {
                TrainingProgressCard("Saving Adapter...", 0.9f + status.progress * 0.1f, null, viewModel)
            }
            is FineTuningStatus.Completed -> {
                CompletedTrainingCard(status.adapter, status.metrics, viewModel)
            }
            is FineTuningStatus.Failed -> {
                FailedTrainingCard(status.error, status.partialMetrics, viewModel)
            }
            is FineTuningStatus.Cancelled -> {
                CancelledTrainingCard(viewModel)
            }
            is FineTuningStatus.CollectingData -> {
                TrainingSetupCard(uiState, viewModel)
            }
        }

        // Quick Config Summary
        ConfigSummaryCard(uiState.config, viewModel)

        // Device Status
        DeviceStatusCard(uiState)
    }
}

@Composable
private fun TrainingSetupCard(uiState: FineTuneUiState, viewModel: FineTuneViewModel) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = AppColors.primaryPurple.copy(alpha = 0.06f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Default.ModelTraining,
                contentDescription = null,
                tint = AppColors.primaryPurple,
                modifier = Modifier.size(48.dp),
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Train Your Adapter",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(8.dp))

            // Adapter name input
            OutlinedTextField(
                value = uiState.adapterName,
                onValueChange = viewModel::updateAdapterName,
                label = { Text("Adapter Name") },
                placeholder = { Text("e.g., My Custom Style") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Data readiness
            val ready = uiState.dataStats.readyForTraining
            val needed = uiState.config.minTrainingSamples
            val progress = (ready.toFloat() / needed).coerceIn(0f, 1f)

            Column(modifier = Modifier.fillMaxWidth()) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = "Training Data",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = "$ready / $needed samples",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (ready >= needed) AppColors.primaryGreen else AppColors.primaryOrange,
                    )
                }
                Spacer(modifier = Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = if (ready >= needed) AppColors.primaryGreen else AppColors.primaryOrange,
                    trackColor = MaterialTheme.colorScheme.outlineVariant,
                    strokeCap = StrokeCap.Round,
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Estimated time
            if (uiState.estimatedTrainingTimeMs > 0) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Timer,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Estimated time: ${formatDuration(uiState.estimatedTrainingTimeMs)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Start button
            Button(
                onClick = viewModel::startTraining,
                enabled = uiState.canStartTraining,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = AppColors.primaryPurple,
                ),
                shape = RoundedCornerShape(14.dp),
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Start Training",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
            }

            if (!uiState.canStartTraining) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = when {
                        uiState.currentModelId == null -> "Load a model first"
                        ready < needed -> "Need ${needed - ready} more training samples"
                        uiState.adapterName.isBlank() -> "Enter an adapter name"
                        !uiState.isDeviceReady -> "Device not ready (check battery/memory)"
                        else -> ""
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = AppColors.primaryRed,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
private fun ActiveTrainingCard(
    status: FineTuningStatus.Training,
    overallProgress: Float,
    viewModel: FineTuneViewModel,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = AppColors.primaryOrange.copy(alpha = 0.06f),
        ),
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            // Pulsing training indicator
            val infiniteTransition = rememberInfiniteTransition(label = "training")
            val pulseAlpha by infiniteTransition.animateFloat(
                initialValue = 0.6f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    tween(1000, easing = LinearEasing),
                    RepeatMode.Reverse,
                ),
                label = "pulse",
            )

            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = AppColors.primaryOrange,
                    strokeWidth = 3.dp,
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "Training in Progress",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = AppColors.primaryOrange.copy(alpha = pulseAlpha),
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Overall progress bar
            val animatedProgress by animateFloatAsState(
                targetValue = overallProgress,
                animationSpec = tween(300),
                label = "progress",
            )
            LinearProgressIndicator(
                progress = { animatedProgress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(10.dp)
                    .clip(RoundedCornerShape(5.dp)),
                color = AppColors.primaryOrange,
                trackColor = AppColors.primaryOrange.copy(alpha = 0.15f),
                strokeCap = StrokeCap.Round,
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Progress text
            Text(
                text = "${"%.1f".format(overallProgress * 100)}% complete",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.End,
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Metrics grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                MetricItem("Epoch", "${status.currentEpoch}/${status.totalEpochs}")
                MetricItem("Step", "${status.currentStep}/${status.totalSteps}")
                MetricItem("Loss", "${"%.4f".format(status.currentLoss)}")
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                MetricItem("Tokens/s", "${"%.1f".format(status.tokensPerSecond)}")
                MetricItem("Memory", "${status.memoryUsageMB}MB")
                MetricItem("ETA", formatDuration(status.estimatedTimeRemainingMs))
            }

            Spacer(modifier = Modifier.height(20.dp))

            // Cancel button
            OutlinedButton(
                onClick = viewModel::cancelTraining,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.primaryRed),
                border = BorderStroke(1.dp, AppColors.primaryRed.copy(alpha = 0.5f)),
            ) {
                Icon(Icons.Default.Stop, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Cancel Training")
            }
        }
    }
}

@Composable
private fun MetricItem(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun TrainingProgressCard(
    title: String,
    progress: Float,
    subtitle: String?,
    viewModel: FineTuneViewModel,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            CircularProgressIndicator(color = AppColors.primaryPurple)
            Spacer(modifier = Modifier.height(16.dp))
            Text(text = title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            subtitle?.let {
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(modifier = Modifier.height(12.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(4.dp)),
                color = AppColors.primaryPurple,
                strokeCap = StrokeCap.Round,
            )
        }
    }
}

@Composable
private fun CompletedTrainingCard(
    adapter: LoraAdapter,
    metrics: TrainingMetrics,
    viewModel: FineTuneViewModel,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = AppColors.primaryGreen.copy(alpha = 0.06f),
        ),
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = AppColors.primaryGreen,
                    modifier = Modifier.size(28.dp),
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "Training Complete!",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = AppColors.primaryGreen,
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Results summary
            OutlinedCard(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(14.dp)) {
                    ResultRow("Adapter", adapter.name)
                    ResultRow("Loss", "${"%.4f".format(metrics.initialLoss)} → ${"%.4f".format(metrics.finalLoss)}")
                    ResultRow("Improvement", "${"%.1f".format((1 - metrics.finalLoss / metrics.initialLoss.coerceAtLeast(0.001f)) * 100)}%")
                    ResultRow("Duration", formatDuration(metrics.trainingDurationMs))
                    ResultRow("Steps", "${metrics.totalSteps}")
                    ResultRow("Speed", "${"%.1f".format(metrics.tokensPerSecond)} tokens/s")
                    ResultRow("Peak Memory", "${metrics.peakMemoryMB}MB")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = { viewModel.activateAdapter(adapter.id) },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = AppColors.primaryGreen),
            ) {
                Icon(Icons.Default.Extension, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Activate Adapter")
            }

            Spacer(modifier = Modifier.height(10.dp))

            OutlinedButton(
                onClick = viewModel::prepareForNewTraining,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Default.ModelTraining, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Train Another Adapter")
            }
        }
    }
}

@Composable
private fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

@Composable
private fun FailedTrainingCard(
    error: String,
    partialMetrics: TrainingMetrics?,
    viewModel: FineTuneViewModel,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = AppColors.primaryRed.copy(alpha = 0.06f)),
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Close, contentDescription = null, tint = AppColors.primaryRed, modifier = Modifier.size(28.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text("Training Failed", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = AppColors.primaryRed)
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(text = error, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

            partialMetrics?.let { metrics ->
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Completed ${metrics.totalSteps} steps before failure",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Spacer(modifier = Modifier.height(14.dp))
            OutlinedButton(
                onClick = viewModel::prepareForNewTraining,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Default.ModelTraining, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Try New Training")
            }
        }
    }
}

@Composable
private fun CancelledTrainingCard(viewModel: FineTuneViewModel) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(Icons.Default.Cancel, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(40.dp))
            Spacer(modifier = Modifier.height(12.dp))
            Text("Training Cancelled", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(4.dp))
            Text("No adapter was saved", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

            Spacer(modifier = Modifier.height(14.dp))
            OutlinedButton(
                onClick = viewModel::prepareForNewTraining,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Default.ModelTraining, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Start New Training")
            }
        }
    }
}

@Composable
private fun ConfigSummaryCard(config: FineTuningConfig, viewModel: FineTuneViewModel) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Tune, contentDescription = null, tint = AppColors.primaryPurple, modifier = Modifier.size(20.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Training Configuration", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                }
                TextButton(onClick = { viewModel.showConfigSheet(true) }) {
                    Icon(Icons.Default.Settings, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Edit")
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                ConfigChip("Rank ${config.loraRank}")
                ConfigChip("α=${config.loraAlpha}")
                ConfigChip("LR ${"%.0e".format(config.learningRate)}")
                ConfigChip("${config.epochs} epochs")
            }
            Spacer(modifier = Modifier.height(6.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                ConfigChip("Seq ${config.maxSeqLength}")
                ConfigChip(if (config.use4BitQuantization) "QLoRA 4-bit" else "Full precision")
                ConfigChip("Grad acc ${config.gradientAccumulationSteps}")
            }
        }
    }
}

@Composable
private fun ConfigChip(label: String) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun DeviceStatusCard(uiState: FineTuneUiState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Memory, contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Device Status", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.height(10.dp))

            val runtime = Runtime.getRuntime()
            val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
            val maxMB = runtime.maxMemory() / (1024 * 1024)

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("Memory", style = MaterialTheme.typography.bodySmall)
                Text("${usedMB}MB / ${maxMB}MB", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
            }
            Spacer(modifier = Modifier.height(4.dp))
            LinearProgressIndicator(
                progress = { (usedMB.toFloat() / maxMB).coerceIn(0f, 1f) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp)),
                color = when {
                    usedMB > maxMB * 0.8 -> AppColors.primaryRed
                    usedMB > maxMB * 0.6 -> AppColors.primaryYellow
                    else -> AppColors.primaryGreen
                },
                strokeCap = StrokeCap.Round,
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("Ready for training", style = MaterialTheme.typography.bodySmall)
                Icon(
                    imageVector = if (uiState.isDeviceReady) Icons.Default.CheckCircle else Icons.Default.Cancel,
                    contentDescription = null,
                    tint = if (uiState.isDeviceReady) AppColors.primaryGreen else AppColors.primaryRed,
                    modifier = Modifier.size(18.dp),
                )
            }
        }
    }
}

// =============================================================================
// Adapters Tab
// =============================================================================

@Composable
private fun AdaptersTab(uiState: FineTuneUiState, viewModel: FineTuneViewModel) {
    if (uiState.adapters.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    Icons.Default.Extension,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                    modifier = Modifier.size(64.dp),
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "No Adapters Yet",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Train your first adapter to personalize\nthe model to your needs",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center,
                )
            }
        }
    } else {
        LazyColumn(
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Active adapter banner
            uiState.activeAdapter?.let { active ->
                item {
                    ActiveAdapterBanner(active, onDeactivate = viewModel::deactivateAdapter)
                }
            }

            items(uiState.adapters, key = { it.id }) { adapter ->
                AdapterCard(
                    adapter = adapter,
                    isActive = adapter.id == uiState.activeAdapter?.id,
                    onActivate = { viewModel.activateAdapter(adapter.id) },
                    onDeactivate = viewModel::deactivateAdapter,
                    onDelete = { viewModel.confirmDeleteAdapter(adapter.id) },
                )
            }
        }
    }
}

@Composable
private fun ActiveAdapterBanner(adapter: LoraAdapter, onDeactivate: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = AppColors.primaryGreen.copy(alpha = 0.1f),
        ),
        border = BorderStroke(1.dp, AppColors.primaryGreen.copy(alpha = 0.3f)),
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Default.CheckCircle,
                contentDescription = null,
                tint = AppColors.primaryGreen,
                modifier = Modifier.size(24.dp),
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Active: ${adapter.name}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = AppColors.primaryGreen,
                )
                Text(
                    text = "Personalized responses enabled",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            OutlinedButton(
                onClick = onDeactivate,
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
            ) {
                Text("Deactivate", style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}

@Composable
private fun AdapterCard(
    adapter: LoraAdapter,
    isActive: Boolean,
    onActivate: () -> Unit,
    onDeactivate: () -> Unit,
    onDelete: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        border = if (isActive) BorderStroke(1.5.dp, AppColors.primaryGreen) else null,
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Extension,
                    contentDescription = null,
                    tint = if (isActive) AppColors.primaryGreen else AppColors.primaryPurple,
                    modifier = Modifier.size(24.dp),
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = adapter.name,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "v${adapter.version} • ${adapter.baseModelName}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (isActive) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = AppColors.primaryGreen.copy(alpha = 0.15f),
                    ) {
                        Text(
                            text = "Active",
                            style = MaterialTheme.typography.labelSmall,
                            color = AppColors.primaryGreen,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Adapter details
            if (adapter.description.isNotBlank()) {
                Text(
                    text = adapter.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            // Metrics row
            adapter.trainingMetrics?.let { metrics ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                ) {
                    MiniMetric("Loss", "${"%.4f".format(metrics.finalLoss)}")
                    MiniMetric("Samples", "${adapter.trainingSampleCount}")
                    MiniMetric("Size", formatFileSize(adapter.adapterSizeBytes))
                    MiniMetric("Steps", "${metrics.totalSteps}")
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Config chips
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ConfigChip("Rank ${adapter.config.loraRank}")
                ConfigChip("${adapter.config.epochs} epochs")
                if (adapter.config.use4BitQuantization) ConfigChip("QLoRA")
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Action buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (isActive) {
                    OutlinedButton(
                        onClick = onDeactivate,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Deactivate")
                    }
                } else {
                    Button(
                        onClick = onActivate,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = AppColors.primaryPurple),
                    ) {
                        Text("Activate")
                    }
                }
                OutlinedButton(
                    onClick = onDelete,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.primaryRed),
                    border = BorderStroke(1.dp, AppColors.primaryRed.copy(alpha = 0.3f)),
                ) {
                    Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                }
            }

            // Timestamp
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Created ${DateUtils.getRelativeTimeSpanString(adapter.createdAt)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
            )
        }
    }
}

@Composable
private fun MiniMetric(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(text = value, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// =============================================================================
// Config Bottom Sheet
// =============================================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ConfigBottomSheet(
    config: FineTuningConfig,
    onDismiss: () -> Unit,
    onSave: (FineTuningConfig) -> Unit,
) {
    var rank by remember { mutableIntStateOf(config.loraRank) }
    var alpha by remember { mutableIntStateOf(config.loraAlpha) }
    var lr by remember { mutableFloatStateOf(config.learningRate) }
    var epochs by remember { mutableIntStateOf(config.epochs) }
    var maxSeq by remember { mutableIntStateOf(config.maxSeqLength) }
    var gradAccum by remember { mutableIntStateOf(config.gradientAccumulationSteps) }
    var use4Bit by remember { mutableStateOf(config.use4BitQuantization) }
    var gradCheckpoint by remember { mutableStateOf(config.gradientCheckpointing) }
    var dropout by remember { mutableFloatStateOf(config.loraDropout) }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 8.dp),
        ) {
            Text(
                text = "Training Configuration",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Advanced settings for LoRA fine-tuning",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Spacer(modifier = Modifier.height(24.dp))

            // LoRA Rank
            ConfigSlider(
                label = "LoRA Rank",
                value = rank.toFloat(),
                valueRange = 2f..64f,
                steps = 30,
                valueLabel = "$rank",
                description = "Lower = smaller adapter, faster training. Higher = more expressive.",
                onValueChange = { rank = it.toInt() },
            )

            // LoRA Alpha
            ConfigSlider(
                label = "LoRA Alpha",
                value = alpha.toFloat(),
                valueRange = 4f..128f,
                steps = 30,
                valueLabel = "$alpha",
                description = "Scaling factor. Typically 2× rank.",
                onValueChange = { alpha = it.toInt() },
            )

            // Learning Rate
            ConfigSlider(
                label = "Learning Rate",
                value = lr * 10000,
                valueRange = 0.1f..50f,
                steps = 49,
                valueLabel = "${"%.1e".format(lr)}",
                description = "Lower = slower but more stable. Higher = faster but may diverge.",
                onValueChange = { lr = it / 10000 },
            )

            // Epochs
            ConfigSlider(
                label = "Training Epochs",
                value = epochs.toFloat(),
                valueRange = 1f..10f,
                steps = 8,
                valueLabel = "$epochs",
                description = "Number of passes over the training data.",
                onValueChange = { epochs = it.toInt() },
            )

            // Max Sequence Length
            ConfigSlider(
                label = "Max Sequence Length",
                value = maxSeq.toFloat(),
                valueRange = 128f..2048f,
                steps = 14,
                valueLabel = "$maxSeq",
                description = "Maximum tokens per training sample.",
                onValueChange = { maxSeq = it.toInt() },
            )

            // Gradient Accumulation
            ConfigSlider(
                label = "Gradient Accumulation Steps",
                value = gradAccum.toFloat(),
                valueRange = 1f..16f,
                steps = 14,
                valueLabel = "$gradAccum",
                description = "Simulates larger batch size. Higher = more stable but slower.",
                onValueChange = { gradAccum = it.toInt() },
            )

            // Dropout
            ConfigSlider(
                label = "LoRA Dropout",
                value = dropout,
                valueRange = 0f..0.5f,
                steps = 9,
                valueLabel = "${"%.2f".format(dropout)}",
                description = "Regularization to prevent overfitting.",
                onValueChange = { dropout = it },
            )

            // Toggle switches
            Spacer(modifier = Modifier.height(8.dp))
            ConfigToggle(
                label = "QLoRA (4-bit Quantization)",
                description = "Reduces memory usage during training",
                checked = use4Bit,
                onCheckedChange = { use4Bit = it },
            )

            ConfigToggle(
                label = "Gradient Checkpointing",
                description = "Trade compute for memory savings",
                checked = gradCheckpoint,
                onCheckedChange = { gradCheckpoint = it },
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Save button
            Button(
                onClick = {
                    onSave(
                        config.copy(
                            loraRank = rank,
                            loraAlpha = alpha,
                            learningRate = lr,
                            epochs = epochs,
                            maxSeqLength = maxSeq,
                            gradientAccumulationSteps = gradAccum,
                            use4BitQuantization = use4Bit,
                            gradientCheckpointing = gradCheckpoint,
                            loraDropout = dropout,
                        ),
                    )
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = AppColors.primaryPurple),
            ) {
                Text("Save Configuration")
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

@Composable
private fun ConfigSlider(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    valueLabel: String,
    description: String,
    onValueChange: (Float) -> Unit,
) {
    Column(modifier = Modifier.padding(vertical = 6.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(text = label, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
            Text(text = valueLabel, style = MaterialTheme.typography.bodyMedium, color = AppColors.primaryPurple, fontWeight = FontWeight.Bold)
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps,
            modifier = Modifier.fillMaxWidth(),
        )
        Text(text = description, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ConfigToggle(
    label: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = label, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
            Text(text = description, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(modifier = Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

private fun formatDuration(ms: Long): String {
    if (ms <= 0) return "—"
    val seconds = ms / 1000
    val minutes = seconds / 60
    val hours = minutes / 60
    return when {
        hours > 0 -> "${hours}h ${minutes % 60}m"
        minutes > 0 -> "${minutes}m ${seconds % 60}s"
        else -> "${seconds}s"
    }
}

private fun formatFileSize(bytes: Long): String {
    return when {
        bytes >= 1024 * 1024 -> "${"%.1f".format(bytes / (1024.0 * 1024))}MB"
        bytes >= 1024 -> "${"%.1f".format(bytes / 1024.0)}KB"
        else -> "${bytes}B"
    }
}
