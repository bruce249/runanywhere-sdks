package com.runanywhere.runanywhereai.presentation.finetune

import android.text.format.DateUtils
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Extension
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Sync
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.runanywhere.runanywhereai.data.cloud.CloudAdapterInfo
import com.runanywhere.runanywhereai.data.cloud.CloudSyncStatus
import com.runanywhere.runanywhereai.data.cloud.CloudUiState
import com.runanywhere.runanywhereai.ui.theme.AppColors

// =============================================================================
// Cloud Fine-Tune Tab — Full Compose UI
// =============================================================================

@Composable
fun CloudFineTuneTab(
    cloudState: CloudUiState,
    onServerUrlChanged: (String) -> Unit,
    onSaveConfig: () -> Unit,
    onCheckConnection: () -> Unit,
    onUploadData: () -> Unit,
    onStartTraining: () -> Unit,
    onPollStatus: () -> Unit,
    onPollAdapters: () -> Unit,
    onDownloadAdapter: (String) -> Unit,
    onToggleAutoSync: (Boolean) -> Unit,
    onClearError: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // ── Server Configuration ────────────────────────────────────────
        ServerConfigCard(
            serverUrl = cloudState.config.serverUrl,
            isReachable = cloudState.isServerReachable,
            isBusy = cloudState.isBusy,
            onServerUrlChanged = onServerUrlChanged,
            onSaveConfig = onSaveConfig,
            onCheckConnection = onCheckConnection,
        )

        // ── Connection Status Banner ────────────────────────────────────
        SyncStatusBanner(
            syncStatus = cloudState.syncStatus,
            onClearError = onClearError,
        )

        // ── Multi-Modal Data Summary ────────────────────────────────────
        DataSummaryCard(
            totalReady = cloudState.dataReadyForUpload,
            textCount = cloudState.textChatCount,
            voiceCount = cloudState.voiceChatCount,
            videoCount = cloudState.videoAnalysisCount,
        )

        // ── Send to Cloud Button ────────────────────────────────────────
        SendToCloudCard(
            dataCount = cloudState.dataReadyForUpload,
            isBusy = cloudState.isBusy,
            hasServerUrl = cloudState.hasServerUrl,
            isReachable = cloudState.isServerReachable,
            onUpload = onUploadData,
        )

        // ── Cloud Training ──────────────────────────────────────────────
        CloudTrainingCard(
            activeJobStatus = cloudState.activeJobStatus,
            syncStatus = cloudState.syncStatus,
            lastUploadId = cloudState.lastUploadId,
            isBusy = cloudState.isBusy,
            onStartTraining = onStartTraining,
            onPollStatus = onPollStatus,
        )

        // ── OTA Adapters ────────────────────────────────────────────────
        OtaAdaptersCard(
            adapters = cloudState.availableAdapters,
            downloadedIds = cloudState.downloadedAdapterIds,
            isBusy = cloudState.isBusy,
            onPollAdapters = onPollAdapters,
            onDownloadAdapter = onDownloadAdapter,
        )

        // ── Auto-Sync Settings ──────────────────────────────────────────
        AutoSyncCard(
            autoSync = cloudState.config.autoSync,
            lastSyncTime = cloudState.lastSyncTime,
            onToggle = onToggleAutoSync,
        )

        Spacer(modifier = Modifier.height(80.dp))
    }
}

// =============================================================================
// Server Configuration Card
// =============================================================================

@Composable
private fun ServerConfigCard(
    serverUrl: String,
    isReachable: Boolean,
    isBusy: Boolean,
    onServerUrlChanged: (String) -> Unit,
    onSaveConfig: () -> Unit,
    onCheckConnection: () -> Unit,
) {
    var editUrl by remember(serverUrl) { mutableStateOf(serverUrl) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Cloud,
                    contentDescription = null,
                    tint = AppColors.primaryAccent,
                    modifier = Modifier.size(24.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Cloud Server",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.weight(1f))
                // Connection indicator
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (isReachable) AppColors.primaryGreen.copy(alpha = 0.15f)
                    else MaterialTheme.colorScheme.error.copy(alpha = 0.15f),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .padding(0.dp),
                        ) {
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = if (isReachable) AppColors.primaryGreen
                                else MaterialTheme.colorScheme.error,
                                modifier = Modifier.fillMaxSize(),
                            ) {}
                        }
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            if (isReachable) "Connected" else "Disconnected",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isReachable) AppColors.primaryGreen
                            else MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = editUrl,
                onValueChange = { editUrl = it },
                label = { Text("Server URL") },
                placeholder = { Text("http://192.168.1.100:8000") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(
                    onClick = {
                        onServerUrlChanged(editUrl)
                        onSaveConfig()
                    },
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Save")
                }
                Button(
                    onClick = {
                        onServerUrlChanged(editUrl)
                        onSaveConfig()
                        onCheckConnection()
                    },
                    enabled = editUrl.isNotBlank() && !isBusy,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.Sync, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Test Connection")
                }
            }
        }
    }
}

// =============================================================================
// Sync Status Banner
// =============================================================================

@Composable
private fun SyncStatusBanner(
    syncStatus: CloudSyncStatus,
    onClearError: () -> Unit,
) {
    when (syncStatus) {
        is CloudSyncStatus.Disconnected -> { /* No banner */ }
        is CloudSyncStatus.Connecting -> StatusBanner(
            icon = Icons.Default.Sync,
            text = "Connecting to cloud...",
            color = AppColors.primaryAccent,
            showProgress = true,
        )
        is CloudSyncStatus.Connected -> StatusBanner(
            icon = Icons.Default.CheckCircle,
            text = "Connected to cloud server",
            color = AppColors.primaryGreen,
        )
        is CloudSyncStatus.Uploading -> StatusBanner(
            icon = Icons.Default.CloudUpload,
            text = syncStatus.message.ifEmpty { "Uploading..." },
            color = AppColors.primaryAccent,
            showProgress = true,
            progress = syncStatus.progress,
        )
        is CloudSyncStatus.Training -> StatusBanner(
            icon = Icons.Default.PlayArrow,
            text = syncStatus.status.ifEmpty { "Training in progress..." },
            color = AppColors.primaryOrange,
            showProgress = true,
            progress = syncStatus.progress,
        )
        is CloudSyncStatus.DownloadingAdapter -> StatusBanner(
            icon = Icons.Default.CloudDownload,
            text = "Downloading adapter...",
            color = AppColors.primaryAccent,
            showProgress = true,
            progress = syncStatus.progress,
        )
        is CloudSyncStatus.Error -> StatusBanner(
            icon = Icons.Default.Error,
            text = syncStatus.message,
            color = MaterialTheme.colorScheme.error,
            onDismiss = onClearError,
        )
        is CloudSyncStatus.Completed -> StatusBanner(
            icon = Icons.Default.CheckCircle,
            text = syncStatus.message,
            color = AppColors.primaryGreen,
        )
    }
}

@Composable
private fun StatusBanner(
    icon: ImageVector,
    text: String,
    color: androidx.compose.ui.graphics.Color,
    showProgress: Boolean = false,
    progress: Float = -1f,
    onDismiss: (() -> Unit)? = null,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f)),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(20.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = color,
                    modifier = Modifier.weight(1f),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                if (onDismiss != null) {
                    IconButton(onClick = onDismiss, modifier = Modifier.size(24.dp)) {
                        Icon(Icons.Default.Error, contentDescription = "Dismiss", tint = color)
                    }
                }
            }
            if (showProgress && progress >= 0f) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth(),
                    color = color,
                    strokeCap = StrokeCap.Round,
                )
            } else if (showProgress) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                    color = color,
                    strokeCap = StrokeCap.Round,
                )
            }
        }
    }
}

// =============================================================================
// Multi-Modal Data Summary Card
// =============================================================================

@Composable
private fun DataSummaryCard(
    totalReady: Int,
    textCount: Int,
    voiceCount: Int,
    videoCount: Int,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Training Data Collected",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "$totalReady interactions ready for upload",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                DataSourceChip(
                    icon = Icons.Default.Chat,
                    label = "Text Chat",
                    count = textCount,
                    color = AppColors.primaryAccent,
                )
                DataSourceChip(
                    icon = Icons.Default.Mic,
                    label = "Voice",
                    count = voiceCount,
                    color = AppColors.primaryPurple,
                )
                DataSourceChip(
                    icon = Icons.Default.Videocam,
                    label = "Video",
                    count = videoCount,
                    color = AppColors.primaryOrange,
                )
            }
        }
    }
}

@Composable
private fun DataSourceChip(
    icon: ImageVector,
    label: String,
    count: Int,
    color: androidx.compose.ui.graphics.Color,
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Surface(
            shape = RoundedCornerShape(12.dp),
            color = color.copy(alpha = 0.12f),
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    "$count",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = color,
                )
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// =============================================================================
// Send to Cloud Card
// =============================================================================

@Composable
private fun SendToCloudCard(
    dataCount: Int,
    isBusy: Boolean,
    hasServerUrl: Boolean,
    isReachable: Boolean,
    onUpload: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = AppColors.primaryAccent.copy(alpha = 0.08f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Default.CloudUpload,
                contentDescription = null,
                tint = AppColors.primaryAccent,
                modifier = Modifier.size(40.dp),
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Send Data to Cloud",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Upload $dataCount interactions to your cloud server for fine-tuning",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Button(
                onClick = onUpload,
                enabled = dataCount > 0 && !isBusy && hasServerUrl && isReachable,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = AppColors.primaryAccent,
                ),
            ) {
                if (isBusy) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Icon(Icons.Default.CloudUpload, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(if (isBusy) "Uploading..." else "Upload All Data")
            }
            if (!hasServerUrl) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Configure server URL above first",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

// =============================================================================
// Cloud Training Card
// =============================================================================

@Composable
private fun CloudTrainingCard(
    activeJobStatus: com.runanywhere.runanywhereai.data.cloud.CloudJobStatus?,
    syncStatus: CloudSyncStatus,
    lastUploadId: String?,
    isBusy: Boolean,
    onStartTraining: () -> Unit,
    onPollStatus: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.PlayArrow,
                    contentDescription = null,
                    tint = AppColors.primaryOrange,
                    modifier = Modifier.size(24.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Cloud Training",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (activeJobStatus != null) {
                // Show active job status
                OutlinedCard(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Job: ${activeJobStatus.jobId.take(8)}...",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.Medium,
                            )
                            Spacer(modifier = Modifier.weight(1f))
                            JobStatusChip(activeJobStatus.status)
                        }
                        if (activeJobStatus.currentEpoch != null && activeJobStatus.totalEpochs != null) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                "Epoch ${activeJobStatus.currentEpoch}/${activeJobStatus.totalEpochs}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (activeJobStatus.currentLoss != null) {
                            Text(
                                "Loss: %.4f".format(activeJobStatus.currentLoss),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (activeJobStatus.progress > 0f) {
                            Spacer(modifier = Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = { activeJobStatus.progress },
                                modifier = Modifier.fillMaxWidth(),
                                color = AppColors.primaryOrange,
                                strokeCap = StrokeCap.Round,
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = onPollStatus,
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isBusy,
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Refresh Status")
                }
            } else {
                Text(
                    if (lastUploadId != null) {
                        "Data uploaded. Start training on the cloud server."
                    } else {
                        "Upload training data first, then start cloud training."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Spacer(modifier = Modifier.height(8.dp))
                Button(
                    onClick = onStartTraining,
                    enabled = lastUploadId != null && !isBusy,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = AppColors.primaryOrange),
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start Cloud Training")
                }
            }
        }
    }
}

@Composable
private fun JobStatusChip(status: String) {
    val (color, label) = when (status) {
        "queued" -> AppColors.primaryAccent to "Queued"
        "processing" -> AppColors.primaryOrange to "Training"
        "completed" -> AppColors.primaryGreen to "Completed"
        "failed" -> MaterialTheme.colorScheme.error to "Failed"
        else -> MaterialTheme.colorScheme.onSurfaceVariant to status
    }
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.15f),
    ) {
        Text(
            label,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.Bold,
        )
    }
}

// =============================================================================
// OTA Adapters Card
// =============================================================================

@Composable
private fun OtaAdaptersCard(
    adapters: List<CloudAdapterInfo>,
    downloadedIds: Set<String>,
    isBusy: Boolean,
    onPollAdapters: () -> Unit,
    onDownloadAdapter: (String) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Extension,
                    contentDescription = null,
                    tint = AppColors.primaryGreen,
                    modifier = Modifier.size(24.dp),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "OTA Adapters",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.weight(1f))
                IconButton(onClick = onPollAdapters, enabled = !isBusy) {
                    Icon(Icons.Default.Refresh, contentDescription = "Refresh")
                }
            }

            if (adapters.isEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "No cloud-trained adapters available yet.\nTap refresh to check for new adapters.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Spacer(modifier = Modifier.height(8.dp))
                adapters.forEach { adapter ->
                    val isDownloaded = downloadedIds.contains(adapter.id)
                    OtaAdapterItem(
                        adapter = adapter,
                        isDownloaded = isDownloaded,
                        isBusy = isBusy,
                        onDownload = { onDownloadAdapter(adapter.id) },
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun OtaAdapterItem(
    adapter: CloudAdapterInfo,
    isDownloaded: Boolean,
    isBusy: Boolean,
    onDownload: () -> Unit,
) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    adapter.name,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                )
                Text(
                    "${adapter.trainingSamples} samples • ${formatFileSize(adapter.sizeBytes)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (adapter.finalLoss != null) {
                    Text(
                        "Loss: %.4f".format(adapter.finalLoss),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            if (isDownloaded) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = AppColors.primaryGreen.copy(alpha = 0.15f),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = AppColors.primaryGreen,
                            modifier = Modifier.size(14.dp),
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            "Downloaded",
                            style = MaterialTheme.typography.labelSmall,
                            color = AppColors.primaryGreen,
                        )
                    }
                }
            } else {
                Button(
                    onClick = onDownload,
                    enabled = !isBusy,
                    colors = ButtonDefaults.buttonColors(containerColor = AppColors.primaryGreen),
                ) {
                    Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Download")
                }
            }
        }
    }
}

// =============================================================================
// Auto-Sync Settings Card
// =============================================================================

@Composable
private fun AutoSyncCard(
    autoSync: Boolean,
    lastSyncTime: Long?,
    onToggle: (Boolean) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "Auto-Sync OTA Adapters",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.weight(1f),
                )
                Switch(
                    checked = autoSync,
                    onCheckedChange = onToggle,
                )
            }
            Text(
                "Automatically poll the cloud server for new adapters in the background",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (lastSyncTime != null) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Last sync: ${DateUtils.getRelativeTimeSpanString(lastSyncTime)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

// =============================================================================
// Utilities
// =============================================================================

private fun formatFileSize(bytes: Long): String {
    return when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        bytes < 1024 * 1024 * 1024 -> "${"%.1f".format(bytes.toFloat() / (1024 * 1024))} MB"
        else -> "${"%.2f".format(bytes.toFloat() / (1024 * 1024 * 1024))} GB"
    }
}
