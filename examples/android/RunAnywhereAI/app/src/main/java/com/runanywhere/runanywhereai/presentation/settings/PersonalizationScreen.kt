@file:OptIn(ExperimentalMaterial3Api::class)

package com.runanywhere.runanywhereai.presentation.settings

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.runanywhere.runanywhereai.data.PersonalizationProfile
import com.runanywhere.runanywhereai.data.PersonalizationTone
import com.runanywhere.runanywhereai.ui.theme.AppColors

// =============================================================================
// Personalization Screen — Premium Full-Screen Dialog
// =============================================================================

/**
 * Full-screen personalization dialog.
 * Designed to feel premium, polished, and alive with micro-animations.
 */
@Composable
fun PersonalizationScreen(
    profile: PersonalizationProfile,
    onDisplayNameChange: (String) -> Unit,
    onAboutUserChange: (String) -> Unit,
    onResponseStyleChange: (String) -> Unit,
    onToneChange: (PersonalizationTone) -> Unit,
    onCustomPromptChange: (String) -> Unit,
    onEnabledChange: (Boolean) -> Unit,
    onReset: () -> Unit,
    onDismiss: () -> Unit,
    systemPromptPreview: String?,
) {
    var showAdvanced by remember { mutableStateOf(profile.customSystemPrompt.isNotBlank()) }
    var showResetConfirm by remember { mutableStateOf(false) }

    // Animated gradient offset for the header
    val infiniteTransition = rememberInfiniteTransition(label = "header_gradient")
    val gradientOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "gradient_shift",
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        // ─── Top Bar ────────────────────────────────────────────────
        Surface(
            tonalElevation = 2.dp,
            color = MaterialTheme.colorScheme.surface,
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(horizontal = 8.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, contentDescription = "Close")
                }
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Personalization",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(modifier = Modifier.weight(1f))
                // Balance the close button
                Spacer(modifier = Modifier.width(48.dp))
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // ─── Hero Banner ────────────────────────────────────────
            HeroBanner(
                isEnabled = profile.isEnabled,
                onEnabledChange = onEnabledChange,
                gradientOffset = gradientOffset,
            )

            // ─── Content (animated visibility based on enabled) ─────
            AnimatedVisibility(
                visible = profile.isEnabled,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically(),
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                ) {
                    // ─── Your Name ──────────────────────────────────
                    PersonalizationTextField(
                        value = profile.displayName,
                        onValueChange = onDisplayNameChange,
                        label = "Your Name",
                        placeholder = "e.g. Kanishk",
                        helperText = "The AI will use your name in conversations",
                        icon = Icons.Outlined.Person,
                        singleLine = true,
                    )

                    // ─── About You ──────────────────────────────────
                    PersonalizationTextField(
                        value = profile.aboutUser,
                        onValueChange = onAboutUserChange,
                        label = "What should the AI know about you?",
                        placeholder = "e.g. I'm a mobile developer working on AI apps. " +
                            "I prefer Kotlin over Java. I use Android Studio daily.",
                        helperText = "Give context so the AI can tailor its responses",
                        icon = Icons.Outlined.Info,
                        singleLine = false,
                        minLines = 3,
                        maxLines = 6,
                    )

                    // ─── Response Style ─────────────────────────────
                    PersonalizationTextField(
                        value = profile.responseStyle,
                        onValueChange = onResponseStyleChange,
                        label = "How should the AI respond?",
                        placeholder = "e.g. Be concise. Use code examples when relevant. " +
                            "Prefer bullet points over paragraphs.",
                        helperText = "Guide the AI's communication style",
                        icon = Icons.Outlined.EditNote,
                        singleLine = false,
                        minLines = 3,
                        maxLines = 6,
                    )

                    // ─── Tone Selector ──────────────────────────────
                    ToneSelector(
                        selectedTone = profile.resolvedTone,
                        onToneSelected = onToneChange,
                    )

                    // ─── Advanced Section ───────────────────────────
                    AdvancedSection(
                        expanded = showAdvanced,
                        onToggle = { showAdvanced = !showAdvanced },
                        customPrompt = profile.customSystemPrompt,
                        onCustomPromptChange = onCustomPromptChange,
                    )

                    // ─── Live Preview ───────────────────────────────
                    SystemPromptPreview(preview = systemPromptPreview)

                    // ─── Reset Button ───────────────────────────────
                    if (profile.hasContent) {
                        OutlinedButton(
                            onClick = { showResetConfirm = true },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = AppColors.primaryRed,
                            ),
                            border = ButtonDefaults.outlinedButtonBorder(enabled = true),
                        ) {
                            Icon(
                                Icons.Outlined.RestartAlt,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp),
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Reset to Defaults")
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }

    // Reset Confirmation Dialog
    if (showResetConfirm) {
        AlertDialog(
            onDismissRequest = { showResetConfirm = false },
            icon = {
                Icon(
                    Icons.Outlined.Warning,
                    contentDescription = null,
                    tint = AppColors.primaryOrange,
                )
            },
            title = { Text("Reset Personalization?") },
            text = { Text("This will clear all your personalization settings. The AI will revert to default behavior.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onReset()
                        showResetConfirm = false
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = AppColors.primaryRed,
                    ),
                ) {
                    Text("Reset")
                }
            },
            dismissButton = {
                TextButton(onClick = { showResetConfirm = false }) {
                    Text("Cancel")
                }
            },
        )
    }
}

// =============================================================================
// Hero Banner
// =============================================================================

@Composable
private fun HeroBanner(
    isEnabled: Boolean,
    onEnabledChange: (Boolean) -> Unit,
    gradientOffset: Float,
) {
    val accentColor = AppColors.primaryAccent
    val secondColor = Color(0xFFFF8C00) // Warm orange

    // Interpolate gradient colors for animation
    val startColor = Color(
        red = accentColor.red + (secondColor.red - accentColor.red) * gradientOffset,
        green = accentColor.green + (secondColor.green - accentColor.green) * gradientOffset,
        blue = accentColor.blue + (secondColor.blue - accentColor.blue) * gradientOffset,
    )
    val endColor = Color(
        red = secondColor.red + (accentColor.red - secondColor.red) * gradientOffset,
        green = secondColor.green + (accentColor.green - secondColor.green) * gradientOffset,
        blue = secondColor.blue + (accentColor.blue - secondColor.blue) * gradientOffset,
    )

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        shadowElevation = 8.dp,
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(startColor, endColor),
                    ),
                )
                .padding(24.dp),
        ) {
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Make the AI yours ✨",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Customize how the AI knows you and responds",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.White.copy(alpha = 0.85f),
                        )
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    // Master toggle
                    Switch(
                        checked = isEnabled,
                        onCheckedChange = onEnabledChange,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = Color.White.copy(alpha = 0.3f),
                            uncheckedThumbColor = Color.White.copy(alpha = 0.7f),
                            uncheckedTrackColor = Color.White.copy(alpha = 0.15f),
                        ),
                    )
                }

                // Status pill
                Spacer(modifier = Modifier.height(12.dp))
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = if (isEnabled) Color.White.copy(alpha = 0.2f)
                    else Color.Black.copy(alpha = 0.2f),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(
                                    if (isEnabled) Color(0xFF4ADE80)
                                    else Color.White.copy(alpha = 0.4f),
                                ),
                        )
                        Text(
                            text = if (isEnabled) "Active — applied to all chats"
                            else "Disabled — using default behavior",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.White,
                        )
                    }
                }
            }
        }
    }
}

// =============================================================================
// Text Fields
// =============================================================================

@Composable
private fun PersonalizationTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    placeholder: String,
    helperText: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    singleLine: Boolean,
    minLines: Int = 1,
    maxLines: Int = 1,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Label with icon
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = AppColors.primaryAccent,
            )
            Text(
                text = label,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
        }

        // Input field
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = {
                Text(
                    text = placeholder,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                    style = MaterialTheme.typography.bodyMedium,
                )
            },
            modifier = Modifier.fillMaxWidth(),
            singleLine = singleLine,
            minLines = if (singleLine) 1 else minLines,
            maxLines = if (singleLine) 1 else maxLines,
            shape = RoundedCornerShape(14.dp),
            keyboardOptions = KeyboardOptions(
                capitalization = KeyboardCapitalization.Sentences,
            ),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = AppColors.primaryAccent,
                focusedLabelColor = AppColors.primaryAccent,
                cursorColor = AppColors.primaryAccent,
            ),
        )

        // Helper text
        Text(
            text = helperText,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
            modifier = Modifier.padding(start = 4.dp),
        )
    }
}

// =============================================================================
// Tone Selector
// =============================================================================

@Composable
private fun ToneSelector(
    selectedTone: PersonalizationTone,
    onToneSelected: (PersonalizationTone) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = Icons.Outlined.Tune,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = AppColors.primaryAccent,
            )
            Text(
                text = "Tone",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
        }

        // Two rows of 3 chips
        val tones = PersonalizationTone.entries
        val rows = tones.chunked(3)

        rows.forEach { rowTones ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                rowTones.forEach { tone ->
                    val isSelected = tone == selectedTone
                    val animatedAlpha by animateFloatAsState(
                        targetValue = if (isSelected) 1f else 0.6f,
                        animationSpec = tween(200),
                        label = "tone_alpha",
                    )

                    Surface(
                        modifier = Modifier
                            .weight(1f)
                            .clickable { onToneSelected(tone) },
                        shape = RoundedCornerShape(14.dp),
                        color = if (isSelected) AppColors.primaryAccent.copy(alpha = 0.12f)
                        else MaterialTheme.colorScheme.surfaceVariant,
                        border = if (isSelected) {
                            androidx.compose.foundation.BorderStroke(
                                1.5.dp,
                                AppColors.primaryAccent,
                            )
                        } else {
                            null
                        },
                    ) {
                        Column(
                            modifier = Modifier
                                .padding(12.dp)
                                .alpha(animatedAlpha),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Text(
                                text = tone.emoji,
                                fontSize = 22.sp,
                            )
                            Text(
                                text = tone.label,
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                color = if (isSelected) AppColors.primaryAccent
                                else MaterialTheme.colorScheme.onSurface,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                    }
                }
                // Fill remaining space if row has < 3 items
                repeat(3 - rowTones.size) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

// =============================================================================
// Advanced Section (Custom System Prompt)
// =============================================================================

@Composable
private fun AdvancedSection(
    expanded: Boolean,
    onToggle: () -> Unit,
    customPrompt: String,
    onCustomPromptChange: (String) -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
    ) {
        Column {
            // Toggle header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(onClick = onToggle)
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = Icons.Outlined.Code,
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = AppColors.primaryPurple,
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Custom System Prompt",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = "Advanced: overrides all settings above",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                    )
                }
                Icon(
                    imageVector = if (expanded) Icons.Default.ExpandLess
                    else Icons.Default.ExpandMore,
                    contentDescription = null,
                    modifier = Modifier.size(24.dp),
                )
            }

            AnimatedVisibility(
                visible = expanded,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically(),
            ) {
                Column(modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
                    // Warning chip
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = AppColors.primaryOrange.copy(alpha = 0.1f),
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Icon(
                                Icons.Outlined.Warning,
                                contentDescription = null,
                                modifier = Modifier.size(14.dp),
                                tint = AppColors.primaryOrange,
                            )
                            Text(
                                text = "When set, this replaces the auto-generated prompt above",
                                style = MaterialTheme.typography.labelSmall,
                                color = AppColors.primaryOrange,
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    OutlinedTextField(
                        value = customPrompt,
                        onValueChange = onCustomPromptChange,
                        placeholder = {
                            Text(
                                "You are a senior Android engineer who specializes in Kotlin, Jetpack Compose, and on-device ML...",
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f),
                                style = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = FontFamily.Monospace,
                                ),
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 4,
                        maxLines = 10,
                        shape = RoundedCornerShape(12.dp),
                        textStyle = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                        ),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = AppColors.primaryPurple,
                            cursorColor = AppColors.primaryPurple,
                        ),
                    )
                }
            }
        }
    }
}

// =============================================================================
// System Prompt Preview
// =============================================================================

@Composable
private fun SystemPromptPreview(preview: String?) {
    if (preview == null) return

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = Icons.Outlined.Preview,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = AppColors.primaryGreen,
            )
            Text(
                text = "Live Preview",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
        }

        Surface(
            shape = RoundedCornerShape(14.dp),
            color = AppColors.primaryGreen.copy(alpha = 0.06f),
            border = androidx.compose.foundation.BorderStroke(
                1.dp,
                AppColors.primaryGreen.copy(alpha = 0.15f),
            ),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(14.dp),
            ) {
                Text(
                    text = "SYSTEM PROMPT",
                    style = MaterialTheme.typography.labelSmall,
                    color = AppColors.primaryGreen,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = preview,
                    style = MaterialTheme.typography.bodySmall.copy(
                        fontFamily = FontFamily.Monospace,
                        lineHeight = 18.sp,
                    ),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.75f),
                )
            }
        }
    }
}

// =============================================================================
// Settings Entry Point Card
// =============================================================================

/**
 * Compact card shown in the Settings screen.
 * Tapping opens the full PersonalizationScreen.
 */
@Composable
fun PersonalizationSettingsCard(
    profile: PersonalizationProfile,
    onClick: () -> Unit,
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        tonalElevation = 1.dp,
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.linearGradient(
                        colors = listOf(
                            AppColors.primaryAccent.copy(alpha = 0.08f),
                            AppColors.primaryAccent.copy(alpha = 0.03f),
                        ),
                    ),
                )
                .border(
                    width = 1.dp,
                    brush = Brush.linearGradient(
                        colors = listOf(
                            AppColors.primaryAccent.copy(alpha = 0.2f),
                            AppColors.primaryAccent.copy(alpha = 0.05f),
                        ),
                    ),
                    shape = RoundedCornerShape(16.dp),
                )
                .padding(16.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Icon with gradient background
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(
                            Brush.linearGradient(
                                colors = listOf(
                                    AppColors.primaryAccent,
                                    Color(0xFFFF8C00),
                                ),
                            ),
                        ),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Outlined.AutoAwesome,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(24.dp),
                    )
                }

                Spacer(modifier = Modifier.width(14.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Personalization",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = when {
                            !profile.isEnabled -> "Disabled"
                            profile.hasContent -> buildString {
                                if (profile.displayName.isNotBlank()) append(profile.displayName)
                                else append("Configured")
                                append(" · ${profile.resolvedTone.label}")
                            }
                            else -> "Tap to customize how the AI responds"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }

                // Status dot + chevron
                if (profile.isEnabled && profile.hasContent) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF4ADE80)),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Icon(
                    imageVector = Icons.Default.ChevronRight,
                    contentDescription = "Open",
                    modifier = Modifier.size(20.dp),
                    tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                )
            }
        }
    }
}
