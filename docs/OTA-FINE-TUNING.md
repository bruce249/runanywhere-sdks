# OTA Fine-Tuning Architecture

## Overview

The OTA (Over-The-Air) Fine-Tuning system extends the RunAnywhere SDK with the ability to:

1. **Collect** user interaction data on-device during normal chat usage
2. **Upload** that data to a cloud server (your laptop or any host)
3. **Fine-tune** LoRA adapters on the cloud using PEFT/Unsloth
4. **Deliver** trained adapters back to devices via OTA polling
5. **Apply** adapters at inference time via llama.cpp's LoRA support

This creates a personalization loop: the more a user interacts, the better the model gets at their specific use cases — all while keeping raw interaction data off third-party clouds.

## System Components

### 1. Cloud Server (`cloud-fine-tuning-server/`)

A FastAPI Python server that runs on your laptop (or any machine with a GPU).

| Component | File | Purpose |
|-----------|------|---------|
| **API Server** | `main.py` | FastAPI app with CORS, health checks, startup |
| **Training Routes** | `routes_training.py` | `/api/v1/training/*` — upload data, start jobs |
| **Adapter Routes** | `routes_adapters.py` | `/api/v1/adapters/*` — poll, list, download |
| **Dashboard Routes** | `routes_dashboard.py` | Stats, health check |
| **Preprocessing** | `preprocessing.py` | Convert interactions → training-ready format |
| **Training Engine** | `training_engine.py` | LoRA fine-tuning (PEFT or Unsloth backend) |
| **Background Worker** | `worker.py` | Async job queue processing |
| **Database** | `database.py` | SQLite via SQLAlchemy async |
| **Config** | `config.py` | Pydantic settings from env vars |
| **Schemas** | `schemas.py` | Request/response Pydantic models |

### 2. Device SDK — C API (`rac_ota_finetune.h`)

Pure C header following the existing `rac_*` vtable pattern:

- `rac_ota_init()` / `rac_ota_shutdown()` — lifecycle
- `rac_ota_upload_training_data()` — async data upload
- `rac_ota_poll_adapters()` — check for new adapters
- `rac_ota_download_adapter()` — download adapter binary
- `rac_ota_apply_adapter()` — hot-load LoRA into active LLM
- `rac_ota_list_local_adapters()` — local adapter management

### 3. Device SDK — Kotlin (`sdk/runanywhere-kotlin/.../features/ota/`)

| File | Purpose |
|------|---------|
| `OtaFineTuningConfig.kt` | Configuration (endpoint, polling, auto-upload) |
| `OtaFineTuningModels.kt` | Domain models, events, API schemas |
| `OtaFineTuningClient.kt` | Full client with upload, poll, download, apply |

### 4. Device SDK — Swift (`sdk/runanywhere-swift/.../Features/OtaFineTuning/`)

| File | Purpose |
|------|---------|
| `OtaFineTuningClient.swift` | Actor-based client matching the Kotlin API |

## Data Flow

```
Phase 1: Data Collection (Continuous)
──────────────────────────────────────
User ↔ Chat UI → TrainingDataStore → OtaFineTuningClient.recordInteraction()
                                     └→ Auto-upload when threshold reached

Phase 2: Upload (Triggered or automatic)
────────────────────────────────────────
OtaFineTuningClient.uploadTrainingData()
  → POST /api/v1/training/upload
    → Server saves interactions as JSON
    → Stores metadata in SQLite
    → Returns: upload_id, sample_count

Phase 3: Fine-Tuning (Server-side)
──────────────────────────────────
POST /api/v1/training/start
  → Worker picks up job from queue
  → DataPreprocessor: filter → weight → format (ChatML/Llama3)
  → FineTuningEngine: load base model → apply LoRA → train
    → PEFT: AutoModelForCausalLM + LoraConfig + Trainer
    → Or Unsloth: FastLanguageModel (2x faster)
  → Save adapter (safetensors)
  → Convert to GGUF (for llama.cpp)
  → Register in adapter database
  → Mark as published for OTA delivery

Phase 4: OTA Delivery (Device polls)
────────────────────────────────────
OtaFineTuningClient.pollForAdapters()
  → GET /api/v1/adapters/poll?device_id=X&since=T
  → If new adapters: auto-download
  → GET /api/v1/adapters/download/{id}
    → Stream binary file to device
    → Verify SHA-256 checksum
    → Store locally in adapter registry
  → Auto-apply to active LLM model

Phase 5: Personalized Inference
───────────────────────────────
llama.cpp loads base GGUF model + LoRA adapter
  → Adapter modifies q_proj, v_proj, k_proj, o_proj
  → User gets personalized responses
  → New interactions feed back to Phase 1
```

## API Reference

### Training Data

```
POST /api/v1/training/upload
Body: {
  "device_id": "string",
  "model_id": "string",
  "interactions": [
    {
      "id": "string",
      "user_prompt": "string",
      "model_response": "string",
      "corrected_response": "string | null",
      "rating": "int | null",
      "is_positive_example": false,
      "include_in_training": true,
      "category": "string | null",
      "timestamp": 1234567890
    }
  ]
}
Response: { "upload_id": "...", "sample_count": 42, "total_samples_for_device": 150 }

POST /api/v1/training/start
Body: { "device_id": "...", "model_id": "...", "base_model": "unsloth/Llama-3.2-1B" }
Response: { "job_id": "...", "status": "pending" }

GET /api/v1/training/status/{job_id}
Response: { "status": "training", "progress": 0.45, "current_loss": 1.23 }
```

### Adapter OTA

```
GET /api/v1/adapters/poll?device_id=X&since=2025-01-01T00:00:00Z
Response: { "has_new_adapters": true, "adapters": [...] }

GET /api/v1/adapters/download/{id}
Response: Binary file stream (GGUF or safetensors)
Headers: X-Adapter-Checksum, X-LoRA-Rank, X-LoRA-Alpha

GET /api/v1/adapters/list?device_id=X
DELETE /api/v1/adapters/{id}
```

## Running Locally

```bash
# Start the server on your laptop
cd cloud-fine-tuning-server
pip install -e .
python -m runanywhere_finetune.main

# Server runs at http://0.0.0.0:8000
# Find your laptop's IP: ipconfig (Windows) / ifconfig (Mac/Linux)
# Use that IP in the device config
```

## Quality Signals for Training

The system uses smart sample weighting based on quality signals:

| Signal | Weight Boost | Rationale |
|--------|-------------|-----------|
| User correction provided | +2x | Explicit preference signal |
| Rating ≥ 4 stars | +1x | High-quality interaction |
| Marked as positive | +1x | Explicit endorsement |
| Frequency > 3 | +1x | Common pattern, model should learn |
| Max weight cap | 5x | Prevent single sample domination |

## Security Considerations

- **Local-only by default**: Server binds to your laptop, data stays on your network
- **API key auth**: Simple shared secret for device ↔ laptop communication
- **Checksum verification**: SHA-256 integrity check on downloaded adapters
- **No third-party cloud**: Training data never leaves your infrastructure
