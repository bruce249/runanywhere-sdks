# RunAnywhere OTA Fine-Tuning Cloud Server

> Turn your laptop into a fine-tuning cloud. Collect on-device interaction data, fine-tune LoRA adapters, and deliver them OTA to your devices.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  📱 Device (Android/iOS)                                │
│    Chat UI → Collect Data → Upload to Cloud             │
│    Poll Adapters → Download → Hot-load LoRA             │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/WiFi
┌────────────────────────▼────────────────────────────────┐
│  ☁️ Your Laptop (This Server)                           │
│                                                         │
│  FastAPI Server (:8000)                                 │
│  ├── /api/v1/training/upload     ← receive device data  │
│  ├── /api/v1/training/start      ← kick off fine-tuning │
│  ├── /api/v1/training/status/:id ← job progress         │
│  ├── /api/v1/adapters/poll       ← device polls for new │
│  ├── /api/v1/adapters/list       ← list all adapters    │
│  └── /api/v1/adapters/download   ← download .gguf       │
│                                                         │
│  Fine-Tuning Engine (PEFT/Unsloth + LoRA)               │
│  ├── Data preprocessor                                  │
│  ├── LoRA trainer (rank=8, α=16)                        │
│  ├── GGUF converter (for llama.cpp)                     │
│  └── Adapter registry (SQLite)                          │
│                                                         │
│  Storage: ./data/                                       │
│  ├── training_data/   ← uploaded interaction data       │
│  ├── adapters/        ← trained .gguf adapter files     │
│  ├── checkpoints/     ← training checkpoints            │
│  └── finetune.db      ← metadata (SQLite)               │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure (optional - defaults work for local dev)
cp .env.example .env

# 3. Run the server
python -m runanywhere_finetune.main

# Server starts at http://0.0.0.0:8000
# Docs at http://0.0.0.0:8000/docs
```

## API Endpoints

### Training Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/training/upload` | Upload training data from device |
| POST | `/api/v1/training/start` | Start a fine-tuning job |
| GET | `/api/v1/training/status/{job_id}` | Check training job status |
| GET | `/api/v1/training/jobs` | List all training jobs |

### Adapter Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/adapters/poll` | Poll for new adapters (device calls this) |
| GET | `/api/v1/adapters/list` | List all trained adapters |
| GET | `/api/v1/adapters/download/{id}` | Download adapter file |
| DELETE | `/api/v1/adapters/{id}` | Delete an adapter |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard` | Web dashboard for monitoring |

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | Server port |
| `DATA_DIR` | `./data` | Data storage directory |
| `BASE_MODEL` | `unsloth/Llama-3.2-1B` | HuggingFace model for fine-tuning |
| `LORA_RANK` | `8` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha |
| `MAX_SEQ_LENGTH` | `512` | Maximum sequence length |
| `TRAINING_EPOCHS` | `3` | Default training epochs |
| `AUTO_CONVERT_GGUF` | `true` | Auto-convert adapters to GGUF |

## Device Integration

Point your RunAnywhere SDK to this server:

```kotlin
// Android / Kotlin
RunAnywhere.configure {
    fineTuning {
        cloudEndpoint = "http://<YOUR_LAPTOP_IP>:8000"
        autoUpload = true
        pollInterval = 30.minutes
    }
}
```

```swift
// iOS / Swift
RunAnywhere.configure {
    $0.fineTuning.cloudEndpoint = "http://<YOUR_LAPTOP_IP>:8000"
    $0.fineTuning.autoUpload = true
    $0.fineTuning.pollInterval = .minutes(30)
}
```
