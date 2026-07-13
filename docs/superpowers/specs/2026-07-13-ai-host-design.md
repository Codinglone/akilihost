# ai-host: Self-Hosted AI Model Manager

## Overview

`ai-host` is a Go CLI + daemon that automates everything painful about running open-weight LLMs on a single GPU: CUDA version detection, vLLM version selection, venv setup, cache management, VRAM math, quantization selection, and multi-model lifecycle.

Inspired by real pain from self-hosting Qwen3-Coder-Next, Qwen2.5-Coder, and Devstral 2 on an H200 (139 GiB VRAM) VM.

## Architecture

```
ai-host
├── cli/          # CLI + TUI frontend (cobra + bubbletea)
├── host/         # Core engine (GPU detection, model DB, process mgmt)
└── api/          # HTTP API for daemon mode
```

Single Go binary. Only runtime dependency is a Python venv (for vLLM serving).

## CLI Commands

| Command | Description |
|---------|-------------|
| `ai-host init` | Detect GPU, install Python venv + vLLM, setup caches |
| `ai-host serve <model>` | Start serving a model (auto-picks best quantization) |
| `ai-host ps` | List running models (port, VRAM, health) |
| `ai-host stop <model\|port>` | Stop a model server |
| `ai-host recommend` | Show models that fit your GPU, with benchmarks |
| `ai-host tui` | TUI dashboard (running models, GPU stats) |
| `ai-host daemon` | Start background daemon for multi-model mgmt |

## Components

### CLI Layer

- **cobra** for subcommands
- **bubbletea** for TUI
- Wraps `host` package for all business logic

### host package (Core Engine)

#### GPU Detector
- Parse `nvidia-smi` output (or driver fallback) for:
  - GPU model name
  - Total VRAM
  - CUDA driver version
- Map CUDA version to compatible vLLM range:
  - CUDA 12.8 → vLLM ≤ 0.18.0 (PyTorch cu128 index)
  - CUDA 13.0 → vLLM ≥ 0.21.0
- Map GPU model to flash-attention backend support

#### Venv Manager
- Create Python venv at configurable path
- Install correct PyTorch (cu118/cu121/cu124/cu128 index)
- Install correct vLLM version
- Install flashinfer if applicable
- Install mistral-common for Mistral models

#### Cache Manager
- Detect largest available disk
- Create/symlink: HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE, vLLM model cache
- FlashInfer JIT workspace symlink

#### Model Database (embedded YAML)
Curated models with:
- HF repo ID
- Architecture (dense, moe)
- Total params, active params (for MoE)
- Context window
- License
- Benchmarks: SWE-bench Verified, HumanEval, LiveBench
- Quantization options with VRAM requirements and dtype flags
- Special flags (tool-call-parser, etc.)

#### Model Sizer
- Given model + quantization, compute VRAM usage including KV cache overhead
- Cross-reference against detected GPU VRAM
- Recommend best quantization that fits

#### Process Manager
- Start vLLM server as subprocess with correct flags
- Health check polling (GET /health)
- Graceful shutdown (SIGTERM → SIGKILL timeout)
- Auto-restart on crash (configurable)
- Track PID, port, VRAM allocation per model

### API Layer (Daemon)

- HTTP server on port 9500
- Endpoints:
  - POST /models/{name} — start serving
  - DELETE /models/{name} — stop
  - GET /models — list running
  - GET /gpu/stats — GPU utilization
  - GET /v1/chat/completions — route to active model (optional proxy)

### TUI

- Running model list: name, port, VRAM, uptime, health status
- GPU stats panel: utilization %, temperature, free VRAM
- Live log tail from selected model
- Keyboard navigation
- Built with bubbletea + lipgloss

## Data Flow

```
init:
  GPU Detector → Venv Manager → Cache Manager → done

serve:
  Model DB lookup → Model Sizer (VRAM check) → Process Manager start → health check → ready

recommend:
  GPU Detector → Model DB (filter by VRAM) → sorted benchmark display

tui:
  Process Manager (running list) + GPU Detector stats → real-time dashboard
```

## Non-Goals

- Model training or fine-tuning
- Multi-node / multi-GPU orchestration (out of scope for MVP)
- API proxy with auth/rate-limiting (bare vLLM API is sufficient)
- Docker images (venv is simpler for GPU pass-through)

## MVP Scope (v0.1)

1. `ai-host init` — GPU detection + venv setup + cache setup
2. `ai-host serve <model>` — start model with auto-flags
3. `ai-host ps` — list running models
4. `ai-host stop` — stop a model
5. `ai-host recommend` — model browser
6. Model DB with 10-15 curated models (Qwen3-Coder-Next, Qwen2.5-Coder, Devstral 2, Llama 4 Scout, DeepSeek variants)
7. Process health monitoring

## Future (Post-MVP)

- Daemon mode + TUI
- Auto-restart on crash
- Multi-quantization support in model DB
- Community model contributions via PR
- `ai-host logs` command
