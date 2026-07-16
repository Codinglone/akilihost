# ai-host

Self-host coding LLMs on single-GPU servers.

## Prerequisites

- Linux with NVIDIA GPU
- CUDA 12.3+
- 24GB+ VRAM (80GB+ recommended)

## Quick Start

1. Build: `go build -o ai-host`
2. Deploy: `./ai-host serve auto`
3. Test: `curl http://localhost:8002/v1/chat/completions`

## Commands

- `serve <model>` - Start a model server
- `ps` - List running models
- `stop <target>` - Stop a model (by service, port, or PID)
- `recommend` - Show models that fit your GPU
- `init` - Setup environment

## Flags

- `--port` - API port (default 8002)
- `--gpu-memory-utilization` - VRAM limit (default 0.85)
- `--max-model-len` - Max context length (default 262144)

## Environment

- `CUDA_VISIBLE_DEVICES` - Which GPU to use
- `HF_HOME` - HuggingFace cache directory

---

See `./ai-host --help` for details.
