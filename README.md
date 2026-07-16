# akilihost

Self-host coding LLMs on single-GPU servers.

## Why

Running open-weight LLMs on your own hardware means no per-token billing, no rate limits, and full control over the stack. But getting there requires wrestling with Python venvs, CUDA versions, vLLM flags, systemd services, and HuggingFace caches. akilihost automates these steps so you can focus on using the models.

## Prerequisites

- Linux with NVIDIA GPU
- CUDA 12.3+
- 24GB+ VRAM (80GB+ recommended)

## Quick Start

1. Build: `go build -o akilihost`
2. Deploy: `./akilihost serve auto`
3. Test: `curl http://localhost:8002/v1/chat/completions`

## Commands

### `serve <model>`

Start a model server. Auto-picks best quantization based on your GPU.

```bash
./akilihost serve auto                # Best model for your GPU
./akilihost serve Qwen/Qwen3-Coder-Next
./akilihost serve Qwen/Qwen3-Coder-Next --port 8003
```

### `ps`

List running models with port, VRAM usage, and health.

### `stop <target>`

Stop a model server. Target can be:

- Service name: `./akilihost stop qwen`
- Port number: `./akilihost stop 8002`
- Process ID: `./akilihost stop 12345`
- Model name (partial): `./akilihost stop Qwen3`

### `recommend`

Show models that fit your GPU with benchmarks.

### `init`

Setup Python venv, vLLM, and cache directories.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8002 | API port |
| `--gpu-memory-utilization` | 0.85 | Max GPU memory fraction |
| `--max-model-len` | 262144 | Max context tokens |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Limit model to specific GPUs |
| `HF_HOME` | HuggingFace cache directory |

---

Run `./akilihost --help` for full reference.
