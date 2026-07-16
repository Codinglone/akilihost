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

### `serve <model>`

Start a model server. Auto-picks best quantization based on your GPU.

```bash
./ai-host serve auto                # Best model for your GPU
./ai-host serve Qwen/Qwen3-Coder-Next
./ai-host serve Qwen/Qwen3-Coder-Next --port 8003
```

### `ps`

List running models with port, VRAM usage, and health.

### `stop <target>`

Stop a model server. Target can be:

- Service name: `./ai-host stop qwen`
- Port number: `./ai-host stop 8002`
- Process ID: `./ai-host stop 12345`
- Model name (partial): `./ai-host stop Qwen3`

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
| `CUDA_VISIBLE_DEVICES` | GPU selection (e.g., `3` for GPU 3) |
| `HF_HOME` | HuggingFace cache directory |

---

Run `./ai-host --help` for full reference.
