# akilihost

One binary to run coding LLMs on a single GPU — no Python env, no vLLM flags to memorize, no manual VRAM math.

`akilihost` picks the right model and backend for your hardware, downloads it, and runs it as a systemd service behind an OpenAI-compatible API. Tested on a T4 (16GB) and A100 (80GB).

## How it works

- **vLLM** for models that fit fully in VRAM (fast, full GPU)
- **llama.cpp** with split GPU/CPU offload for models that don't (e.g. 27B on a 16GB T4 — 14GB on GPU, rest on CPU)
- Auto-fallback: if a model needs more VRAM than you have, it runs on llama.cpp instead of failing

## Models

| Model | Size | Context | Backend | Best on |
|-------|------|---------|---------|---------|
| `Qwen3.8-27B` (unsloth GGUF) — `UD-Q4_K_XL` / `UD-Q3_K_XL` | 17GB / 13GB | 262K (configurable) | llama.cpp | T4 16GB |
| `Qwen2.5-Coder-32B` — `BF16` | 64GB | 131K | vLLM | A100 80GB |
| `Qwen3-Coder-Next` (80B MoE) — `FP8` | 75GB | 131K | vLLM | A100 80GB |
| `Devstral 2 123B` — `FP8` | 119GB | 131K | vLLM | 2×GPU or large CPU RAM |

Sizing is estimated from real architecture params (layers, KV heads, head dim, bits/weight) — the `≈Size` column in `recommend` is an estimate, the fit decision uses the curated `MinVRAM` value.

## Prerequisites

- Linux with NVIDIA GPU and driver (`nvidia-smi` works)
- `systemd` (for `serve`/`stop`/`ps`)
- `huggingface_hub` CLI (`pip install huggingface_hub[cli]`) — only needed the first time a GGUF is downloaded
- `llama-server` in `PATH` — built by `akilihost init`, or install from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

A T4 with 16GB VRAM + 28GB RAM is enough for Qwen3.8-27B Q4.

## Quick start

```bash
go build -o akilihost
./akilihost recommend          # what fits on this GPU?
./akilihost serve Qwen3.8-27B  # or: ./akilihost serve auto
./akilihost ps                 # is it running?
curl http://localhost:8002/v1/models | jq
```

## Commands

### `akilihost recommend`

What fits on the current machine, with estimated size and headroom.

```bash
$ akilihost recommend
GPU: Tesla T4 (16384 MB VRAM, 28015 MB RAM)

Model                     Quant        Backend         ≈Size   Headroom
------------------------------------------------------------------------
Qwen3.8-27B               UD-Q4_K_XL   llama-cpp     17043MB    20331MB
Qwen3.8-27B               UD-Q3_K_XL   llama-cpp     13825MB    24427MB

Recommended: Qwen3.8-27B UD-Q4_K_XL (16.6 GB estimated)
```

Context length affects KV cache — try `--max-model-len 8192` vs `131072` to see the difference.

### `akilihost serve <model|auto>`

Starts the model as a systemd service. Downloads the GGUF on first run if needed. Verifies the API is up before returning.

```bash
akilihost serve auto                      # best fit for this GPU
akilihost serve Qwen3.8-27B               # specific model, default quant
akilihost serve Qwen3.8-27B --port 8003   # custom port
akilihost serve Qwen3.8-27B --max-model-len 8192   # smaller context = less VRAM
```

Flags:

| Flag | Default | What it does |
|------|---------|--------------|
| `--port` | `8002` | API port. If the default is busy it auto-increments (8003, 8004…). If you set it explicitly and it's busy, you get an error. |
| `--gpu-memory-utilization` | `0.90` | Passed to vLLM as `--gpu-memory-utilization` |
| `--max-model-len` | `32768` | Passed to vLLM as `--max-model-len` and to llama.cpp as `--ctx-size`. Also used for the sizing estimate. |

Example with a busy port:

```bash
$ akilihost serve Qwen3.8-27B --port 8002
Error: port 8002 is already in use

$ akilihost serve Qwen3.8-27B   # no --port → auto-increments to 8003
Port: 8003
```

### `akilihost ps`

Shows systemd services and raw processes, plus health and VRAM.

```bash
$ akilihost ps
Systemd Services:
-----------------
   Active: active (running) since Sat 2026-08-29 18:44:57 UTC; 1min ago
   Main PID: 28962 (llama-server)
  Model: unsloth/Qwen3.8-27B-GGUF
  Backend: llama-cpp
  Port: 8002
    Health: ok (model: /opt/akilihost/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf)
  VRAM: 14115 / 16384 MiB

Running Processes:
------------------
  PID 28962: llama-server
    Port: 8002
    Health: ok (model: /opt/akilihost/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf)
```

### `akilihost stop <target>`

Stops a service. `target` can be a partial model name or a port.

```bash
akilihost stop qwen        # matches Qwen3.8-27B → stops akilihost-Qwen3.8-27B
akilihost stop 8002        # kills whatever is on port 8002
akilihost stop devstral    # matches Devstral 2 123B
```

### `akilihost init`

One-time setup on a fresh VM: builds `llama-server` with CUDA, installs `huggingface_hub`, creates `/opt/akilihost/models`.

```bash
akilihost init
```

## Using with opencode

The model speaks OpenAI API on `http://localhost:8002/v1`. This repo already has `~/.config/opencode/opencode.json` wired up:

```json
{
  "provider": {
    "selfhosted": {
      "options": { "baseURL": "http://localhost:8002/v1" },
      "models": {
        "unsloth/Qwen3.8-27B-GGUF": { "name": "Qwen3.8-27B UD-Q4_K_XL (T4 split)" }
      }
    }
  },
  "model": "selfhosted/unsloth/Qwen3.8-27B-GGUF"
}
```

**If the model runs on a remote server** (`api-product-dev`), open a tunnel first:

```bash
# from your laptop
AI_HOST_SERVER=api-product-dev ./scripts/ai-host-tunnel.sh start
./scripts/ai-host-tunnel.sh status   # should print Model: .../Qwen3.8-27B...
curl -s http://localhost:8002/v1/models | jq

opencode   # already set to selfhosted/unsloth/Qwen3.8-27B-GGUF
```

Stop the tunnel with `./scripts/ai-host-tunnel.sh stop`. The script auto-reconnects if the SSH connection drops.

**Direct test without opencode:**

```bash
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/opt/akilihost/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf","messages":[{"role":"user","content":"Write a Python factorial function. Just code."}],"max_tokens":100}' | jq
```

## Project layout

```
host/           # pure logic — sizing, backend selection, GPU detection (unit tested)
  models.go     # curated model DB (in-memory, not YAML)
  sizers.go     # real VRAM math from architecture params
  backend.go    # SelectBackend + ServiceName
  llamacpp.go   # llama-server command builder
  gpu.go        # nvidia-smi + /proc/meminfo parsing
cli/            # cobra commands — serve, ps, stop, recommend, init
scripts/ai-host-tunnel.sh
```

## Development

```bash
go vet ./...
go test ./... -count=1   # 30 tests (21 host + 9 cli)
go build -o akilihost
```

No `models.yaml` — the DB is `host/prepopulatedModels`. No committed binary — `bin/` is gitignored.
