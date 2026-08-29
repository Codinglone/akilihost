# akilihost: llama.cpp Backend for Split GPU/CPU Inference

**Date:** 2026-08-28
**Status:** Approved
**Target VM:** `api-product-dev` (Standard_NC4as_T4_v3, 1× T4 16GB VRAM, 27GB RAM, westus2)

## Problem

akilihost currently only supports vLLM as its inference backend. vLLM cannot offload to
system RAM — a model must fit entirely in VRAM. This blocks the primary use case:
running Qwen3.8-27B (Unsloth dynamic 4-bit GGUF, ~17GB) on a single T4 with only 16GB
VRAM. The model fits when split across GPU + CPU RAM, but vLLM has no mechanism for that.

llama.cpp's `llama-server` supports partial GPU offload via `--n-gpu-layers`, spilling
remaining layers to CPU RAM. This makes 27B-class models viable on 16GB GPUs.

## Goal

Extend akilihost with a llama.cpp backend so that:

1. `akilihost serve Qwen3.8-27B` starts llama-server with GPU/CPU split inference
2. `akilihost stop Qwen3.8-27B` stops the server and frees all GPU VRAM + RAM
3. `akilihost ps` shows running models with VRAM usage and health
4. The user can switch the shared T4 VM between the model and other GPU tasks instantly
5. opencode connects to the model via an SSH tunnel to `localhost:8002`

## Design

### Architecture

```
opencode (laptop)
   │  POST /v1/chat/completions → localhost:8002
   ▼
SSH tunnel (port 22, existing key, Host model-tunnel)
   ▼
api-product-dev VM (T4, westus2)
   │  llama-server bound to 127.0.0.1:8002 (OpenAI-compatible /v1, not public)
   ▼
GPU (T4 16GB) + CPU RAM (27GB) split inference
```

The model API is never exposed publicly — only reachable through the SSH tunnel.
This matches the existing `opencode.json` which points at `http://localhost:8002/v1`.

### Backend Selection

A new `backend` field in `models.yaml` selects the inference engine:

- `vllm` (default, existing behavior) — for high-VRAM GPUs where the model fits entirely
- `llama-cpp` — for split GPU/CPU inference when `min_vram_mb` exceeds available VRAM

`cmd_serve.go` logic:
1. Detect GPU VRAM via `host.DetectGPU()` (existing)
2. Load model + quantization from `models.yaml` (existing)
3. If `backend == "llama-cpp"` OR `quant.min_vram_mb > available VRAM` → use llama-server
4. Otherwise → use vLLM (existing path, unchanged)

This keeps vLLM working for big GPUs and adds llama.cpp for small-GPU split inference.

### models.yaml — New Entry

```yaml
- repo_id: unsloth/Qwen3.8-27B-GGUF
  name: Qwen3.8-27B
  description: "Qwen3.8 27B with vision + reasoning, 256K context"
  architecture: dense
  total_params: "27B"
  active_params: "27B"
  context_tokens: 262144
  license: "Apache 2.0"
  backend: llama-cpp
  benchmarks:
    HumanEval: 90.3
    SWE-bench: 61.7
    LiveCodeBench: 90.3
  quantizations:
    - name: UD-Q4_K_XL
      description: "Unsloth Dynamic 4-bit (~17GB, GPU+CPU split)"
      min_vram_mb: 17408
      file_pattern: "*UD-Q4_K_XL*"
      flags:
        - "--n-gpu-layers"
        - "auto"
    - name: UD-Q3_K_XL
      description: "Unsloth Dynamic 3-bit (~13GB, fits T4 fully)"
      min_vram_mb: 13312
      file_pattern: "*UD-Q3_K_XL*"
      flags:
        - "--n-gpu-layers"
        - "999"
```

The `file_pattern` and `backend` fields are new. The `flags` for llama-cpp entries are
llama-server flags (not vLLM flags). `--n-gpu-layers auto` lets llama.cpp auto-detect
how many layers fit in VRAM; `999` forces all layers to GPU (for quants that fit).

### Code Changes

#### `host/models.go`
- Add `Backend string` field to `Model` struct (`yaml:"backend"`)
- Add `FilePattern string` field to `Quantization` struct (`yaml:"file_pattern"`)
- Add Qwen3.8-27B entry to `prepopulatedModels`
- Default `Backend` to `"vllm"` when empty (backward compat)

#### `host/gpu.go`
- No changes (GPU detection already works)

#### `host/sizers.go`
- `FindFit`: when `backend == "llama-cpp"`, a model is considered "fitting" if
  `min_vram_mb <= total_memory (VRAM + RAM)`, not just VRAM. This allows the sizer to
  recommend split-inference models on small GPUs.
- Add a `SystemRAMMB` field to `GPUInfo` (detect via `/proc/meminfo` or `free`)

#### `cli/cmd_init.go`
- Add llama.cpp build step (when GPU detected and CUDA available):
  - `git clone https://github.com/ggml-org/llama.cpp`
  - `cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF`
  - `cmake --build build --target llama-server`
  - Copy `llama-server` to `/usr/local/bin/`
- Install `huggingface_hub` (pip) for GGUF downloads
- Create `/opt/akilihost/models/` directory
- Keep existing vLLM init steps (run conditionally based on backend / VRAM)

#### `cli/cmd_serve.go`
- Backend selection logic (see "Backend Selection" above)
- For `llama-cpp` backend:
  - Resolve GGUF file path in `/opt/akilihost/models/`
  - If not present, run `hf download <repo_id> --include "<file_pattern>" --local-dir /opt/akilihost/models/<name>`
  - Build llama-server command:
    ```
    llama-server \
      --model /opt/akilihost/models/Qwen3.8-27B/<gguf-file> \
      --host 127.0.0.1 \
      --port 8002 \
      --cache-type-k q8_0 \
      --cache-type-v q8_0 \
      --ctx-size 32768 \
      <quant.flags...>
    ```
  - Bind to `127.0.0.1` (not `0.0.0.0`) — only SSH tunnel access
  - Create systemd service named `akilihost-<model>` (not `vllm-<model>`)
- For `vllm` backend: existing behavior unchanged
- `waitAndVerify`: unchanged (curl `/v1/models` works for both backends)

#### `cli/cmd_stop.go`
- Update `getSystemdServiceName` to handle `akilihost-` prefix
- Also match by partial model name for llama.cpp services
- `stopByModelName`: also search for `llama-server` processes (not just `vllm serve`)

#### `cli/cmd_ps.go`
- Show both `vllm` and `akilihost-` systemd services
- For llama-server: query `/v1/models` and `nvidia-smi` for VRAM usage

### systemd Service

Service name: `akilihost-<model-name>.service` (e.g. `akilihost-Qwen3.8-27B.service`)

```ini
[Unit]
Description=akilihost llama-server: <model>
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStart=/usr/local/bin/llama-server <flags>
ExecStop=/bin/kill -TERM $MAINPID

[Install]
WantedBy=multi-user.target
```

Services are **not** enabled on boot (the VM is shared). The user starts them on demand
with `akilihost serve`. This keeps the GPU free for other tasks until the model is needed.

### VM Provisioning

#### Disk resize: 29GB → 256GB
- `az vm deallocate -g PRODUCT-DEV -n api-product-dev`
- `az disk update --resource-group PRODUCT-DEV --name <os-disk> --size-gb 256`
- `az vm start -g PRODUCT-DEV -n api-product-dev`
- `sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1` (on VM)
- Gives room for multiple GGUFs (Q4 ~17GB + Q3 ~13GB) plus other VM tasks

#### CUDA toolkit
- Install `cuda-toolkit` from the NVIDIA repo already added during driver setup
- Provides `nvcc` needed to build llama.cpp with `-DGGML_CUDA=ON`

#### llama.cpp
- Built by `akilihost init` on the VM (first run)
- CUDA-enabled `llama-server` binary at `/usr/local/bin/llama-server`

### opencode Integration (Local Laptop)

#### `~/.ssh/config` — new tunnel host
```
Host model-tunnel
    HostName api-product-dev.westus2.cloudapp.azure.com
    User du-admin
    IdentityFile /home/codinglone/Documents/DU/devops/keys/api-product-dev_key.pem
    IdentitiesOnly yes
    LocalForward 8002 127.0.0.1:8002
```

Usage: `ssh -N model-tunnel` (background), then opencode reaches `localhost:8002`.

#### `~/.config/opencode/opencode.json` — add Qwen3.8-27B
```json
{
  "provider": {
    "selfhosted": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Self-Hosted (llama.cpp)",
      "options": {
        "baseURL": "http://localhost:8002/v1",
        "timeout": 600000,
        "chunkTimeout": 120000
      },
      "models": {
        "unsloth/Qwen3.8-27B-GGUF": {
          "name": "Qwen3.8-27B UD-Q4_K_XL (T4 split)"
        }
      }
    }
  },
  "model": "selfhosted/unsloth/Qwen3.8-27B-GGUF"
}
```

### Verification

1. `akilihost init` on VM → llama-server binary exists, `huggingface_hub` installed
2. `akilihost serve Qwen3.8-27B` → GGUF downloads, llama-server starts, `curl localhost:8002/v1/models` returns model ID
3. `akilihost ps` → shows Qwen3.8-27B running, VRAM usage, port 8002
4. `nvidia-smi` → shows ~13-14GB VRAM used (GPU layers), rest in RAM
5. `akilihost stop Qwen3.8-27B` → service stops, `nvidia-smi` shows VRAM freed
6. `ssh -N model-tunnel` from laptop → `curl localhost:8002/v1/models` works through tunnel
7. Launch opencode → send a coding prompt → response streams back

### Tradeoffs / Notes

- Q4_K_XL spills ~3GB to CPU → expect ~3-5 tok/s. Acceptable for coding agent use.
  If too slow, swap to UD-Q3_K_XL (~13GB, fits fully in T4, ~8-12 tok/s) — one line
  change in the serve command or models.yaml.
- Qwen3.8-27B is multimodal (vision). Vision requires `llama-mtmd-cli`, out of scope for
  opencode text coding. Can add later.
- The T4 is Turing architecture (compute 7.5). llama.cpp CUDA builds support this.
  NVFP4 quants require Blackwell — not applicable here, GGUF is the right choice.
- Disk resize to 256GB adds ~$12/mo (covered by startup credits).
