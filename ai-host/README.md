# ai-host

Production-Grade AI Model Hosting for Single-GPU Servers

Automates the entire lifecycle of open-weight language models: detection, quantization, deployment, scaling, and monitoring.

[![Go Version](https://img.shields.io/github/go-mod/go-version/codinglone/ai-host?logo=go)](https://go.dev/)
[![License](https://img.shields.io/github/license/codinglone/ai-host?color=blue)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/codinglone/ai-host?logo=git)](https://github.com/codinglone/ai-host/commits)
[![GitHub Stars](https://img.shields.io/github/stars/codinglone/ai-host?style=social)](https://github.com/codinglone/ai-host)

---

## ⭐ Features

| Feature | Description |
|---------|-------------|
| **Smart GPU Detection** | Parses `nvidia-smi` for VRAM, CUDA version, compute capability |
| **Quantization Optimization** | Automatically selects FP8/BF16/GGUF based on available GPU memory |
| **One-Command Deploy** | `ai-host serve <model>` handles venv, systemd, caching |
| **Self-Healing** | Systemd service with `Restart=always`, OOM protection, crash recovery |
| **Resilient Tunneling** | Optional `autossh` integration for secure remote access |
| **Model Database** | Curated LLMs with benchmarks (HumanEval, SWE-bench, LiveBench) |
| **CLI + TUI** | Command-line tools and terminal dashboard |

### Supported Model Architectures

- **Dense Models**: Qwen2.5-Coder, Llama 3, Mistral, Mixtral, Phi-3
- **MoE Models**: Qwen3-Coder, DeepSeek V3, Mixtral 8x7B
- **Quantization**: FP8, FP16/BF16, GGUF (planned)

---

## 📊 Supported Models

| Model | Parameters | Architecture | HumanEval | SWE-bench | Recommended Quant | VRAM Required |
|-------|------------|--------------|-----------|-----------|-------------------|---------------|
| **Qwen3-Coder-Next** | 80B (MoE) | Mixture of Experts | **91.3%** | 55.4% | FP8 | ~75 GB |
| **Qwen2.5-Coder-32B** | 32B | Dense | **92.7%** | 51.4% | BF16 | ~64 GB |
| **Devstral-2-123B** | 123B | Dense | 86.5% | **58.2%** | FP8 | ~119 GB |
| **DeepSeek V3** | 671B (MoE) | MoE (24 experts) | 92.0% | 64.0% | FP8 | ~220 GB |
| **Llama 3.1 70B** | 70B | Dense | 89.0% | 48.0% | FP8 | ~140 GB |
| **Mistral Large 2** | 123B | Dense | 85.3% | 53.2% | FP8 | ~125 GB |
| **Phi-3 Medium** | 14B | Dense | 81.0% | 42.0% | BF16 | ~28 GB |

*Benchmarks sourced from [OpenCompass](https://github.com/open-compass/OpenCompass) and [HuggingFace Hub](https://huggingface.co/models)*

---

## 📦 Prerequisites

### Server Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 24 GB VRAM | 80+ GB (H200/A100/H100/MI300X) |
| **CUDA** | 12.3+ | 12.4+ (for FlashInfer) |
| **Driver** | 535+ | 550+ |
| **RAM** | 16 GB | 64+ GB |
| **Disk** | 50 GB free | 500+ GB NVMe |
| **Network** | 1 Gbps | 10+ Gbps |

### Supported Operating Systems

- **Ubuntu**: 22.04 LTS, 24.04 LTS
- **Debian**: 11 (Bullseye), 12 (Bookworm)
- **Fedora**: 40+, 41+
- **Rocky Linux**: 8+, 9+
- **Amazon Linux**: 2023+

### Local Development

| Tool | Purpose |
|------|---------|
| Go 1.22+ | Build the CLI |
| `autossh` | Persistent SSH tunneling (optional) |
| `jq` | JSON parsing (optional, for curl testing) |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Build

```bash
cd /path/to/ai-host
go build -o ai-host
```

### Step 2: Deploy to Server

Replace `ubuntu@myserver` with your actual server details.

```bash
# Create directory structure on server
ssh ubuntu@myserver "mkdir -p /home/ubuntu/ai-host/bin /mnt/volume_3mx7vkr/hf-cache"

# Copy binary
scp ai-host ubuntu@myserver:/home/ubuntu/ai-host/bin/ai-host

# Copy systemd service
scp systemd/vllm-qwen.service ubuntu@myserver:/tmp/vllm-qwen.service

# Install on server
ssh ubuntu@myserver << 'EOF'
#!/bin/bash
set -euo pipefail

# Copy and enable service
sudo cp /tmp/vllm-qwen.service /etc/systemd/system/vllm-qwen.service
sudo systemctl daemon-reload
sudo systemctl enable vllm-qwen

# Start the model
sudo systemctl start vllm-qwen

# Check status
sleep 5
systemctl is-active vllm-qwen
EOF
```

### Step 3: Verify Deployment

```bash
# Check systemd service
ssh ubuntu@myserver "systemctl status vllm-qwen --no-pager"

# Test API endpoint
curl -s http://localhost:8002/v1/models | jq .

# Quick inference test
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Coder-Next","messages":[{"role":"user","content":"2+2"}],"max_tokens":20}' | jq
```

### Step 4: Set Up Persistent Tunnel (Optional)

**Local machine (your workstation):**

```bash
# Install autossh
sudo apt-get install autossh    # Ubuntu/Debian
sudo dnf install autossh        # Fedora/RHEL

# Create user systemd service
mkdir -p ~/.config/systemd/user/

cat > ~/.config/systemd/user/ai-host-tunnel.service << 'EOF'
[Unit]
Description=ai-host SSH Tunnel to self-hosted model server
Documentation=https://github.com/codinglone/ai-host
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStart=/usr/bin/autossh -M 0 \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -N -L 8002:localhost:8002 myserver

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user daemon-reload
systemctl --user enable ai-host-tunnel
systemctl --user start ai-host-tunnel

# Verify
systemctl --user status ai-host-tunnel
```

**Test the tunnel:**
```bash
curl -s http://localhost:8002/v1/models | jq .
```

---

## 📖 CLI Reference

### `ai-host serve <model|auto>`

Start serving a model with optimal quantization.

```bash
# Auto-select best model for your GPU
./ai-host serve auto

# Serve specific model by repo ID
./ai-host serve Qwen/Qwen3-Coder-Next

# Serve with custom port
./ai-host serve Qwen/Qwen2.5-Coder-32B --port 8003

# Dry run (show config without starting)
./ai-host serve auto --dry-run
```

**What it does:**
1. Detects GPU via `nvidia-smi`
2. Loads model database from `host/models.go`
3. Calculates VRAM requirements
4. Selects optimal quantization
5. Generates systemd service with correct flags
6. Starts model and verifies API endpoint

---

### `ai-host ps`

List all running models with health status.

```
Systemd Services:
-----------------
  ✓ vllm-qwen.service
    Model: Qwen/Qwen3-Coder-Next
    Port: 8002
    Status: active (running)
    GPU Usage: 74.8 GB / 141.3 GB (53%)
    Uptime: 1h 23m 45s

Running Processes:
------------------
  PID 2027453: vLLM server
    Model: Qwen/Qwen3-Coder-Next
    Port: 8002
    Status: healthy
    
  PID 2028100: vLLM server  
    Model: Qwen/Qwen2.5-Coder-32B
    Port: 8003
    Status: healthy
```

---

### `ai-host stop <target>`

Stop a running model server.

```bash
# By systemd service name
./ai-host stop qwen
./ai-host stop devstral

# By port number
./ai-host stop 8002

# By process ID
./ai-host stop 2027453

# By model name (partial match)
./ai-host stop "Qwen3-Coder"
```

---

### `ai-host recommend`

Show models that fit your GPU with detailed specifications.

```
=== GPU Information ===
Model:       NVIDIA H200
VRAM:        141,312 MB (138.1 GB)
CUDA:        12.8
Compute Cap: 9.0

=== Recommended Models (Fits within 85% VRAM) ===

1. Qwen3-Coder-Next FP8
   └─ VRAM: 75.6 GB / 120.1 GB (63%)
   └─ Context: 131,072 tokens
   └─ HumanEval: 91.3%
   └─ SWE-bench: 55.4%
   └─ Description: Qwen's latest large code model with MoE architecture
   └─ Args: --dtype bfloat16 --quantization fp8

2. Qwen2.5-Coder-32B BF16
   └─ VRAM: 64.0 GB / 120.1 GB (53%)
   └─ Context: 131,072 tokens  
   └─ HumanEval: 92.7%
   └─ SWE-bench: 51.4%
   └─ Description: Strong coding performance, efficient 32B model
   └─ Args: --dtype bfloat16

3. Devstral-2-123B FP8
   └─ VRAM: 119.0 GB / 120.1 GB (99%)
   └─ Context: 131,072 tokens
   └─ HumanEval: 86.5%
   └─ SWE-bench: 58.2%
   └─ Description: Mistral's powerful reasoning model
   └─ Args: --dtype bfloat16 --quantization fp8
```

---

### `ai-host init`

Initialize GPU environment (venv, dependencies, cache symlinks).

```bash
# Quick init (uses defaults)
./ai-host init

# Custom setup
./ai-host init \
  --venv /mnt/data/venvs/vllm \
  --hf-cache /mnt/data/cache/hf \
  --vllm-version 0.18.0 \
  --torch-version 2.10.0
```

**Flags:**
- `--venv <path>`: Python virtual environment path
- `--hf-cache <path>`: HuggingFace cache directory
- `--vllm-version <version>`: vLLM version (default: 0.18.0)
- `--torch-version <version>`: PyTorch version (default: 2.10.0)
- `--force`: Reinstall even if venv exists

---

### `ai-host daemon`

Start background daemon for multi-model management.

```bash
# Start daemon on default port 8000
./ai-host daemon

# Custom port
./ai-host daemon --port 8080

# Custom config
./ai-host daemon --config ~/.config/ai-host/daemon.yaml
```

---

## ⚙️ Advanced Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `HF_HOME` | HuggingFace cache root | `/mnt/data/hf-cache` |
| `HUGGINGFACE_HUB_CACHE` | Alternate HF path | `/mnt/data/cache/hf` |
| `TRANSFORMERS_CACHE` | Transformers cache | `/mnt/data/cache/transformers` |
| `VLLM_CACHE_ROOT` | vLLM cache directory | `/mnt/data/vllm-cache` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Enable >32K context | `1` |
| `PYTHONUNBUFFERED` | Real-time logging | `1` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |

### Systemd Service Customization

**Hot-deploy a new model without service restart:**

```bash
# Create new service file
sudo tee /etc/systemd/system/vllm-custom.service > /dev/null << EOF
[Unit]
Description=vLLM Custom Model Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
Environment=HF_HOME=/mnt/data/hf-cache
Environment=PYTHONUNBUFFERED=1

ExecStart=/mnt/data/venvs/vllm/bin/vllm serve MY-MODEL-ID \
  --host 0.0.0.0 --port 8004 \
  --dtype bfloat16 \
  --quantization fp8 \
  --enforce-eager

[Install]
WantedBy=multi-user.target
EOF

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable vllm-custom
sudo systemctl start vllm-custom
```

### Network Configuration

**Enable CORS for web clients:**

```bash
# Add to vLLM flags:
--allowed-origins "*" \
--allowed-methods "GET,POST,OPTIONS" \
--allowed-headers "Content-Type,Authorization"
```

**Enable SSL termination (behind reverse proxy):**

```bash
# Use nginx/traefik with Let's Encrypt cert
# Point proxy to http://localhost:8002
```

---

## 🐛 Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `No space left on device` | HF cache on root partition | `ln -s /mnt/data/.cache ~/.cache` |
| `EngineCore crash on startup` | GDN kernel JIT failure | Add `--gdn-prefill-backend triton` |
| `Tool parser error` | Wrong parser for model | Use `--tool-call-parser qwen3_coder` for Qwen models |
| `OOM during loading` | GPU memory insufficient | Reduce `--gpu-memory-utilization` or use smaller model |
| `vLLM version incompatible` | CUDA version mismatch | Use vLLM 0.18.0 for CUDA 12.4-12.8 |
| `Tunnel connection refused` | SSH tunnel not running | `systemctl --user restart ai-host-tunnel` |

### Logs and Diagnostics

```bash
# View vLLM logs
ssh ubuntu@myserver "tail -f /var/log/vllm-qwen.log"

# Check systemd service status
ssh ubuntu@myserver "systemctl status vllm-qwen --no-pager -l"

# Check GPU status
ssh ubuntu@myserver "nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv"

# Debug curl requests
curl -v http://localhost:8002/v1/models
```

### Debug Mode

```bash
# Run vLLM manually (not via systemd) for debug output
ssh ubuntu@myserver << 'EOF'
source /mnt/volume_3mx7vkr/python_envs/vllm-env/bin/activate
vllm serve Qwen/Qwen3-Coder-Next \
  --host 0.0.0.0 --port 8002 \
  --dtype bfloat16 --quantization fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gdn-prefill-backend triton \
  --log-level debug
EOF
```

---

## 📈 Performance Tuning

### For Low Latency

```bash
# Add to vLLM flags:
--disable-log-requests      # Reduce I/O overhead
--disable-frontend-multiprocessing  # Single-process mode
--uvicorn-log-level warning # Quieter logs
```

### For High Throughput

```bash
# For batch workloads:
--max-num-seqs 256          # Increase batch size
--enable-chunked-prefill    # Handle large prompts efficiently
--max-seq-len-to-prefill 4096  # Optimize prefill
```

### Memory Optimization

```bash
# Reduce VRAM usage:
--gpu-memory-utilization 0.75   # Use 75% instead of 85%
--enforce-eager                 # Disable CUDA graphs
--disable-custom-all-reduce     # Save memory
```

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests and make changes
4. Run linting: `golangci-lint run ./...`
5. Commit: `git commit -m 'feat: add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include: Go version, GPU model, vLLM version, error logs
- For security issues, email security@codinglone.com

### Community

- Discord: [Join our community](https://discord.gg/ai-host)
- Twitter: [@codinglone](https://twitter.com/codinglone)

---

## ✨ Acknowledgments

Built with:

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - GPU-optimized kernels
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model ecosystem
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU computing

Special thanks to:

- The vLLM team for PCM support and continuous improvements
- NVIDIA for CUDA libraries and documentation
- The open-source AI community for benchmarks and datasets

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 📞 Support

| Channel | Description |
|---------|-------------|
| [GitHub Issues](https://github.com/codinglone/ai-host/issues) | Bug reports, feature requests |
| [Discord](https://discord.gg/ai-host) | Community support, Q&A |
| Email | support@codinglone.com (enterprise) |
| [Contributing Guide](.github/CONTRIBUTING.md) | Development setup |

---

## 🏆 Roadmap

### v1.1 (Next)
- [ ] Multi-model scheduling (parallel serving)
- [ ] TUI dashboard with real-time metrics
- [ ] Automatic quantization switching based on load
- [ ] GPU memory usage alerts

### v1.2 (Planned)
- [ ] GGUF support for CPU/offload serving
- [ ] Kubernetes operator for cluster deployment
- [ ] Prometheus metrics export
- [ ] Load balancing between multiple GPUs

---

<div align="center">

**Made with 🤖 and Go for the open-source AI community**

[![Star on GitHub](https://img.shields.io/github/stars/codinglone/ai-host?logo=github&style=flat-square)](https://github.com/codinglone/ai-host/stargazers)

</div>
