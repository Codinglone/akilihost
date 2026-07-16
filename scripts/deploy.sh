#!/usr/bin/env bash
set -euo pipefail

# ============================================
# ai-host — Deploy to Server (No Local Sudo)
# ============================================
# Run this from your LOCAL machine to:
#   1. Install vLLM venv and dependencies (server)
#   2. Install and start vLLM systemd service (server)
#   3. Copy ai-host binary
# ============================================

SERVER="${1:-myserver}"
MODEL="${2:-Qwen/Qwen3-Coder-Next}"
PORT="${3:-8002}"

echo "=== ai-host: Deploy to $SERVER ==="

# --- Step 1: Install system dependencies on server ---
echo "[1/4] Installing vLLM venv and dependencies..."
ssh "$SERVER" bash -s << 'SRV'
  set -euo pipefail
  VENV_DIR="/mnt/volume_3mx7vkr/python_envs/vllm-env"
  if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating Python venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    pip install vllm==0.18.0
    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6/
    mkdir -p /mnt/volume_3mx7vkr/hf-cache
    mkdir -p /mnt/volume_3mx7vkr/python_envs
    # Symlink cache directories
    ln -sf /mnt/volume_3mx7vkr/hf-cache /home/ubuntu/.cache/huggingface 2>/dev/null || true
    mkdir -p /mnt/volume_3mx7vkr/vllm-cache
    echo "  Venv ready!"
  else
    echo "  Venv already exists at $VENV_DIR"
  fi
SRV

# --- Step 2: Copy ai-host binary and systemd files ---
echo "[2/4] Copying ai-host binary..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp "$SCRIPT_DIR/../ai-host" "$SERVER:/home/ubuntu/ai-host/bin/"
scp "$SCRIPT_DIR/../systemd/vllm-qwen.service" "$SERVER:/tmp/vllm-qwen.service"

# --- Step 3: Install systemd service on server ---
echo "[3/4] Installing vLLM systemd service..."
ssh "$SERVER" bash -s << 'SRV'
  sudo cp /tmp/vllm-qwen.service /etc/systemd/system/vllm-qwen.service
  sudo systemctl daemon-reload
  sudo systemctl enable vllm-qwen
  echo "  vLLM systemd service enabled (starts on boot)"
SRV

# --- Step 4: Start the model now ---
echo "[4/4] Starting model..."
ssh "$SERVER" "sudo systemctl start vllm-qwen"
sleep 5
ssh "$SERVER" "systemctl is-active vllm-qwen"
echo "  Model started!"

echo ""
echo "=== Done ==="
echo "Model serving at http://localhost:$PORT/v1"
echo "Logs: ssh $SERVER 'tail -f /var/log/vllm-qwen.log'"
echo "Run: curl http://localhost:$PORT/v1/models"
