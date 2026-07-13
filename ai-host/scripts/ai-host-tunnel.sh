#!/usr/bin/env bash
# ai-host-tunnel: Resilient SSH tunnel for self-hosted models
# Usage: ./ai-host-tunnel.sh [start|stop|status|restart]
# Source this in .bashrc/.zshrc for auto-setup on shell start

set -euo pipefail

SERVER="${AI_HOST_SERVER:-myserver}"
PORTS="${AI_HOST_PORTS:-8002 8003}"
LOG="${AI_HOST_TUNNEL_LOG:-/tmp/ai-host-tunnel.log}"
PIDFILE="${AI_HOST_TUNNEL_PID:-/tmp/ai-host-tunnel.pid}"

tunnel_start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Tunnel already running (PID $(cat "$PIDFILE"))"
    return 0
  fi

  echo "Starting tunnel to $SERVER..."
  nohup bash -c "
    while true; do
      ssh -fN -o ServerAliveInterval=30 \
             -o ServerAliveCountMax=3 \
             -o ExitOnForwardFailure=yes \
             -L 8002:localhost:8002 \
             -L 8003:localhost:8003 \
             $SERVER 2>&1
      echo \"[\$(date)] Tunnel dropped, reconnecting in 5s...\"
      sleep 5
    done
  " > "$LOG" 2>&1 &

  echo "$!" > "$PIDFILE"
  echo "Tunnel started (PID $(cat "$PIDFILE"))"
}

tunnel_stop() {
  if [ ! -f "$PIDFILE" ]; then
    echo "No tunnel running"
    return 0
  fi
  pid=$(cat "$PIDFILE")
  # Kill the watcher loop
  kill "$pid" 2>/dev/null || true
  # Kill any lingering SSH tunnels
  pkill -f "ssh.*-L 800[23].*$SERVER" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Tunnel stopped"
}

tunnel_status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Tunnel is running (PID $(cat "$PIDFILE"))"
    curl -s --max-time 3 http://localhost:8002/v1/models 2>/dev/null | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print('  Model:', d['data'][0]['id'])" 2>/dev/null || \
      echo "  Waiting for model to load..."
    return 0
  else
    echo "Tunnel is not running"
    return 1
  fi
}

case "${1:-start}" in
  start)   tunnel_start ;;
  stop)    tunnel_stop  ;;
  restart) tunnel_stop; sleep 1; tunnel_start ;;
  status)  tunnel_status ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
