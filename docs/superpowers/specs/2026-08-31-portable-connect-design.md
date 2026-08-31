# Portable Connect: Few-Minute Setup for Any Ubuntu VM

**Date:** 2026-08-31
**Status:** Approved
**Author:** opencode/muse-spark-1.2
**Target:** Any Ubuntu VM with NVIDIA GPU + SSH (Azure, GCP, bare metal)

## Problem

Current setup requires manual steps that caused the `bind [127.0.0.1]:8002: Address already in use` outage debugged on 2026-08-29/31:
- User edits `~/.ssh/config` by hand (Host `myserver` vs `api-product-dev` vs `model-tunnel` drift)
- Two systemd services fight for port 8002: system `/etc/systemd/system/ai-host-tunnel.service` (to dead `myserver`) vs user `~/.config/systemd/user/ai-host-tunnel.service` (to `api-product-dev`)
- `~/.config/opencode/opencode.json` `baseURL http://localhost:8002/v1` is hand-edited
- Model selection is manual (`akilihost recommend` then `serve`) with no guidance on which fits VRAM
- No health verification until `opencode run` fails in TUI
- `scripts/ai-host-tunnel.sh` defaults to `myserver` and uses `pkill -f` without `ExitOnForwardFailure`
- `scripts/deploy.sh` only handles vLLM `Qwen3-Coder-Next` on `myserver` with no llama.cpp path

A new VM needs 30+ minutes of copy-paste and debugging. Goal is 2 minutes for any Ubuntu VM.

## Goal

`akilihost connect` — single laptop command that, given a fresh Ubuntu VM IP/user/key, achieves **C) Both + VM health**:
1. `ssh <alias> "akilihost ps"` shows `Health: ok` for chosen model on remote `localhost:8002`
2. `curl http://localhost:8002/health` on laptop returns `{"status":"ok"}` via tunnel
3. `curl http://localhost:8002/v1/models` and `opencode models selfhosted` list the model
4. `opencode run "hello" --model selfhosted/unsloth/Qwen3.8-27B-GGUF` streams response

Constraints from brainstorming:
- **Generic Ubuntu + SSH** (not cloud-specific)
- **Auto-create Host** in `~/.ssh/config` from `--host/--user/--key/--alias`
- **Always prompt** model choice from `akilihost recommend` table
- **Auto-rewrite + backup** `opencode.json`

## Design

### Architecture

```
laptop (akilihost connect)
  ├─ (laptop) ensure ~/.ssh/config Host <alias> with LocalForward 8002/8003
  ├─ ssh <alias> "akilihost init"          # idempotent, installs llama-server, hf_hub, /opt/akilihost
  ├─ ssh <alias> "akilihost recommend"     # parse table for prompt
  ├─ prompt: select model [1..N] (always)
  ├─ ssh <alias> "akilihost serve <model> --port 8002" + poll health 300s
  ├─ (laptop) backup+rewrite ~/.config/opencode/opencode.json
  └─ (laptop) systemctl --user ai-host-tunnel -> <alias> + verify curl + opencode
```

All orchestration via `ssh` exec (Go `os/exec` `ssh` binary), not Go SSH library, to reuse `~/.ssh/config`, keys, `StrictHostKeyChecking`, and existing `autossh` reliability. VM-side truth is `akilihost recommend/serve/ps` over SSH; laptop never does VRAM math locally.

Isolation: 6 functions, each testable without live VM:

```
ensureSSHConfig(alias, host, user, key) -> (created, diff)
remoteInit(alias) -> error
remoteRecommend(alias) -> []ModelOption
remoteServe(alias, model, port) -> error (with polling)
patchOpencode(tunnelPort) -> backupPath
startTunnel(alias, port) -> error
```

### CLI UX

```
akilihost connect [alias] [flags]
  --host string        IP or HostName (required if alias not in ~/.ssh/config)
  --user string        SSH user (default ubuntu)
  --key  string        IdentityFile path (default ~/.ssh/id_ed25519)
  --port int           remote API port (default 8002)
  --tunnel-port int    local port (default same as --port)
  --alias string       Host alias (positional or --alias, default mygpu)
  --yes                skip overwrite confirmations
  --dry-run            print planned actions without executing
```

Behavior:
- If `alias` exists and `--host` omitted → reuse Host, validate `ssh <alias> true` 5s.
- If `alias` missing and `--host` given → create Host, prompt overwrite if file would be clobbered.
- If neither alias nor host given → error with example: `akilihost connect mygpu --host 1.2.3.4 --user ubuntu --key ~/.ssh/id_ed25519`.
- Shows `recommend` table, prompts `Select [1..N]:`.
- Idempotent: re-running `connect mygpu` checks `curl localhost:<port>/health` and `ssh <alias> "akilihost ps"` health; if same model healthy, skips `serve`.
- On success prints:
  ```
  ✓ VM mygpu (1.2.3.4) healthy: Qwen3.8-27B UD-Q4_K_XL on 8002
  ✓ Tunnel localhost:8002 -> mygpu:8002
  ✓ opencode ready: opencode run "hello" --model selfhosted/unsloth/Qwen3.8-27B-GGUF
  ```

### SSH Config Handling

- Path: `~/.ssh/config`, preserve `gcloud` auto-block (between `Google Compute Engine Section` markers), comments, other Hosts. Use Go parser that reads line-by-line, not regexp overwrite.
- Create Host block:
  ```
  Host <alias>
      HostName <host>
      User <user>
      IdentityFile <key>
      IdentitiesOnly yes
      StrictHostKeyChecking accept-new
      ServerAliveInterval 30
      ServerAliveCountMax 3
      LocalForward 8002 127.0.0.1:8002
      LocalForward 8003 127.0.0.1:8003
  ```
- Update: if Host exists but fields differ, show unified diff and prompt `Overwrite? [y/N]` unless `--yes`.
- Validation: `ssh -o ConnectTimeout=5 -o BatchMode=yes <alias> true` before continuing; on fail print `ssh -v <alias>` hint and `~/.ssh/config` snippet.
- Tunnel service: overwrite `~/.config/systemd/user/ai-host-tunnel.service` `ExecStart=-L <port>:localhost:<port> <alias>` (replace old `myserver`), `systemctl --user daemon-reload; systemctl --user enable --now ai-host-tunnel.service`. If system `/etc/systemd/system/ai-host-tunnel.service` is active, warn `sudo systemctl disable --now ai-host-tunnel.service` and fallback `pkill -f "autossh.*:8002"` as we did for pid `1154/1163` vs `7459`.

### VM Side Flow

Executed as `ssh <alias> "bash -lc 'set -euo pipefail; ...'"`:

1. **Binary ensure:** `command -v akilihost >/dev/null || (scp ./akilihost && ssh chmod +x)`. Laptop builds `go build -o /tmp/akilihost` if needed and `scp /tmp/akilihost <alias>:~/akilihost` then `ssh <alias> "mkdir -p ~/bin && mv ~/akilihost ~/bin/ && echo ~/bin >> ~/.bashrc"` (ensures PATH). Uses `scp` via `os/exec scp`.
2. **init:** `~/bin/akilihost init` — idempotent: checks `which llama-server`, `pip show huggingface_hub`, `test -d /opt/akilihost/models`. Logs to `/tmp/akilihost-connect.log` remote. Reuses existing `cli/cmd_init.go`.
3. **recommend:** `~/bin/akilihost recommend` capture stdout; parse table lines with `host/models.go` `Recommend` logic client-side for display; present numbered list.
4. **serve:** `~/bin/akilihost serve <model> --port <port>` — uses chosen model name (e.g., `Qwen3.8-27B`). Poll `~/bin/akilihost ps` and `curl -s http://localhost:<port>/health` every 3s up to 300s (same as `cli/cmd_serve.go:waitAndVerify`). On timeout, `ssh <alias> "journalctl --user -u akilihost* --no-pager -n 50; ss -tlnp | grep <port>"`.
5. **Health:** final `ssh <alias> "~/bin/akilihost ps"` must contain `Health: ok` and remote `curl -s http://localhost:<port>/v1/models | jq -e '.data[0].id'` valid.

### Laptop Side Flow

1. **Backup:** `cp ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json.bak.<unix>` if exists (always, per B choice).
2. **Rewrite:** Go struct for `opencode.json` (fields `$schema`, `plugin`, `provider`, `model`). Merge:
   ```json
   "provider": {
     "selfhosted": {
       "npm": "@ai-sdk/openai-compatible",
       "name": "Self-Hosted (llama.cpp)",
       "options": {
         "baseURL": "http://localhost:<tunnel-port>/v1",
         "timeout": 600000,
         "chunkTimeout": 120000
       },
       "models": {
         "unsloth/Qwen3.8-27B-GGUF": {"name": "Qwen3.8-27B UD-Q4_K_XL (T4 split)", "maxTokens": 16384},
         ...existing models from host/models.go
       }
     }
   }
   ```
   Preserves other `provider`s (e.g., `anthropic`), only touches `selfhosted`. Creates file if missing.
3. **Tunnel:** `systemctl --user daemon-reload; systemctl --user enable --now ai-host-tunnel.service` (user service we fixed in `~/.config/systemd/user/ai-host-tunnel.service:11` to use `<alias>`). Fallback if `systemctl --user` not available: `ssh -fN -o ExitOnForwardFailure=yes -L <tunnel-port>:localhost:<port> <alias>`.
4. **Verify:** `curl --max-time 5 http://localhost:<tunnel-port>/health` → `{"status":"ok"}`, `curl /v1/models`, `opencode models selfhosted` must list `selfhosted/unsloth/Qwen3.8-27B-GGUF`.

### Error Handling & Idempotency

- **Idempotent:** each remote step checks before acting (`test -x ~/bin/akilihost`, `akilihost ps --json` health, `grep -q "^Host <alias>" ~/.ssh/config`). Re-running is safe.
- **SSH fail:** run `ssh -T <alias> echo ok` diagnostics, print `~/.ssh/config` block and `ssh -v` hint.
- **Serve timeout:** print remote `journalctl --user` tail and `nvidia-smi` VRAM, suggest `akilihost serve <model> --max-model-len 8192` for smaller KV cache.
- **Tunnel bind error:** on `bind: Address already in use`, run `lsof -i :<port>` + `ss -tlnp | grep <port>` + `systemctl --user status ai-host-tunnel.service` and `systemctl status ai-host-tunnel.service` (system), then `pkill -f "ssh.*-L <port>"` fallback, retry once.
- **Opencode missing:** if `which opencode` fails, warn but succeed (tunnel + VM health are primary).
- **Portable guards:** checks `nvidia-smi` on VM, warns if no GPU; checks `systemd --user` exists; requires `go` only on laptop for build.

### Testing

- **Unit:** `cli/cmd_connect_test.go` (no live SSH) — 10 tests:
  1. SSH config create new Host preserves gcloud block
  2. SSH config update existing Host shows diff
  3. SSH config keep LocalForward idempotency
  4. Opencode JSON merge adds selfhosted without deleting anthropic
  5. Opencode backup created
  6. Recommend parse from `akilihost recommend` fixture
  7. Tunnel service write with `<alias>` substitution
  8. Idempotent re-run skips serve when health ok
  9. Dry-run prints without side effects
  10. Port collision handling calls pkill fallback
- **Manual integration (no CI VM):** on fresh Ubuntu VM, `go build -o akilihost && ./akilihost connect testgpu --host <IP> --user ubuntu --key ~/.ssh/id --alias testgpu --port 8002` → verify 3 health checks pass.
- `go vet ./... && go test ./... -count=1` must pass.

### Verification

1. Fresh VM (T4 16GB, Ubuntu 22.04) → laptop `akilihost connect newgpu --host <IP> --user ubuntu --key <pem>` → prompts model → `akilihost serve auto` → `akilihost ps` shows `Health: ok`.
2. Laptop `curl http://localhost:8002/health` → `ok`, `curl /v1/models` → model ID.
3. `opencode run "hello" --model selfhosted/unsloth/Qwen3.8-27B-GGUF` streams 3-word reply (like `Hello there friend` verified 2026-08-31).
4. Re-run same `connect` → skips serve, reuses tunnel, still health ok (idempotent).

### Tradeoffs

- **Go vs bash:** Go is heavier to build but gives JSON-safe opencode rewrite and testable SSH config parsing; bash `jq` would be fragile (seen in `ai-host-tunnel.sh` vs `opencode.json`).
- **ssh exec vs Go SSH lib:** exec reuses `~/.ssh/config`, agents, `ControlMaster`, and `autossh`; library would need to reimplement `IdentityFile`, `StrictHostKeyChecking`, `LocalForward`.
- **Always prompt vs auto:** prompting satisfies B choice but adds 1 interaction; justified because `recommend` table headroom helps user choose Q4 vs Q3 for T4 vs A100.
- **System vs user systemd:** we keep user service as primary (no sudo) and warn to disable system service; avoids password prompt during `connect`.

## Out of Scope

- Terraform/cloud VM creation
- Vision/multimodal `llama-mtmd` support
- Disk resize automation (existing `az disk update` docs remain)
