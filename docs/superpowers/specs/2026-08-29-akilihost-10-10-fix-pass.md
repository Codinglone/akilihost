# Spec: akilihost 10/10 fix pass

Date: 2026-08-29
Status: Approved

## Context

`akilihost` is a Go CLI (cobra) for self-hosting coding LLMs on single-GPU
NVIDIA servers. Post-PR#1 review rated it 7/10. This spec covers all fixes to
reach 10/10. User decisions: implement `recommend` (delete `tui`/`daemon`
stubs), implement the three README flags as real vLLM passthroughs.

## Changes

### 1. Sizing — make it real (`host/sizers.go`, `host/models.go`)

- `parseBParams`: real regex parsing of the `...-<N>B` / `...-<N>B-A<M>B`
  model-name suffix → total params in millions. Remove the `activeParams=2`
  MoE hack (active params are a compute metric, not a memory metric).
- New curated fields on `Model`: `Layers`, `KVHeads`, `HeadDim`.
- New curated field on `Quantization`: `BitsPerWeight` (e.g. Q4_K_M ≈ 4.83,
  Q8_0 ≈ 8.5, F16 = 16).
- `SizeModel` formulas (all MiB):
  - `WeightsMB = totalParams × 1e6 × BitsPerWeight / 8 / 1024²`
  - `KVCacheMB = 2 × Layers × KVHeads × HeadDim × 2 (f16 elems) × ctx / 1024²`
  - `TotalMB = WeightsMB + KVCacheMB + 512 (activation reserve)`
- `Quantization.MinVRAMMB` (curated) remains the **authoritative fit decision**
  input for `FindFit`/`SelectBackend`. The `SizingResult` breakdown is a
  display-only estimate, labeled "≈".

### 2. Health-check port bug (`cli/cmd_serve.go`)

- `waitAndVerify` was called with hardcoded `8080` while the service listens
  on `port` (8002+). Pass the actual `port`.
- Timeout: named constant, 60s → 300s (large model loads are slow).

### 3. Serve flags (`cli/cmd_serve.go`)

- `--port` (default 8002). Explicitly set + busy → error. Unset/default + busy
  → auto-increment.
- `--gpu-memory-utilization` (default 0.90) → vLLM `--gpu-memory-utilization`.
- `--max-model-len` (default 32768) → vLLM `--max-model-len`; also the ctx
  used for the sizing display.
- Delete `determinePort` substring hack.

### 4. Stubs

- `recommend`: real command. Same model/auto resolution as `serve`; runs
  `host.FindFit`; prints a table (quant, backend, fit?, ≈size, headroom) +
  recommendation line.
- Delete `cli/cmd_tui.go`, `cli/cmd_daemon.go`.

### 5. Dead code & races

- `cmd_serve.go`: dead if/else around service name → always
  `host.ServiceName(model)` (verified identical result).
- `cmd_stop.go`: unreachable `stopByPID` branch → remove.
- `cmd_ps.go`: fire-and-forget `checkHealth` goroutines → WaitGroup, collect
  lines, print after join.

### 6. Housekeeping + tests

- `root.go`: `Use: "ai-host"` → `"akilihost"`.
- Delete `bin/akilihost` (committed binary); add `bin/` to `.gitignore`.
- Delete `models.yaml` (dead); keep `LoadModelDB()` with an honest comment.
- New unit tests:
  - `host/`: real param parsing, sizing math (known-model fixture values).
  - `cli/`: port selection (explicit-busy → error, auto-increment),
    `checkHealth` against `httptest` servers, recommend table rendering.
- Out of scope: `--enforce-eager` and `--gdn-prefill-backend triton` stay as-is.

## Verification

`go build ./...`, `go vet ./...`, `go test ./...` all green.
