# akilihost 10/10 Fix Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring akilihost from 7/10 to 10/10 by implementing real VRAM sizing, three serve flags, a functional `recommend` command, dead-code/race fixes, and housekeeping — all per the approved spec at `docs/superpowers/specs/2026-08-29-akilihost-10-10-fix-pass.md`.

**Architecture:** The `host` package gains architecture fields (`Layers`, `KVHeads`, `HeadDim`, `BitsPerWeight`) on the in-memory model database, enabling real weight + KV-cache math. The `cli` package gains `--port` / `--gpu-memory-utilization` / `--max-model-len` flags, a port-resolution helper with busy detection, a real `recommend` command, and synchronized health checks. `models.yaml` is deleted (dead — `LoadModelDB` already uses in-memory `prepopulatedModels`).

**Tech Stack:** Go 1.26 + cobra CLI, systemd, net/http for testable health checks

---

## File Structure

**New files:**
- `cli/cmd_serve_test.go` — port resolution tests
- `cli/cmd_ps_test.go` — health-check tests with httptest
- `cli/cmd_recommend_test.go` — recommend table rendering tests

**Modified files:**
- `host/models.go` — add `Layers`, `KVHeads`, `HeadDim` to `Model`; add `BitsPerWeight` to `Quantization`; populate for all 4 models; honest comment on `LoadModelDB`
- `host/sizers.go` — real `parseBParams`, real `calcKVCache`, rewritten `SizeModel` (add `contextTokens` param), remove MoE hack, update `FindFit` signature
- `host/llamacpp.go` — add `ctxSize int` param to `BuildLlamaServerCommand`
- `host/llamacpp_test.go` — update test for new `ctxSize` param
- `host/sizers_test.go` — update for new `FindFit`/`SizeModel` signatures; add real-param sizing tests
- `host/models_test.go` — add architecture-field assertions
- `cli/cmd_serve.go` — add 3 flags, `resolvePort`, delete `determinePort`, fix `waitAndVerify` timeout, remove dead if/else, pass flags to vllm/llama-cpp
- `cli/cmd_recommend.go` — real implementation with `FindFit` + table output
- `cli/cmd_ps.go` — remove `go` from `checkHealth`, refactor to `net/http`, collect results
- `cli/cmd_stop.go` — remove unreachable `stopByPID` branch
- `cli/root.go` — `Use: "akilihost"`, remove `tuiCmd`/`daemonCmd` registrations
- `README.md` — update flag defaults (0.90, 32768), remove `recommend` stub note
- `.gitignore` — new, with `bin/`

**Deleted files:**
- `cli/cmd_tui.go`
- `cli/cmd_daemon.go`
- `bin/akilihost`
- `models.yaml`

---

### Task 1: Add Architecture Fields to Model/Quantization Structs

**Files:**
- Modify: `host/models.go`
- Modify: `host/models_test.go`

- [ ] **Step 1: Write the failing test**

Add to `host/models_test.go` (append to existing file):

```go
func TestModelArchitectureFields(t *testing.T) {
	models, err := LoadModelDB()
	if err != nil {
		t.Fatalf("LoadModelDB failed: %v", err)
	}

	want := map[string]struct {
		Layers   int
		KVHeads  int
		HeadDim  int
	}{
		"Qwen3-Coder-Next":    {12, 2, 256},  // 80B MoE, hybrid: 12 full-attn layers of 48
		"Qwen2.5-Coder-32B":   {64, 8, 128},  // dense, all 64 layers
		"Devstral 2 123B":     {88, 8, 128},  // dense, all 88 layers
		"Qwen3.8-27B":         {16, 4, 256},  // hybrid: 16 Gated Attention layers of 64
	}

	for _, m := range models {
		exp, ok := want[m.Name]
		if !ok {
			continue
		}
		if m.Layers != exp.Layers {
			t.Errorf("%s: Layers = %d, want %d", m.Name, m.Layers, exp.Layers)
		}
		if m.KVHeads != exp.KVHeads {
			t.Errorf("%s: KVHeads = %d, want %d", m.Name, m.KVHeads, exp.KVHeads)
		}
		if m.HeadDim != exp.HeadDim {
			t.Errorf("%s: HeadDim = %d, want %d", m.Name, m.HeadDim, exp.HeadDim)
		}
	}
}

func TestQuantizationBitsPerWeight(t *testing.T) {
	models, err := LoadModelDB()
	if err != nil {
		t.Fatalf("LoadModelDB failed: %v", err)
	}

	for _, m := range models {
		for _, q := range m.Quantizations {
			if q.BitsPerWeight <= 0 {
				t.Errorf("%s/%s: BitsPerWeight = %f, want > 0", m.Name, q.Name, q.BitsPerWeight)
			}
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./host/ -run TestModelArchitecture -v`
Expected: FAIL — `m.Layers` undefined (field doesn't exist yet)

- [ ] **Step 3: Add fields to structs**

In `host/models.go`, add `Layers`, `KVHeads`, `HeadDim` to `Model` (after `Backend`):

```go
type Model struct {
	RepoID         string             `yaml:"repo_id"`
	Name           string             `yaml:"name"`
	Description    string             `yaml:"description"`
	Architecture   string             `yaml:"architecture"`
	TotalParams    string             `yaml:"total_params"`
	ActiveParams   string             `yaml:"active_params"`
	ContextTok     int                `yaml:"context_tokens"`
	License        string             `yaml:"license"`
	Backend        string             `yaml:"backend"`
	Layers         int                `yaml:"layers"`
	KVHeads        int                `yaml:"kv_heads"`
	HeadDim        int                `yaml:"head_dim"`
	Benchmarks     map[string]float32 `yaml:"benchmarks,omitempty"`
	Quantizations  []Quantization     `yaml:"quantizations"`
}
```

Add `BitsPerWeight` to `Quantization` (after `FilePattern`):

```go
type Quantization struct {
	Name          string   `yaml:"name"`
	Description   string   `yaml:"description"`
	DType         string   `yaml:"dtype"`
	QuantMode     string   `yaml:"quant_mode"`
	MinVRAMMB     int      `yaml:"min_vram_mb"`
	Flags         []string `yaml:"flags,omitempty"`
	FilePattern   string   `yaml:"file_pattern,omitempty"`
	BitsPerWeight float32  `yaml:"bits_per_weight"`
}
```

- [ ] **Step 4: Populate architecture fields for all 4 models**

Update each model entry in `prepopulatedModels`:

**Qwen3-Coder-Next** (add after `Backend: "vllm"`):
```go
		Layers:   12,   // 48 total layers, full_attention_interval=4 → 12 full-attn layers
		KVHeads:  2,
		HeadDim:  256,
```
And its FP8 quantization:
```go
				BitsPerWeight: 8.0,
```

**Qwen2.5-Coder-32B** (add after `Backend: "vllm"` — default):
```go
		Layers:   64,
		KVHeads:  8,
		HeadDim:  128,
```
And its BF16 quantization:
```go
				BitsPerWeight: 16.0,
```

**Devstral 2 123B** (add after `Backend: "vllm"` — default):
```go
		Layers:   88,
		KVHeads:  8,
		HeadDim:  128,
```
And its FP8 quantization:
```go
				BitsPerWeight: 8.0,
```

**Qwen3.8-27B** (add after `Backend: "llama-cpp"`):
```go
		Layers:   16,   // 64 total layers, hybrid: 16 Gated Attention layers (every 4th)
		KVHeads:  4,
		HeadDim:  256,
```
And its quantizations:
```go
				BitsPerWeight: 4.5,  // UD-Q4_K_XL: ~4.5 bits/weight
```
```go
				BitsPerWeight: 3.5,  // UD-Q3_K_XL: ~3.5 bits/weight
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./host/ -run TestModelArchitecture -v && go test ./host/ -run TestQuantizationBits -v`
Expected: PASS

- [ ] **Step 6: Run full host test suite**

Run: `go test ./host/ -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 7: Commit**

```bash
git add host/models.go host/models_test.go
git commit -m "feat: add Layers/KVHeads/HeadDim/BitsPerWeight fields to model structs"
```

---

### Task 2: Real parseBParams + Sizing Math

**Files:**
- Modify: `host/sizers.go`
- Modify: `host/sizers_test.go`
- Modify: `host/llamacpp.go`
- Modify: `host/llamacpp_test.go`

- [ ] **Step 1: Write the failing tests**

Add to `host/sizers_test.go` (append to existing file):

```go
func TestParseBParams(t *testing.T) {
	tests := []struct {
		input    string
		expected int // in millions
	}{
		{"80B", 80000},
		{"32B", 32000},
		{"123B", 123000},
		{"27B", 27000},
		{"7B", 7000},
		{"0.5B", 500},
		{"80B-40B", 80000},  // MoE: take total
		{"", 0},
		{"invalid", 0},
	}
	for _, tt := range tests {
		got := parseBParams(tt.input)
		if got != tt.expected {
			t.Errorf("parseBParams(%q) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

func TestSizeModelRealParams(t *testing.T) {
	gpu := &GPUInfo{Name: "Test", TotalVRAMMB: 16384, SystemRAMMB: 28456}
	sizer := NewModelSizer(gpu)

	// Qwen2.5-Coder-32B, BF16 (16 bits/weight), 32K ctx
	model := &Model{
		Name:        "Qwen2.5-Coder-32B",
		TotalParams: "32B",
		Layers:      64,
		KVHeads:     8,
		HeadDim:     128,
	}
	quant := &Quantization{Name: "BF16", BitsPerWeight: 16.0}

	r := sizer.SizeModel(model, quant, 32768)

	// WeightsMB = 32000 * 1e6 * 16 / 8 / 1048576 = 61035
	if r.WeightsMB != 61035 {
		t.Errorf("WeightsMB = %d, want 61035", r.WeightsMB)
	}
	// KVCacheMB = 2 * 64 * 8 * 128 * 2 * 32768 / 1048576 = 8192
	if r.KVCacheMB != 8192 {
		t.Errorf("KVCacheMB = %d, want 8192", r.KVCacheMB)
	}
	// TotalMB = 61035 + 8192 + 512 = 69739
	if r.TotalMB != 69739 {
		t.Errorf("TotalMB = %d, want 69739", r.TotalMB)
	}
	if r.MaxContext != 32768 {
		t.Errorf("MaxContext = %d, want 32768", r.MaxContext)
	}
}

func TestSizeModelQwen38Q4(t *testing.T) {
	gpu := &GPUInfo{Name: "Test", TotalVRAMMB: 16384, SystemRAMMB: 28456}
	sizer := NewModelSizer(gpu)

	model := &Model{
		Name:        "Qwen3.8-27B",
		TotalParams: "27B",
		Layers:      16,
		KVHeads:     4,
		HeadDim:     256,
	}
	quant := &Quantization{Name: "UD-Q4_K_XL", BitsPerWeight: 4.5}

	r := sizer.SizeModel(model, quant, 32768)

	// WeightsMB = 27000 * 1e6 * 4.5 / 8 / 1048576 = 14488
	if r.WeightsMB != 14488 {
		t.Errorf("WeightsMB = %d, want 14488", r.WeightsMB)
	}
	// KVCacheMB = 2 * 16 * 4 * 256 * 2 * 32768 / 1048576 = 2048
	if r.KVCacheMB != 2048 {
		t.Errorf("KVCacheMB = %d, want 2048", r.KVCacheMB)
	}
	// TotalMB = 14488 + 2048 + 512 = 17048
	if r.TotalMB != 17048 {
		t.Errorf("TotalMB = %d, want 17048", r.TotalMB)
	}
}
```

Also update the existing `TestFindFitLlamaCppUsesTotalMemory` and `TestFindFitVllmRequiresVRAMOnly` to pass the new `contextTokens` parameter:

```go
results := sizer.FindFit([]Model{qwen38}, gpu.TotalVRAMMB*85/100, 32768)
```
```go
results := sizer.FindFit([]Model{qwen3}, gpu.TotalVRAMMB*85/100, 32768)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./host/ -run TestParseBParams -v`
Expected: FAIL — `parseBParams` still returns 80000 for everything

Run: `go test ./host/ -run TestSizeModelReal -v`
Expected: FAIL — `SizeModel` signature doesn't accept `contextTokens`

- [ ] **Step 3: Implement real parseBParams**

In `host/sizers.go`, replace the `parseBParams` function:

```go
import (
	"regexp"
	"strconv"
)

var paramRe = regexp.MustCompile(`(\d+(?:\.\d+)?)B`)

func parseBParams(s string) int {
	matches := paramRe.FindStringSubmatch(s)
	if len(matches) < 2 {
		return 0
	}
	f, err := strconv.ParseFloat(matches[1])
	if err != nil {
		return 0
	}
	return int(f * 1000)
}
```

- [ ] **Step 4: Rewrite calcKVCache with real formula**

In `host/sizers.go`, replace `calcKVCache`:

```go
func (s *ModelSizer) calcKVCache(model *Model, tokens int) int {
	// 2 (K+V) × Layers × KVHeads × HeadDim × 2 (f16 bytes/elem) × tokens / 1024²
	return 2 * model.Layers * model.KVHeads * model.HeadDim * 2 * tokens / (1024 * 1024)
}
```

- [ ] **Step 5: Rewrite SizeModel with real formulas + contextTokens param**

In `host/sizers.go`, replace `SizeModel`:

```go
func (s *ModelSizer) SizeModel(model *Model, q *Quantization, contextTokens int) *SizingResult {
	totalParams := parseBParams(model.TotalParams) // in millions

	// Weights: params × bits/weight / 8 bits/byte / 1024² bytes/MiB
	weightsMB := int(float64(totalParams) * 1e6 * float64(q.BitsPerWeight) / 8 / (1024 * 1024))

	// KV cache: real architecture-based formula
	kvMB := s.calcKVCache(model, contextTokens)

	// 512 MiB activation overhead
	totalMB := weightsMB + kvMB + 512

	availableMB := s.GPU.TotalVRAMMB * 85 / 100
	headroomMB := availableMB - totalMB

	return &SizingResult{
		Model:        model,
		Quantization: q,
		WeightsMB:    weightsMB,
		KVCacheMB:    kvMB,
		TotalMB:      totalMB,
		AvailableMB:  availableMB,
		HeadroomMB:   headroomMB,
		MaxContext:   contextTokens,
	}
}
```

- [ ] **Step 6: Update FindFit signature**

In `host/sizers.go`, update `FindFit`:

```go
func (s *ModelSizer) FindFit(models []Model, availableMB int, contextTokens int) []*SizingResult {
	var results []*SizingResult
	totalMemoryMB := s.GPU.TotalVRAMMB + s.GPU.SystemRAMMB
	for _, model := range models {
		for _, q := range model.Quantizations {
			effectiveLimit := availableMB
			if model.Backend == "llama-cpp" {
				effectiveLimit = totalMemoryMB * 85 / 100
			}

			if q.MinVRAMMB <= effectiveLimit {
				r := s.SizeModel(&model, &q, contextTokens)
				r.AvailableMB = effectiveLimit
				r.HeadroomMB = effectiveLimit - q.MinVRAMMB
				if r.HeadroomMB >= 0 {
					results = append(results, r)
				}
			}
		}
	}
	return results
}
```

- [ ] **Step 7: Add ctxSize param to BuildLlamaServerCommand**

In `host/llamacpp.go`, update the function signature and `--ctx-size`:

```go
func BuildLlamaServerCommand(model *Model, quant *Quantization, modelPath string, port int, ctxSize int) []string {
	args := []string{
		"llama-server",
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"--ctx-size", strconv.Itoa(ctxSize),
	}
	args = append(args, quant.Flags...)
	return args
}
```

In `host/llamacpp_test.go`, update the test call (line 22):

```go
	args := BuildLlamaServerCommand(model, quant, modelPath, port, 32768)
```

- [ ] **Step 8: Run all host tests to verify they pass**

Run: `go test ./host/ -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 9: Commit**

```bash
git add host/sizers.go host/sizers_test.go host/llamacpp.go host/llamacpp_test.go
git commit -m "feat: real VRAM sizing from architecture params + BitsPerWeight"
```

---

### Task 3: Fix waitAndVerify Timeout + Dead Code in cmd_serve.go

**Files:**
- Modify: `cli/cmd_serve.go`

- [ ] **Step 1: Fix waitAndVerify timeout**

In `cli/cmd_serve.go`, replace the `waitAndVerify` function:

```go
const verifyTimeoutSeconds = 300

func waitAndVerify(port int) {
	fmt.Printf("  Waiting for server to start...\n")
	attempts := verifyTimeoutSeconds / 5
	for i := 0; i < attempts; i++ {
		cmd := exec.Command("curl", "-s", "--max-time", "5", fmt.Sprintf("http://localhost:%d/v1/models", port))
		output, err := cmd.Output()
		if err == nil {
			var data struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if json.Unmarshal(output, &data) == nil && len(data.Data) > 0 {
				fmt.Printf("  Server ready: %s\n", data.Data[0].ID)
				fmt.Printf("  Port: %d\n", port)
				return
			}
		}
		fmt.Printf("  Waiting... (%ds elapsed)\n", (i+1)*5)
	}
	fmt.Println("  Timeout - check logs for errors")
}
```

- [ ] **Step 2: Remove dead if/else in serve command**

In `cli/cmd_serve.go`, replace lines 63-73 (the `if len(results) == 1` / `else` block where both branches do the same thing):

```go
			modelToServe = results[0].Model
			quantization = results[0].Quantization
			fmt.Printf("\nAuto-selecting: %s %s\n", modelToServe.Name, quantization.Name)
```

This replaces the entire if/else that had identical branches.

- [ ] **Step 3: Build and verify**

Run: `go build ./... && go vet ./...`
Expected: build and vet succeed

- [ ] **Step 4: Commit**

```bash
git add cli/cmd_serve.go
git commit -m "fix: waitAndVerify 300s timeout + remove dead if/else in serve"
```

---

### Task 4: Serve Flags + Delete determinePort

**Files:**
- Modify: `cli/cmd_serve.go`
- Create: `cli/cmd_serve_test.go`

- [ ] **Step 1: Write the failing test**

Create `cli/cmd_serve_test.go`:

```go
package cli

import (
	"net"
	"testing"
)

func TestResolvePortExplicitFree(t *testing.T) {
	// Get a free port
	listener, _ := net.Listen("tcp", ":0")
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	got, err := resolvePort(port, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != port {
		t.Errorf("resolvePort(%d, true) = %d, want %d", port, got, port)
	}
}

func TestResolvePortExplicitBusy(t *testing.T) {
	listener, _ := net.Listen("tcp", ":0")
	defer listener.Close()
	port := listener.Addr().(*net.TCPAddr).Port

	_, err := resolvePort(port, true)
	if err == nil {
		t.Fatal("expected error for busy explicit port, got nil")
	}
}

func TestResolvePortAutoIncrement(t *testing.T) {
	// Occupy a port, then verify auto-increment skips it
	listener, _ := net.Listen("tcp", ":0")
	defer listener.Close()
	busyPort := listener.Addr().(*net.TCPAddr).Port

	got, err := resolvePort(busyPort, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == busyPort {
		t.Fatalf("expected auto-increment past busy port %d, got %d", busyPort, got)
	}
	if got < busyPort {
		t.Errorf("expected port >= %d, got %d", busyPort, got)
	}
}

func TestResolvePortDefaultFree(t *testing.T) {
	// Use a high port unlikely to be in use
	port := 18002
	listener, err := net.Listen("tcp", ":18002")
	if err != nil {
		t.Skip("port 18002 not available, skipping")
	}
	listener.Close()

	got, err := resolvePort(18002, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 18002 {
		t.Errorf("resolvePort(18002, false) = %d, want 18002", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cli/ -run TestResolvePort -v`
Expected: FAIL — `resolvePort` undefined

- [ ] **Step 3: Add flag variables and init()**

In `cli/cmd_serve.go`, add package-level flag variables after the imports:

```go
var servePort int
var serveGpuMemUtil float64
var serveMaxModelLen int

func init() {
	serveCmd.Flags().IntVar(&servePort, "port", 8002, "API port (auto-increments if busy unless explicitly set)")
	serveCmd.Flags().Float64Var(&serveGpuMemUtil, "gpu-memory-utilization", 0.90, "Max GPU memory fraction (vLLM)")
	serveCmd.Flags().IntVar(&serveMaxModelLen, "max-model-len", 32768, "Max context tokens")
}
```

- [ ] **Step 4: Implement resolvePort + isPortBusy**

Add to `cli/cmd_serve.go`:

```go
import (
	"net"
)

func resolvePort(explicitPort int, explicit bool) (int, error) {
	if explicit {
		if isPortBusy(explicitPort) {
			return 0, fmt.Errorf("port %d is already in use", explicitPort)
		}
		return explicitPort, nil
	}
	port := explicitPort
	for isPortBusy(port) {
		port++
		if port > 8099 {
			return 0, fmt.Errorf("no free port in range %d-8099", explicitPort)
		}
	}
	return port, nil
}

func isPortBusy(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return true
	}
	listener.Close()
	return false
}
```

- [ ] **Step 5: Delete determinePort and wire up flags in Run**

Delete the `determinePort` function entirely.

In the `Run` function, replace the port assignment:

```go
		// Select backend
		backend := host.SelectBackend(modelToServe, quantization, gpu)
		port, err := resolvePort(servePort, cmd.Flags().Changed("port"))
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
```

Update the `FindFit` call to pass `serveMaxModelLen`:

```go
		results := sizer.FindFit(models, gpu.TotalVRAMMB*85/100, serveMaxModelLen)
```

Update the vllm command building in `createVllmService` — after `args = append(args, "--enforce-eager")`, add:

```go
	args = append(args, "--gpu-memory-utilization", strconv.FormatFloat(serveGpuMemUtil, 'f', -1, 64))
	args = append(args, "--max-model-len", strconv.Itoa(serveMaxModelLen))
```

Update the llama-cpp call in `serveLlamaCpp`:

```go
	args := host.BuildLlamaServerCommand(model, quant, ggufPath, port, serveMaxModelLen)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `go test ./cli/ -run TestResolvePort -v`
Expected: PASS

- [ ] **Step 7: Build and verify**

Run: `go build ./... && go vet ./...`
Expected: build and vet succeed

- [ ] **Step 8: Commit**

```bash
git add cli/cmd_serve.go cli/cmd_serve_test.go
git commit -m "feat: --port/--gpu-memory-utilization/--max-model-len flags with port auto-increment"
```

---

### Task 5: Implement recommend Command

**Files:**
- Modify: `cli/cmd_recommend.go`
- Create: `cli/cmd_recommend_test.go`

- [ ] **Step 1: Write the failing test**

Create `cli/cmd_recommend_test.go`:

```go
package cli

import (
	"strings"
	"testing"

	"github.com/Codinglone/akilihost/host"
)

func TestRenderRecommendation(t *testing.T) {
	gpu := &host.GPUInfo{Name: "Tesla T4", TotalVRAMMB: 16384, SystemRAMMB: 28456}

	results := []*host.SizingResult{
		{
			Model: &host.Model{
				Name:    "Qwen3.8-27B",
				Backend: "llama-cpp",
			},
			Quantization: &host.Quantization{Name: "UD-Q4_K_XL"},
			TotalMB:      17048,
			HeadroomMB:   5000,
		},
	}

	out := renderRecommendation(gpu, results)

	if !strings.Contains(out, "Tesla T4") {
		t.Errorf("expected GPU name in output, got: %s", out)
	}
	if !strings.Contains(out, "Qwen3.8-27B") {
		t.Errorf("expected model name in output, got: %s", out)
	}
	if !strings.Contains(out, "UD-Q4_K_XL") {
		t.Errorf("expected quant name in output, got: %s", out)
	}
	if !strings.Contains(out, "llama-cpp") {
		t.Errorf("expected backend in output, got: %s", out)
	}
	if !strings.Contains(out, "Recommended") {
		t.Errorf("expected recommendation line, got: %s", out)
	}
}

func TestRenderRecommendationEmpty(t *testing.T) {
	gpu := &host.GPUInfo{Name: "Tesla T4", TotalVRAMMB: 16384, SystemRAMMB: 28456}
	out := renderRecommendation(gpu, nil)
	if !strings.Contains(out, "No models fit") {
		t.Errorf("expected 'No models fit' message, got: %s", out)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cli/ -run TestRenderRecommendation -v`
Expected: FAIL — `renderRecommendation` undefined

- [ ] **Step 3: Implement recommend command**

Replace all of `cli/cmd_recommend.go`:

```go
package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var recommendCmd = &cobra.Command{
	Use:   "recommend [model]",
	Short: "Show models that fit your GPU, with benchmarks",
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		gpu, err := host.DetectGPU()
		if err != nil {
			fmt.Printf("Error detecting GPU: %v\n", err)
			os.Exit(1)
		}

		models, err := host.LoadModelDB()
		if err != nil {
			fmt.Printf("Error loading model DB: %v\n", err)
			os.Exit(1)
		}

		sizer := host.NewModelSizer(gpu)
		maxCtx := 32768
		if cmd.Flags().Changed("max-model-len") {
			maxCtx = serveMaxModelLen
		}
		results := sizer.FindFit(models, gpu.TotalVRAMMB*85/100, maxCtx)

		fmt.Print(renderRecommendation(gpu, results))
	},
}

func init() {
	recommendCmd.Flags().IntVar(&serveMaxModelLen, "max-model-len", 32768, "Max context tokens for sizing estimate")
}

func renderRecommendation(gpu *host.GPUInfo, results []*host.SizingResult) string {
	var buf strings.Builder

	fmt.Fprintf(&buf, "GPU: %s (%d MB VRAM, %d MB RAM)\n\n",
		gpu.Name, gpu.TotalVRAMMB, gpu.SystemRAMMB)

	if len(results) == 0 {
		buf.WriteString("No models fit on this GPU.\n")
		return buf.String()
	}

	fmt.Fprintf(&buf, "%-25s %-12s %-10s %10s %10s\n",
		"Model", "Quant", "Backend", "≈Size", "Headroom")
	buf.WriteString(strings.Repeat("-", 72))
	buf.WriteString("\n")

	for _, r := range results {
		fmt.Fprintf(&buf, "%-25s %-12s %-10s %8dMB %8dMB\n",
			r.Model.Name, r.Quantization.Name, r.Model.Backend,
			r.TotalMB, r.HeadroomMB)
	}

	best := results[0]
	fmt.Fprintf(&buf, "\nRecommended: %s %s (%.1f GB estimated)\n",
		best.Model.Name, best.Quantization.Name,
		float64(best.TotalMB)/1024)

	return buf.String()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./cli/ -run TestRenderRecommendation -v`
Expected: PASS

- [ ] **Step 5: Build and verify**

Run: `go build ./... && go vet ./...`
Expected: build and vet succeed

- [ ] **Step 6: Commit**

```bash
git add cli/cmd_recommend.go cli/cmd_recommend_test.go
git commit -m "feat: real recommend command with VRAM sizing table"
```

---

### Task 6: Delete tui/daemon Stubs + Fix root.go

**Files:**
- Delete: `cli/cmd_tui.go`
- Delete: `cli/cmd_daemon.go`
- Modify: `cli/root.go`

- [ ] **Step 1: Delete stub files**

Run: `rm cli/cmd_tui.go cli/cmd_daemon.go`

- [ ] **Step 2: Fix root.go — remove registrations + fix Use**

In `cli/root.go`, change `Use: "ai-host"` to `Use: "akilihost"`:

```go
var rootCmd = &cobra.Command{
	Use:   "akilihost",
	Short: "Self-hosted AI model manager for single GPU",
	Long: `akilihost automates everything painful about running open-weight LLMs on a single GPU:
- CUDA version detection
- vLLM version selection
- venv setup
- Cache management
- VRAM math
- Quantization selection
- Multi-model lifecycle management`,
}
```

Remove the tui/daemon registrations from `init()`:

```go
func init() {
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(psCmd)
	rootCmd.AddCommand(stopCmd)
	rootCmd.AddCommand(recommendCmd)
}
```

- [ ] **Step 3: Build and verify**

Run: `go build ./... && go vet ./...`
Expected: build and vet succeed (no references to tuiCmd/daemonCmd)

- [ ] **Step 4: Commit**

```bash
git add cli/root.go
git rm cli/cmd_tui.go cli/cmd_daemon.go
git commit -m "chore: delete tui/daemon stubs, fix root command name to akilihost"
```

---

### Task 7: Fix cmd_stop.go + cmd_ps.go Goroutine Race

**Files:**
- Modify: `cli/cmd_stop.go`
- Modify: `cli/cmd_ps.go`
- Create: `cli/cmd_ps_test.go`

- [ ] **Step 1: Remove unreachable branch in cmd_stop.go**

In `cli/cmd_stop.go`, remove lines 33-37 (the unreachable `stopByPID` branch). The `strconv.Atoi` check at line 28 already handles all numeric targets via `stopByPort`. The resulting `Run` function:

```go
	Run: func(cmd *cobra.Command, args []string) {
		target := args[0]

		// Try to stop as systemd service first
		serviceName := getSystemdServiceName(target)
		if serviceName != "" {
			stopSystemdService(serviceName)
			return
		}

		// Try to stop as port
		if port, err := strconv.Atoi(target); err == nil {
			stopByPort(port)
			return
		}

		// Try to stop by model name (matches part of command line)
		stopByModelName(target)
	},
```

- [ ] **Step 2: Write the failing test for checkHealth**

Create `cli/cmd_ps_test.go`:

```go
package cli

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCheckHealthResultOK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"id": "test-model"}},
		})
	}))
	defer ts.Close()

	got := checkHealthResult(ts.URL + "/v1/models")
	if !strings.Contains(got, "ok") {
		t.Errorf("expected health ok, got: %s", got)
	}
	if !strings.Contains(got, "test-model") {
		t.Errorf("expected model id in output, got: %s", got)
	}
}

func TestCheckHealthResultUnreachable(t *testing.T) {
	got := checkHealthResult("http://127.0.0.1:1/v1/models")
	if !strings.Contains(got, "unreachable") && !strings.Contains(got, "timeout") {
		t.Errorf("expected unreachable/timeout, got: %s", got)
	}
}

func TestCheckHealthResultNoModels(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data": []}`)
	}))
	defer ts.Close()

	got := checkHealthResult(ts.URL + "/v1/models")
	if strings.Contains(got, "ok (model:") {
		t.Errorf("should not report ok with empty data, got: %s", got)
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test ./cli/ -run TestCheckHealth -v`
Expected: FAIL — `checkHealthResult` undefined

- [ ] **Step 4: Refactor checkHealth to net/http + return string**

In `cli/cmd_ps.go`, add `"net/http"` and `"time"` to imports. Replace the `checkHealth` function:

```go
func checkHealthResult(url string) string {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "    Health: timeout/unreachable"
	}
	defer resp.Body.Close()

	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "    Health: error parsing response"
	}

	if models, ok := data["data"].([]interface{}); ok && len(models) > 0 {
		if model, ok := models[0].(map[string]interface{}); ok {
			if id, ok := model["id"].(string); ok {
				return fmt.Sprintf("    Health: ok (model: %s)", id)
			}
		}
	}
	return "    Health: no models in response"
}
```

- [ ] **Step 5: Fix goroutine race — make health checks synchronous**

In `checkSystemdServices`, replace `go checkHealth(port)` (line 61) with:

```go
			fmt.Println(checkHealthResult(fmt.Sprintf("http://localhost:%s/v1/models", port)))
```

In `checkRunningProcesses`, replace `go checkHealth(parts[i+1])` (line 96) with:

```go
						fmt.Println(checkHealthResult(fmt.Sprintf("http://localhost:%s/v1/models", parts[i+1])))
```

Remove the old `checkHealth` function (the one that printed directly).

- [ ] **Step 6: Run test to verify it passes**

Run: `go test ./cli/ -run TestCheckHealth -v`
Expected: PASS

- [ ] **Step 7: Build, vet, and run all tests**

Run: `go build ./... && go vet ./... && go test ./... -v`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add cli/cmd_stop.go cli/cmd_ps.go cli/cmd_ps_test.go
git commit -m "fix: remove unreachable stop branch, fix ps goroutine race, testable health checks"
```

---

### Task 8: Housekeeping — bin, .gitignore, models.yaml, README

**Files:**
- Delete: `bin/akilihost`
- Create: `.gitignore`
- Delete: `models.yaml`
- Modify: `host/models.go`
- Modify: `README.md`

- [ ] **Step 1: Delete committed binary and dead YAML**

Run: `git rm bin/akilihost models.yaml`

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

```
# Build output
bin/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
```

- [ ] **Step 3: Add honest comment to LoadModelDB**

In `host/models.go`, update the `LoadModelDB` comment:

```go
// LoadModelDB returns the curated in-memory model database.
// The database is hardcoded in prepopulatedModels (not loaded from YAML).
func LoadModelDB() ([]Model, error) {
```

- [ ] **Step 4: Update README flag defaults**

In `README.md`, update the Flags table:

```markdown
| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8002 | API port (auto-increments if busy unless explicitly set) |
| `--gpu-memory-utilization` | 0.90 | Max GPU memory fraction (vLLM) |
| `--max-model-len` | 32768 | Max context tokens |
```

Remove the `recommend` stub section description and update it to reflect the real command:

```markdown
### `recommend [model]`

Show models that fit your GPU with sizing estimates and benchmarks.

```bash
./akilihost recommend           # Show all fitting models
./akilihost recommend Qwen3     # Show specific model sizing
```
```

- [ ] **Step 5: Build and verify**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add .gitignore host/models.go README.md
git commit -m "chore: delete committed binary + dead models.yaml, add .gitignore, update README"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Full build + vet + test**

Run:
```bash
go build ./... && echo "BUILD OK" && go vet ./... && echo "VET OK" && go test ./... -v && echo "TESTS OK"
```
Expected: all three pass with no errors

- [ ] **Step 2: Verify CLI help output**

Run: `go run . --help`
Expected: shows `akilihost` as command name, lists `init`, `serve`, `ps`, `stop`, `recommend` (no `tui`, `daemon`)

Run: `go run . serve --help`
Expected: shows `--port`, `--gpu-memory-utilization`, `--max-model-len` flags

- [ ] **Step 3: Verify no dead files remain**

Run: `ls cli/cmd_*.go`
Expected: `cmd_init.go  cmd_ps.go  cmd_recommend.go  cmd_serve.go  cmd_stop.go` (no tui/daemon)

Run: `ls bin/ 2>/dev/null; ls models.yaml 2>/dev/null`
Expected: no output (both deleted)

---

## Verification Checklist

After all tasks complete:

- [ ] `go build ./...` succeeds
- [ ] `go vet ./...` succeeds
- [ ] `go test ./...` all green (host + cli packages)
- [ ] `akilihost --help` shows correct command name and no tui/daemon
- [ ] `akilihost serve --help` shows --port, --gpu-memory-utilization, --max-model-len
- [ ] `akilihost recommend` prints a sizing table (requires GPU — test on VM)
- [ ] `parseBParams` correctly parses "80B", "32B", "123B", "27B", "0.5B"
- [ ] `SizeModel` produces correct WeightsMB/KVCacheMB/TotalMB for known fixtures
- [ ] `resolvePort` errors on explicit-busy, auto-increments on default-busy
- [ ] `checkHealthResult` works with httptest servers
- [ ] No goroutine races in `cmd_ps.go`
- [ ] No unreachable code in `cmd_stop.go`
- [ ] No dead if/else in `cmd_serve.go`
- [ ] `bin/` in `.gitignore`, no committed binary
- [ ] `models.yaml` deleted
- [ ] README flag defaults match implementation (0.90, 32768)
