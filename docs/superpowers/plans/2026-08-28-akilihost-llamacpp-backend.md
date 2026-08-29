# akilihost llama.cpp Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a llama.cpp backend to akilihost so Qwen3.8-27B (Unsloth dynamic 4-bit GGUF) runs on a single T4 GPU via split GPU/CPU inference, controllable with `akilihost serve/stop/ps`.

**Architecture:** A new `backend` field in `models.yaml` selects between `vllm` (existing, GPU-only) and `llama-cpp` (new, split GPU+CPU). The serve command auto-falls back to llama.cpp when a model's VRAM requirement exceeds available VRAM. llama-server runs as a systemd service bound to `127.0.0.1:8002`, reached by opencode through an SSH tunnel.

**Tech Stack:** Go 1.26 + cobra CLI, llama.cpp (CUDA build), systemd, HuggingFace Hub CLI, Azure CLI

---

## File Structure

**New files:**
- `host/backend.go` — backend selection logic + service name generation (pure functions, testable)
- `host/backend_test.go` — tests for backend selection + service names
- `host/llamacpp.go` — llama-server command builder + GGUF path resolution (pure functions, testable)
- `host/llamacpp_test.go` — tests for command builder
- `host/models_test.go` — tests for new struct fields + Qwen3.8-27B entry
- `host/gpu_test.go` — tests for SystemRAMMB parsing
- `host/sizers_test.go` — tests for split-inference fit logic

**Modified files:**
- `host/models.go` — add `Backend` + `FilePattern` fields, add Qwen3.8-27B entry
- `host/gpu.go` — add `SystemRAMMB` field + detection
- `host/sizers.go` — update `FindFit` to consider total memory for llama-cpp
- `cli/cmd_serve.go` — backend selection, GGUF download, llama-server systemd service
- `cli/cmd_stop.go` — handle `akilihost-` prefix + llama-server processes
- `cli/cmd_ps.go` — list `akilihost-` services + VRAM via nvidia-smi
- `cli/cmd_init.go` — llama.cpp build step + huggingface_hub install
- `scripts/ai-host-tunnel.sh` — add your-gpu-server as default server
- `~/.config/opencode/opencode.json` — add Qwen3.8-27B model (local, not in repo)

---

### Task 1: Add Backend and FilePattern Fields to Model Structs

**Files:**
- Modify: `host/models.go`
- Create: `host/models_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/models_test.go`:

```go
package host

import "testing"

func TestQwen38ModelHasLlamaCppBackend(t *testing.T) {
	models, err := LoadModelDB()
	if err != nil {
		t.Fatalf("LoadModelDB failed: %v", err)
	}

	var qwen38 *Model
	for i := range models {
		if models[i].Name == "Qwen3.8-27B" {
			qwen38 = &models[i]
			break
		}
	}
	if qwen38 == nil {
		t.Fatal("Qwen3.8-27B not found in model database")
	}

	if qwen38.Backend != "llama-cpp" {
		t.Errorf("expected backend 'llama-cpp', got '%s'", qwen38.Backend)
	}

	if len(qwen38.Quantizations) < 2 {
		t.Fatalf("expected at least 2 quantizations, got %d", len(qwen38.Quantizations))
	}

	q4 := qwen38.Quantizations[0]
	if q4.Name != "UD-Q4_K_XL" {
		t.Errorf("expected first quant 'UD-Q4_K_XL', got '%s'", q4.Name)
	}
	if q4.FilePattern != "*UD-Q4_K_XL*" {
		t.Errorf("expected file pattern '*UD-Q4_K_XL*', got '%s'", q4.FilePattern)
	}
	if q4.MinVRAMMB != 17408 {
		t.Errorf("expected min_vram_mb 17408, got %d", q4.MinVRAMMB)
	}
}

func TestDefaultBackendIsVllm(t *testing.T) {
	models, err := LoadModelDB()
	if err != nil {
		t.Fatalf("LoadModelDB failed: %v", err)
	}

	var qwen3 *Model
	for i := range models {
		if models[i].Name == "Qwen3-Coder-Next" {
			qwen3 = &models[i]
			break
		}
	}
	if qwen3 == nil {
		t.Fatal("Qwen3-Coder-Next not found")
	}

	if qwen3.Backend != "vllm" {
		t.Errorf("expected default backend 'vllm', got '%s'", qwen3.Backend)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestQwen38 -v`
Expected: FAIL — "Qwen3.8-27B not found in model database"

- [ ] **Step 3: Add Backend and FilePattern fields to structs**

In `host/models.go`, update the `Quantization` struct — add `FilePattern` field after `Flags`:

```go
type Quantization struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	DType       string   `yaml:"dtype"`
	QuantMode   string   `yaml:"quant_mode"`
	MinVRAMMB   int      `yaml:"min_vram_mb"`
	Flags       []string `yaml:"flags,omitempty"`
	FilePattern string   `yaml:"file_pattern,omitempty"`
}
```

Update the `Model` struct — add `Backend` field after `License`:

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
	Benchmarks     map[string]float32 `yaml:"benchmarks,omitempty"`
	Quantizations  []Quantization     `yaml:"quantizations"`
}
```

- [ ] **Step 4: Add Qwen3.8-27B to prepopulatedModels and set default backends**

In `host/models.go`, add a helper to default empty Backend to "vllm". Update `LoadModelDB`:

```go
func LoadModelDB() ([]Model, error) {
	models := make([]Model, len(prepopulatedModels))
	copy(models, prepopulatedModels)
	for i := range models {
		if models[i].Backend == "" {
			models[i].Backend = "vllm"
		}
	}
	return models, nil
}
```

Add the Qwen3.8-27B entry to `prepopulatedModels` (after the Devstral entry):

```go
	{
		RepoID:       "unsloth/Qwen3.8-27B-GGUF",
		Name:         "Qwen3.8-27B",
		Description:  "Qwen3.8 27B with vision + reasoning, 256K context",
		Architecture: "dense",
		TotalParams:  "27B",
		ActiveParams: "27B",
		ContextTok:   262144,
		License:      "Apache 2.0",
		Backend:      "llama-cpp",
		Benchmarks: map[string]float32{
			"HumanEval":      90.3,
			"SWE-bench":      61.7,
			"LiveCodeBench":  90.3,
		},
		Quantizations: []Quantization{
			{
				Name:        "UD-Q4_K_XL",
				Description: "Unsloth Dynamic 4-bit (~17GB, GPU+CPU split)",
				MinVRAMMB:   17408,
				FilePattern: "*UD-Q4_K_XL*",
				Flags:       []string{"--n-gpu-layers", "auto"},
			},
			{
				Name:        "UD-Q3_K_XL",
				Description: "Unsloth Dynamic 3-bit (~13GB, fits T4 fully)",
				MinVRAMMB:   13312,
				FilePattern: "*UD-Q3_K_XL*",
				Flags:       []string{"--n-gpu-layers", "999"},
			},
		},
	},
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestQwen38 -v`
Expected: PASS

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestDefaultBackend -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /tmp/opencode/akilihost
git add host/models.go host/models_test.go
git commit -m "feat: add Backend/FilePattern fields + Qwen3.8-27B model entry"
```

---

### Task 2: Add SystemRAMMB to GPUInfo

**Files:**
- Modify: `host/gpu.go`
- Create: `host/gpu_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/gpu_test.go`:

```go
package host

import "testing"

func TestParseSystemRAMMB(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"MemTotal:       28456320 kB\n", 28456},
		{"MemTotal:       16384 kB\n", 16},
		{"MemTotal:         1024 kB\n", 1},
		{"", 0},
	}

	for _, tt := range tests {
		got := parseSystemRAMMB(tt.input)
		if got != tt.expected {
			t.Errorf("parseSystemRAMMB(%q) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

func TestDetectGPUIncludesSystemRAM(t *testing.T) {
	gpu := &GPUInfo{
		Name:        "Tesla T4",
		TotalVRAMMB: 16384,
		SystemRAMMB: 28456,
	}
	if gpu.SystemRAMMB != 28456 {
		t.Errorf("expected SystemRAMMB 28456, got %d", gpu.SystemRAMMB)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestParseSystemRAM -v`
Expected: FAIL — `parseSystemRAMMB` undefined

- [ ] **Step 3: Add SystemRAMMB field and parsing**

In `host/gpu.go`, add `SystemRAMMB` to the `GPUInfo` struct:

```go
type GPUInfo struct {
	Name          string  `json:"name"`
	TotalVRAMMB   int     `json:"total_vram_mb"`
	SystemRAMMB   int     `json:"system_ram_mb"`
	CUDAMajor     int     `json:"cuda_major"`
	CUDAMinor     int     `json:"cuda_minor"`
	ComputeCap    string  `json:"compute_cap"`
	CUDAVersion   string  `json:"cuda_version"`
}
```

Add the `parseSystemRAMMB` function (after the existing `parseVRAM` function):

```go
func parseSystemRAMMB(meminfo string) int {
	for _, line := range strings.Split(meminfo, "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			line = strings.TrimPrefix(line, "MemTotal:")
			line = strings.TrimSpace(line)
			line = strings.TrimSuffix(line, "kB")
			line = strings.TrimSpace(line)
			kb, err := strconv.Atoi(line)
			if err != nil {
				return 0
			}
			return kb / 1024
		}
	}
	return 0
}
```

Add system RAM detection to `DetectGPU`. After the VRAM parsing and before the return, add:

```go
	// Detect system RAM from /proc/meminfo
	systemRAMMB := 0
	meminfo, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		systemRAMMB = parseSystemRAMMB(string(meminfo))
	}
```

Update the return statement to include `SystemRAMMB`:

```go
	return &GPUInfo{
		Name:          name,
		TotalVRAMMB:   vramMB,
		SystemRAMMB:   systemRAMMB,
		CUDAMajor:     cudaMajor,
		CUDAMinor:     cudaMinor,
		ComputeCap:    computeCap,
		CUDAVersion:   "CUDA " + strings.Join([]string{strconv.Itoa(cudaMajor), strconv.Itoa(cudaMinor)}, "."),
	}, nil
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /tmp/opencode/akilihost
git add host/gpu.go host/gpu_test.go
git commit -m "feat: add SystemRAMMB detection to GPUInfo"
```

---

### Task 3: Update Sizer for Split Inference

**Files:**
- Modify: `host/sizers.go`
- Create: `host/sizers_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/sizers_test.go`:

```go
package host

import "testing"

func TestFindFitLlamaCppUsesTotalMemory(t *testing.T) {
	gpu := &GPUInfo{
		Name:        "Tesla T4",
		TotalVRAMMB: 16384,
		SystemRAMMB: 28456,
	}
	sizer := NewModelSizer(gpu)

	models, _ := LoadModelDB()
	var qwen38 Model
	for _, m := range models {
		if m.Name == "Qwen3.8-27B" {
			qwen38 = m
			break
		}
	}

	// Q4 needs 17408 MB total, VRAM is 16384 — doesn't fit VRAM alone
	// but total memory (16384 + 28456 = 44840) is plenty
	results := sizer.FindFit([]Model{qwen38}, gpu.TotalVRAMMB*85/100)
	if len(results) == 0 {
		t.Fatal("Qwen3.8-27B should fit via split inference but FindFit returned no results")
	}

	if results[0].Model.Backend != "llama-cpp" {
		t.Errorf("expected llama-cpp backend, got '%s'", results[0].Model.Backend)
	}
}

func TestFindFitVllmRequiresVRAMOnly(t *testing.T) {
	gpu := &GPUInfo{
		Name:        "Tesla T4",
		TotalVRAMMB: 16384,
		SystemRAMMB: 28456,
	}
	sizer := NewModelSizer(gpu)

	// Qwen3-Coder-Next needs 75000 MB VRAM — won't fit on T4 even with RAM
	models, _ := LoadModelDB()
	var qwen3 Model
	for _, m := range models {
		if m.Name == "Qwen3-Coder-Next" {
			qwen3 = m
			break
		}
	}

	results := sizer.FindFit([]Model{qwen3}, gpu.TotalVRAMMB*85/100)
	if len(results) != 0 {
		t.Fatal("Qwen3-Coder-Next (vLLM, 75GB) should not fit on T4")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestFindFitLlamaCpp -v`
Expected: FAIL — "Qwen3.8-27B should fit via split inference but FindFit returned no results"

- [ ] **Step 3: Update FindFit to consider total memory for llama-cpp**

In `host/sizers.go`, update `FindFit`:

```go
func (s *ModelSizer) FindFit(models []Model, availableMB int) []*SizingResult {
	var results []*SizingResult
	totalMemoryMB := s.GPU.TotalVRAMMB + s.GPU.SystemRAMMB
	for _, model := range models {
		for _, q := range model.Quantizations {
			// llama-cpp backend can split across GPU + CPU RAM
			effectiveLimit := availableMB
			if model.Backend == "llama-cpp" {
				effectiveLimit = totalMemoryMB * 85 / 100
			}

			if q.MinVRAMMB <= effectiveLimit {
				r := s.SizeModel(&model, &q)
				if r.HeadroomMB >= 0 {
					results = append(results, r)
				}
			}
		}
	}
	return results
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /tmp/opencode/akilihost
git add host/sizers.go host/sizers_test.go
git commit -m "feat: sizer considers total memory for llama-cpp split inference"
```

---

### Task 4: Backend Selection Logic + Service Name Helpers

**Files:**
- Create: `host/backend.go`
- Create: `host/backend_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/backend_test.go`:

```go
package host

import "testing"

func TestSelectBackendLlamaCppExplicit(t *testing.T) {
	model := Model{Name: "Qwen3.8-27B", Backend: "llama-cpp"}
	quant := Quantization{MinVRAMMB: 17408}
	gpu := &GPUInfo{TotalVRAMMB: 16384, SystemRAMMB: 28456}

	backend := SelectBackend(&model, &quant, gpu)
	if backend != "llama-cpp" {
		t.Errorf("expected 'llama-cpp', got '%s'", backend)
	}
}

func TestSelectBackendVllmExplicit(t *testing.T) {
	model := Model{Name: "Qwen3-Coder-Next", Backend: "vllm"}
	quant := Quantization{MinVRAMMB: 75000}
	gpu := &GPUInfo{TotalVRAMMB: 163840, SystemRAMMB: 128000}

	backend := SelectBackend(&model, &quant, gpu)
	if backend != "vllm" {
		t.Errorf("expected 'vllm', got '%s'", backend)
	}
}

func TestSelectBackendAutoFallbackWhenVRAMTooSmall(t *testing.T) {
	model := Model{Name: "SomeModel", Backend: "vllm"}
	quant := Quantization{MinVRAMMB: 20000}
	gpu := &GPUInfo{TotalVRAMMB: 16384, SystemRAMMB: 28456}

	backend := SelectBackend(&model, &quant, gpu)
	if backend != "llama-cpp" {
		t.Errorf("expected auto-fallback to 'llama-cpp' when VRAM too small, got '%s'", backend)
	}
}

func TestSelectBackendVllmWhenFits(t *testing.T) {
	model := Model{Name: "SomeModel", Backend: "vllm"}
	quant := Quantization{MinVRAMMB: 10000}
	gpu := &GPUInfo{TotalVRAMMB: 16384, SystemRAMMB: 28456}

	backend := SelectBackend(&model, &quant, gpu)
	if backend != "vllm" {
		t.Errorf("expected 'vllm' when model fits VRAM, got '%s'", backend)
	}
}

func TestServiceNameLlamaCpp(t *testing.T) {
	model := Model{Name: "Qwen3.8-27B", Backend: "llama-cpp"}
	name := ServiceName(&model)
	if name != "akilihost-Qwen3.8-27B" {
		t.Errorf("expected 'akilihost-Qwen3.8-27B', got '%s'", name)
	}
}

func TestServiceNameVllm(t *testing.T) {
	model := Model{Name: "Qwen3-Coder-Next", Backend: "vllm"}
	name := ServiceName(&model)
	if name != "vllm-Qwen3-Coder-Next" {
		t.Errorf("expected 'vllm-Qwen3-Coder-Next', got '%s'", name)
	}
}

func TestServiceNameHandlesSpaces(t *testing.T) {
	model := Model{Name: "Devstral 2 123B", Backend: "vllm"}
	name := ServiceName(&model)
	if name != "vllm-Devstral-2-123B" {
		t.Errorf("expected 'vllm-Devstral-2-123B', got '%s'", name)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestSelectBackend -v`
Expected: FAIL — `SelectBackend` undefined

- [ ] **Step 3: Implement backend.go**

Create `host/backend.go`:

```go
package host

import "strings"

// SelectBackend decides which inference backend to use.
// If the model explicitly sets backend to "llama-cpp", use it.
// If the model is "vllm" but the quant's VRAM requirement exceeds
// available VRAM (85% of total), auto-fallback to "llama-cpp"
// (which can split across GPU + CPU RAM).
func SelectBackend(model *Model, quant *Quantization, gpu *GPUInfo) string {
	if model.Backend == "llama-cpp" {
		return "llama-cpp"
	}

	availableVRAM := gpu.TotalVRAMMB * 85 / 100
	if quant.MinVRAMMB > availableVRAM {
		return "llama-cpp"
	}

	return "vllm"
}

// ServiceName generates the systemd service name for a model.
// llama-cpp backend: "akilihost-<ModelName>" (spaces replaced with "-")
// vllm backend: "vllm-<ModelName>" (spaces replaced with "-")
func ServiceName(model *Model) string {
	prefix := "vllm"
	if model.Backend == "llama-cpp" {
		prefix = "akilihost"
	}
	sanitized := strings.ReplaceAll(model.Name, " ", "-")
	return prefix + "-" + sanitized
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /tmp/opencode/akilihost
git add host/backend.go host/backend_test.go
git commit -m "feat: backend selection logic + service name helpers"
```

---

### Task 5: Llama-Server Command Builder

**Files:**
- Create: `host/llamacpp.go`
- Create: `host/llamacpp_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/llamacpp_test.go`:

```go
package host

import (
	"strings"
	"testing"
)

func TestBuildLlamaServerCommand(t *testing.T) {
	model := &Model{
		Name:    "Qwen3.8-27B",
		RepoID:  "unsloth/Qwen3.8-27B-GGUF",
		Backend: "llama-cpp",
	}
	quant := &Quantization{
		Name:        "UD-Q4_K_XL",
		FilePattern: "*UD-Q4_K_XL*",
		Flags:       []string{"--n-gpu-layers", "auto"},
	}
	modelPath := "/opt/akilihost/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf"
	port := 8002

	args := BuildLlamaServerCommand(model, quant, modelPath, port)

	if args[0] != "llama-server" {
		t.Errorf("expected first arg 'llama-server', got '%s'", args[0])
	}

	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--model "+modelPath) {
		t.Errorf("expected --model %s in command, got: %s", modelPath, joined)
	}
	if !strings.Contains(joined, "--host 127.0.0.1") {
		t.Errorf("expected --host 127.0.0.1, got: %s", joined)
	}
	if !strings.Contains(joined, "--port 8002") {
		t.Errorf("expected --port 8002, got: %s", joined)
	}
	if !strings.Contains(joined, "--cache-type-k q8_0") {
		t.Errorf("expected --cache-type-k q8_0, got: %s", joined)
	}
	if !strings.Contains(joined, "--cache-type-v q8_0") {
		t.Errorf("expected --cache-type-v q8_0, got: %s", joined)
	}
	if !strings.Contains(joined, "--ctx-size 32768") {
		t.Errorf("expected --ctx-size 32768, got: %s", joined)
	}
	if !strings.Contains(joined, "--n-gpu-layers auto") {
		t.Errorf("expected quant flags --n-gpu-layers auto, got: %s", joined)
	}
}

func TestResolveGGUFPath(t *testing.T) {
	files := []string{
		"Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf",
		"Qwen3.8-27B-UD-Q4_K_XL-00002-of-00002.gguf",
		"Qwen3.8-27B-UD-Q3_K_XL.gguf",
	}
	pattern := "*UD-Q4_K_XL*"

	matched := ResolveGGUFFromList(files, pattern)
	if len(matched) == 0 {
		t.Fatal("expected 2 matching files for Q4 pattern, got 0")
	}
	if len(matched) != 2 {
		t.Errorf("expected 2 files, got %d", len(matched))
	}
}

func TestResolveGGUFPathSingleFile(t *testing.T) {
	files := []string{"Qwen3.8-27B-UD-Q3_K_XL.gguf"}
	pattern := "*UD-Q3_K_XL*"

	matched := ResolveGGUFFromList(files, pattern)
	if len(matched) != 1 {
		t.Errorf("expected 1 file, got %d", len(matched))
	}
	if matched[0] != "Qwen3.8-27B-UD-Q3_K_XL.gguf" {
		t.Errorf("expected 'Qwen3.8-27B-UD-Q3_K_XL.gguf', got '%s'", matched[0])
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -run TestBuildLlamaServer -v`
Expected: FAIL — `BuildLlamaServerCommand` undefined

- [ ] **Step 3: Implement llamacpp.go**

Create `host/llamacpp.go`:

```go
package host

import (
	"path/filepath"
	"strconv"
)

// BuildLlamaServerCommand constructs the llama-server argument list
// for serving a model with split GPU/CPU inference.
func BuildLlamaServerCommand(model *Model, quant *Quantization, modelPath string, port int) []string {
	args := []string{
		"llama-server",
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"--ctx-size", "32768",
	}
	args = append(args, quant.Flags...)
	return args
}

// ResolveGGUFFromList filters a list of filenames by the quant's file pattern,
// returning the matching GGUF files (may be multiple for split shards).
func ResolveGGUFFromList(files []string, pattern string) []string {
	var matched []string
	for _, f := range files {
		ok, _ := filepath.Match(pattern, f)
		if ok {
			matched = append(matched, f)
		}
	}
	return matched
}

// ModelDir returns the local directory path for a model's GGUF files.
func ModelDir(model *Model) string {
	return filepath.Join("/opt/akilihost/models", model.Name)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/opencode/akilihost && go test ./host/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /tmp/opencode/akilihost
git add host/llamacpp.go host/llamacpp_test.go
git commit -m "feat: llama-server command builder + GGUF path resolution"
```

---

### Task 6: Update cmd_serve.go for Backend Selection

**Files:**
- Modify: `cli/cmd_serve.go`

- [ ] **Step 1: Add backend selection to the serve command**

In `cli/cmd_serve.go`, add the import for the `host` package's new functions (already imported). Add `path/filepath` and `os` to imports:

```go
import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)
```

Replace the section after model/quant selection (after the `if modelToServe == nil` check, around line 98) with backend-aware logic. Replace from the "Show selection and confirm" comment through the end of the `Run` function:

```go
		if modelToServe == nil || quantization == nil {
			fmt.Println("No model selected")
			os.Exit(1)
		}

		// Select backend
		backend := host.SelectBackend(modelToServe, quantization, gpu)
		port := determinePort(modelToServe.RepoID)

		fmt.Printf("\nSelected:\n")
		fmt.Printf("  Model: %s\n", modelToServe.Name)
		fmt.Printf("  Repo: %s\n", modelToServe.RepoID)
		fmt.Printf("  Quantization: %s\n", quantization.Name)
		fmt.Printf("  Backend: %s\n", backend)
		fmt.Printf("  Port: %d\n", port)

		if backend == "llama-cpp" {
			fmt.Printf("\nPreparing llama.cpp backend...\n")
			serveLlamaCpp(modelToServe, quantization, port)
		} else {
			fmt.Printf("\nPreparing vLLM backend...\n")
			serveVllm(modelToServe, quantization, port)
		}

		fmt.Printf("\nVerifying...\n")
		waitAndVerify(port)
```

- [ ] **Step 2: Add serveLlamaCpp function**

Add this function at the end of `cli/cmd_serve.go`:

```go
func serveLlamaCpp(model *host.Model, quant *host.Quantization, port int) {
	modelDir := host.ModelDir(model)

	// Download GGUF if not present
	ggufPath := filepath.Join(modelDir, "model.gguf")
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		fmt.Printf("  Downloading %s %s...\n", model.Name, quant.Name)
		fmt.Printf("  Pattern: %s\n", quant.FilePattern)

		cmd := exec.Command("hf", "download", model.RepoID,
			"--local-dir", modelDir,
			"--include", quant.FilePattern)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("  Download failed: %v\n", err)
			os.Exit(1)
		}

		// Find the downloaded GGUF file(s)
		entries, err := os.ReadDir(modelDir)
		if err != nil {
			fmt.Printf("  Cannot read model dir: %v\n", err)
			os.Exit(1)
		}
		var files []string
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".gguf") {
				files = append(files, e.Name())
			}
		}
		matched := host.ResolveGGUFFromList(files, quant.FilePattern)
		if len(matched) == 0 {
			fmt.Printf("  No GGUF file matching %s found in %s\n", quant.FilePattern, modelDir)
			os.Exit(1)
		}
		ggufPath = filepath.Join(modelDir, matched[0])
		fmt.Printf("  Using: %s\n", ggufPath)
	} else {
		fmt.Printf("  Model already downloaded: %s\n", ggufPath)
	}

	// Build command and create systemd service
	args := host.BuildLlamaServerCommand(model, quant, ggufPath, port)
	serviceName := host.ServiceName(model)

	fmt.Printf("  Creating systemd service: %s\n", serviceName)
	createLlamaCppService(serviceName, args)

	fmt.Printf("\n  Starting service...\n")
	startService(serviceName)
}
```

- [ ] **Step 3: Add createLlamaCppService and refactor startService**

Add these functions at the end of `cli/cmd_serve.go`:

```go
func createLlamaCppService(serviceName string, args []string) {
	serviceContent := fmt.Sprintf(`[Unit]
Description=akilihost llama-server: %s
Documentation=https://github.com/Codinglone/akilihost
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStart=%s
ExecStop=/bin/kill -TERM $MAINPID

[Install]
WantedBy=multi-user.target
`, serviceName, strings.Join(args, " "))

	servicePath := "/etc/systemd/system/" + serviceName + ".service"
	fmt.Printf("  Writing service file: %s\n", servicePath)

	cmd := exec.Command("sudo", "bash", "-c", fmt.Sprintf("cat > %s", servicePath))
	cmd.Stdin = strings.NewReader(serviceContent)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to write service file: %s\n", string(output))
		return
	}

	fmt.Printf("  Reloading systemd daemon...\n")
	cmd = exec.Command("sudo", "systemctl", "daemon-reload")
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed: %s\n", string(output))
	} else {
		fmt.Println("  Done")
	}
}

func startService(serviceName string) {
	cmd := exec.Command("sudo", "systemctl", "start", serviceName)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to start: %s\n", string(output))
	} else {
		fmt.Printf("  Service started: %s\n", serviceName)
	}
}
```

- [ ] **Step 4: Rename old vLLM functions for clarity**

Rename `createSystemdService` to `createVllmService` and `startSystemdService` to `startVllmService`. Update the call site in the `Run` function's vLLM branch. Add the `serveVllm` wrapper:

```go
func serveVllm(model *host.Model, quant *host.Quantization, port int) {
	fmt.Printf("  Flags: %s\n", strings.Join(quant.Flags, " "))
	serviceName := host.ServiceName(model)
	fmt.Printf("\nCreating systemd service for %s...\n", model.Name)
	createVllmService(model, quant, port)
	fmt.Printf("\nStarting service...\n")
	startVllmService(model.Name)
}
```

Update `createVllmService` to use `host.ServiceName(model)` instead of the hardcoded `vllm-` prefix. Replace the `servicePath` line:

```go
	serviceName := host.ServiceName(model)
	servicePath := "/etc/systemd/system/" + serviceName + ".service"
```

And remove the old `enable` call (services are not enabled on boot per spec — the VM is shared).

- [ ] **Step 5: Build and verify**

Run: `cd /tmp/opencode/akilihost && go build -o /dev/null .`
Expected: build succeeds with no errors

- [ ] **Step 6: Commit**

```bash
cd /tmp/opencode/akilihost
git add cli/cmd_serve.go
git commit -m "feat: serve command supports llama-cpp backend with GGUF download"
```

---

### Task 7: Update cmd_stop.go for akilihost- Prefix

**Files:**
- Modify: `cli/cmd_stop.go`

- [ ] **Step 1: Update getSystemdServiceName**

In `cli/cmd_stop.go`, replace `getSystemdServiceName` with a version that queries the model database for service names:

```go
func getSystemdServiceName(target string) string {
	models, err := host.LoadModelDB()
	if err != nil {
		return ""
	}
	targetLower := strings.ToLower(target)
	for _, m := range models {
		nameLower := strings.ToLower(m.Name)
		if strings.Contains(nameLower, targetLower) || strings.Contains(targetLower, nameLower) {
			return host.ServiceName(&m)
		}
	}
	return ""
}
```

Add `"github.com/Codinglone/akilihost/host"` to the imports.

- [ ] **Step 2: Update stopByModelName to search for llama-server processes**

In `cli/cmd_stop.go`, update `stopByModelName` to also search for `llama-server`:

```go
func stopByModelName(name string) {
	fmt.Printf("Stopping model matching: %s\n", name)

	// Search for both vllm and llama-server processes
	for _, procPattern := range []string{"vllm serve", "llama-server"} {
		cmd := exec.Command("pgrep", "-af", procPattern)
		output, err := cmd.Output()
		if err != nil {
			continue
		}

		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		for _, line := range lines {
			if strings.Contains(line, name) {
				parts := strings.Fields(line)
				if len(parts) > 0 {
					stopByPID(parts[0])
				}
			}
		}
	}
}
```

- [ ] **Step 3: Build and verify**

Run: `cd /tmp/opencode/akilihost && go build -o /dev/null .`
Expected: build succeeds

- [ ] **Step 4: Commit**

```bash
cd /tmp/opencode/akilihost
git add cli/cmd_stop.go
git commit -m "feat: stop command handles akilihost- services + llama-server processes"
```

---

### Task 8: Update cmd_ps.go for akilihost- Services

**Files:**
- Modify: `cli/cmd_ps.go`

- [ ] **Step 1: Add llama-server service detection**

In `cli/cmd_ps.go`, add `"github.com/Codinglone/akilihost/host"` to imports. Replace `checkSystemdServices` with a dynamic version that queries the model DB:

```go
func checkSystemdServices() {
	fmt.Println("Systemd Services:")
	fmt.Println("-----------------")

	models, err := host.LoadModelDB()
	if err != nil {
		fmt.Printf("  Error loading model DB: %v\n", err)
		return
	}

	anyActive := false
	for _, m := range models {
		serviceName := host.ServiceName(&m)
		cmd := exec.Command("systemctl", "is-active", serviceName, "--quiet")
		if err := cmd.Run(); err != nil {
			continue
		}

		anyActive = true
		out, _ := exec.Command("systemctl", "status", serviceName, "--no-pager", "--lines=5").Output()
		lines := strings.Split(string(out), "\n")
		for _, l := range lines {
			if strings.Contains(l, "Active:") || strings.Contains(l, "Main PID") {
				fmt.Println("  ", strings.TrimSpace(l))
			}
		}
		fmt.Printf("  Model: %s\n", m.RepoID)
		fmt.Printf("  Backend: %s\n", m.Backend)

		// Get port from command line
		showCmd, _ := exec.Command("systemctl", "show", serviceName, "--property=ExecStart", "--no-pager").Output()
		cmdStr := strings.TrimSpace(string(showCmd))
		if port := extractPort(cmdStr); port != "" {
			fmt.Printf("  Port: %s\n", port)
			go checkHealth(port)
		}

		// Show VRAM usage
		showVRAM()
		fmt.Println()
	}

	if !anyActive {
		fmt.Println("  No active model services")
	}
}
```

- [ ] **Step 2: Add helper functions showVRAM and extractPort**

Add at the end of `cli/cmd_ps.go`:

```go
func showVRAM() {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}
	line := strings.TrimSpace(string(out))
	parts := strings.Split(line, ",")
	if len(parts) == 2 {
		used := strings.TrimSpace(parts[0])
		total := strings.TrimSpace(parts[1])
		fmt.Printf("  VRAM: %s / %s MiB\n", used, total)
	}
}

func extractPort(cmdStr string) string {
	// Parse ExecStart=llama-server ... --port 8002 ...
	// or ExecStart=vllm serve ... --port 8002 ...
	parts := strings.Fields(cmdStr)
	for i, p := range parts {
		if p == "--port" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}
```

- [ ] **Step 3: Update checkRunningProcesses to include llama-server**

In `cli/cmd_ps.go`, update `checkRunningProcesses` to search for both `vllm serve` and `llama-server`:

```go
func checkRunningProcesses() {
	fmt.Println("Running Processes:")
	fmt.Println("------------------")

	found := false
	for _, pattern := range []string{"vllm serve", "llama-server"} {
		cmd := exec.Command("pgrep", "-af", pattern)
		output, err := cmd.Output()
		if err != nil {
			continue
		}

		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		for _, line := range lines {
			if strings.Contains(line, pattern) && !strings.Contains(line, "grep") {
				found = true
				parts := strings.Fields(line)
				if len(parts) > 1 {
					pid := parts[0]
					fmt.Printf("  PID %s: %s\n", pid, pattern)
					for i, p := range parts {
						if p == "--port" && i+1 < len(parts) {
							fmt.Printf("    Port: %s\n", parts[i+1])
							go checkHealth(parts[i+1])
							break
						}
					}
				}
			}
		}
	}

	if !found {
		fmt.Println("  No model processes found")
	}
}
```

- [ ] **Step 4: Build and verify**

Run: `cd /tmp/opencode/akilihost && go build -o /dev/null .`
Expected: build succeeds

- [ ] **Step 5: Commit**

```bash
cd /tmp/opencode/akilihost
git add cli/cmd_ps.go
git commit -m "feat: ps command shows akilihost- services + VRAM usage"
```

---

### Task 9: Implement cmd_init.go with llama.cpp Build Step

**Files:**
- Modify: `cli/cmd_init.go`

- [ ] **Step 1: Implement the init command**

Replace the stub in `cli/cmd_init.go`:

```go
package cli

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Detect GPU, build llama.cpp, install vLLM venv, setup caches",
	Run: func(cmd *cobra.Command, args []string) {
		gpu, err := host.DetectGPU()
		if err != nil {
			fmt.Printf("Error detecting GPU: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("GPU: %s\n", gpu.Name)
		fmt.Printf("VRAM: %d MB (%.1f GB)\n", gpu.TotalVRAMMB, float64(gpu.TotalVRAMMB)/1024)
		fmt.Printf("System RAM: %d MB (%.1f GB)\n", gpu.SystemRAMMB, float64(gpu.SystemRAMMB)/1024)
		fmt.Printf("CUDA: %s\n\n", gpu.CUDAVersion)

		// Step 1: Build llama.cpp with CUDA
		fmt.Println("[1/3] Building llama.cpp with CUDA support...")
		buildLlamaCpp()

		// Step 2: Install huggingface_hub
		fmt.Println("\n[2/3] Installing huggingface_hub...")
		installHuggingFaceHub()

		// Step 3: Create model cache directory
		fmt.Println("\n[3/3] Creating model cache directory...")
		createModelDir()

		fmt.Println("\nDone! Ready to serve models.")
		fmt.Println("Run: akilihost serve Qwen3.8-27B")
	},
}

func buildLlamaCpp() {
	llamaDir := "/opt/akilihost/llama.cpp"

	if _, err := os.Stat("/usr/local/bin/llama-server"); err == nil {
		fmt.Println("  llama-server already installed, skipping build")
		return
	}

	// Install build dependencies
	fmt.Println("  Installing build dependencies...")
	deps := "build-essential cmake git pciutils libcurl4-openssl-dev"
	cmd := exec.Command("sudo", "bash", "-c", "apt-get update -y && apt-get install -y "+deps)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to install deps: %v\n", err)
		os.Exit(1)
	}

	// Clone llama.cpp
	fmt.Println("  Cloning llama.cpp...")
	cmd = exec.Command("sudo", "bash", "-c",
		fmt.Sprintf("mkdir -p /opt/akilihost && git clone https://github.com/ggml-org/llama.cpp %s", llamaDir))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to clone: %v\n", err)
		os.Exit(1)
	}

	// Build with CUDA
	fmt.Println("  Building with CUDA (this takes several minutes)...")
	buildCmd := fmt.Sprintf(
		"cd %s && cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON && "+
			"cmake --build build --config Release -j --clean-first --target llama-server && "+
			"cp build/bin/llama-server /usr/local/bin/",
		llamaDir)
	cmd = exec.Command("sudo", "bash", "-c", buildCmd)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Build failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("  llama-server installed to /usr/local/bin/")
}

func installHuggingFaceHub() {
	cmd := exec.Command("pip3", "install", "-U", "huggingface_hub[cli]")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  pip install failed (trying with --user): %v\n", err)
		cmd = exec.Command("pip3", "install", "--user", "-U", "huggingface_hub[cli]")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("  Failed to install huggingface_hub: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Println("  huggingface_hub installed")
}

func createModelDir() {
	cmd := exec.Command("sudo", "mkdir", "-p", "/opt/akilihost/models")
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed: %v\n", err)
		os.Exit(1)
	}
	// Make it owned by current user
	cmd = exec.Command("sudo", "chown", "-R",
		fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()), "/opt/akilihost/models")
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to chown: %v\n", err)
	}
	fmt.Println("  Model directory: /opt/akilihost/models/")
}
```

- [ ] **Step 2: Build and verify**

Run: `cd /tmp/opencode/akilihost && go build -o /dev/null .`
Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
cd /tmp/opencode/akilihost
git add cli/cmd_init.go
git commit -m "feat: init command builds llama.cpp with CUDA + installs huggingface_hub"
```

---

### Task 10: Resize VM Disk to 256GB

**Files:**
- None (Azure CLI operations)

- [ ] **Step 1: Deallocate the VM**

Run:
```bash
az vm deallocate -g YOUR_RESOURCE_GROUP -n your-gpu-server
```
Expected: VM deallocated successfully

- [ ] **Step 2: Find the OS disk name and resize to 256GB**

Run:
```bash
DISK=$(az vm show -g YOUR_RESOURCE_GROUP -n your-gpu-server --query "storageProfile.osDisk.name" -o tsv)
echo "OS disk: $DISK"
az disk update -g YOUR_RESOURCE_GROUP -n "$DISK" --size-gb 256
```
Expected: disk updated to 256GB

- [ ] **Step 3: Start the VM**

Run:
```bash
az vm start -g YOUR_RESOURCE_GROUP -n your-gpu-server
```
Expected: VM running

- [ ] **Step 4: Grow the filesystem inside the VM**

Run:
```bash
ssh your-gpu-server 'sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1 && df -h / | tail -1'
```
Expected: filesystem shows ~256GB

---

### Task 11: Install CUDA Toolkit on VM

**Files:**
- None (remote operations on VM)

- [ ] **Step 1: Install cuda-toolkit via the NVIDIA repo already added**

Run:
```bash
ssh your-gpu-server 'sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8 && echo "=== nvcc ===" && nvcc --version'
```
Expected: nvcc reports CUDA 12.8 (or similar)

If `nvcc` is not in PATH after install, add it:
```bash
ssh your-gpu-server 'echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc && export PATH=/usr/local/cuda/bin:$PATH && nvcc --version'
```

- [ ] **Step 2: Verify CUDA + driver together**

Run:
```bash
ssh your-gpu-server 'nvidia-smi | head -5 && echo "---" && nvcc --version | tail -2'
```
Expected: both nvidia-smi and nvcc report compatible CUDA versions

---

### Task 12: Build, Deploy, and Wire opencode

**Files:**
- Modify: `scripts/ai-host-tunnel.sh`
- Modify (local, not in repo): `~/.config/opencode/opencode.json`

- [ ] **Step 1: Update the tunnel script default server**

In `scripts/ai-host-tunnel.sh`, change the default server from `your-server` to `your-gpu-server`:

```bash
SERVER="${AI_HOST_SERVER:-your-gpu-server}"
```

- [ ] **Step 2: Build akilihost binary**

Run:
```bash
cd /tmp/opencode/akilihost && go build -o bin/akilihost .
```
Expected: `bin/akilihost` created

- [ ] **Step 3: Copy binary to VM**

Run:
```bash
scp bin/akilihost your-gpu-server:~/akilihost
ssh your-gpu-server 'sudo mv ~/akilihost /usr/local/bin/akilihost && sudo chmod +x /usr/local/bin/akilihost'
```
Expected: `akilihost` available in PATH on VM

- [ ] **Step 4: Run akilihost init on the VM**

Run:
```bash
ssh your-gpu-server 'akilihost init'
```
Expected: llama-server built and installed, huggingface_hub installed, model dir created

- [ ] **Step 5: Start Qwen3.8-27B on the VM**

Run:
```bash
ssh your-gpu-server 'akilihost serve Qwen3.8-27B'
```
Expected: GGUF downloads (~17GB, takes several minutes), llama-server starts, verification shows model ready on port 8002

- [ ] **Step 6: Verify GPU usage**

Run:
```bash
ssh your-gpu-server 'akilihost ps && nvidia-smi'
```
Expected: ps shows Qwen3.8-27B running, nvidia-smi shows ~13-14GB VRAM used

- [ ] **Step 7: Update opencode.json locally**

Read `~/.config/opencode/opencode.json`, then update the `selfhosted` provider models to include Qwen3.8-27B and set it as the default model:

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
        },
        "Qwen/Qwen3-Coder-Next": {
          "name": "Qwen3-Coder-Next FP8 (262K ctx)"
        },
        "Qwen/Qwen2.5-Coder-32B-Instruct": {
          "name": "Qwen2.5-Coder 32B (HumanEval 92.7%)"
        }
      }
    }
  },
  "model": "selfhosted/unsloth/Qwen3.8-27B-GGUF"
}
```

- [ ] **Step 8: Start the SSH tunnel and test end-to-end**

Run:
```bash
./scripts/ai-host-tunnel.sh start
sleep 3
curl -s http://localhost:8002/v1/models | python3 -m json.tool
```
Expected: JSON response listing `unsloth/Qwen3.8-27B-GGUF` (or the GGUF filename as model ID)

- [ ] **Step 9: Test with a chat completion**

Run:
```bash
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen3.8-27B-GGUF","messages":[{"role":"user","content":"Write a Python function that returns the factorial of n."}],"max_tokens":200}' \
  | python3 -m json.tool
```
Expected: JSON response with generated code in `choices[0].message.content`

- [ ] **Step 10: Launch opencode and verify**

Run:
```bash
opencode
```
Expected: opencode starts with `selfhosted/unsloth/Qwen3.8-27B-GGUF` as the model, can send a coding prompt and get a streamed response

- [ ] **Step 11: Commit all remaining changes**

```bash
cd /tmp/opencode/akilihost
git add scripts/ai-host-tunnel.sh
git commit -m "feat: update tunnel script default to gpu-server"
git push origin main
```

---

## Verification Checklist

After all tasks complete:

- [ ] `akilihost serve Qwen3.8-27B` starts llama-server on port 8002
- [ ] `akilihost ps` shows the model running with VRAM usage
- [ ] `akilihost stop Qwen3.8-27B` stops the service and frees VRAM
- [ ] `akilihost serve Qwen3.8-27B` restarts after stop (switch workflow works)
- [ ] SSH tunnel connects opencode to the model API
- [ ] opencode can send prompts and receive streamed responses
- [ ] Disk is 256GB with adequate free space
- [ ] `go test ./host/ -v` all tests pass
