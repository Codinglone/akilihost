package host

import "testing"

func TestParseBParams(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"80B", 80000},
		{"32B", 32000},
		{"123B", 123000},
		{"27B", 27000},
		{"7B", 7000},
		{"0.5B", 500},
		{"80B-40B", 80000},
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

	model := &Model{
		Name:        "Qwen2.5-Coder-32B",
		TotalParams: "32B",
		Layers:      64,
		KVHeads:     8,
		HeadDim:     128,
	}
	quant := &Quantization{Name: "BF16", BitsPerWeight: 16.0}

	r := sizer.SizeModel(model, quant, 32768)

	if r.WeightsMB != 61035 {
		t.Errorf("WeightsMB = %d, want 61035", r.WeightsMB)
	}
	if r.KVCacheMB != 8192 {
		t.Errorf("KVCacheMB = %d, want 8192", r.KVCacheMB)
	}
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

	if r.WeightsMB != 14483 {
		t.Errorf("WeightsMB = %d, want 14483", r.WeightsMB)
	}
	if r.KVCacheMB != 2048 {
		t.Errorf("KVCacheMB = %d, want 2048", r.KVCacheMB)
	}
	if r.TotalMB != 17043 {
		t.Errorf("TotalMB = %d, want 17043", r.TotalMB)
	}
}

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
	results := sizer.FindFit([]Model{qwen38}, gpu.TotalVRAMMB*85/100, 32768)
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

	results := sizer.FindFit([]Model{qwen3}, gpu.TotalVRAMMB*85/100, 32768)
	if len(results) != 0 {
		t.Fatal("Qwen3-Coder-Next (vLLM, 75GB) should not fit on T4")
	}
}
