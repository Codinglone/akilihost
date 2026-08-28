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
