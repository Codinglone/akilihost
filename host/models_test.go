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
