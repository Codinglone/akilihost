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

func TestModelArchitectureFields(t *testing.T) {
	models, err := LoadModelDB()
	if err != nil {
		t.Fatalf("LoadModelDB failed: %v", err)
	}

	want := map[string]struct {
		Layers  int
		KVHeads int
		HeadDim int
	}{
		"Qwen3-Coder-Next":  {12, 2, 256},
		"Qwen2.5-Coder-32B": {64, 8, 128},
		"Devstral 2 123B":   {88, 8, 128},
		"Qwen3.8-27B":       {16, 4, 256},
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
