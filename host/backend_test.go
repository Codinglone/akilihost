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
