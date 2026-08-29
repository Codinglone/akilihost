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
