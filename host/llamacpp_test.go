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

	args := BuildLlamaServerCommand(model, quant, modelPath, port, 32768)

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

func TestModelDir(t *testing.T) {
	model := &Model{Name: "Qwen3.8-27B"}
	got := ModelDir(model)
	want := "/opt/akilihost/models/Qwen3.8-27B"
	if got != want {
		t.Errorf("ModelDir() = %q, want %q", got, want)
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
