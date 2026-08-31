package host

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPatchOpencodeCreate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opencode.json")
	backup, err := PatchOpencode(path, 8002)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if backup != "" {
		t.Error("backup should be empty for new file")
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "\"baseURL\": \"http://localhost:8002/v1\"") {
		t.Error("baseURL missing")
	}
	if !strings.Contains(string(data), "unsloth/Qwen3.8-27B-GGUF") {
		t.Error("model missing")
	}
}

func TestPatchOpencodeBackupAndMerge(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opencode.json")
	os.WriteFile(path, []byte(`{"provider":{"anthropic":{"options":{"apiKey":"x"}}},"model":"a"}`), 0644)
	backup, err := PatchOpencode(path, 8002)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if backup == "" {
		t.Error("backup expected")
	}
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, "anthropic") {
		t.Error("should preserve anthropic")
	}
	if !strings.Contains(s, "selfhosted") {
		t.Error("should add selfhosted")
	}
	if !strings.Contains(s, "8002") {
		t.Error("port")
	}
	// backup file exists
	if _, err := os.Stat(backup); err != nil {
		t.Error("backup file missing")
	}
}

func TestPatchOpencodeCustomPort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opencode.json")
	PatchOpencode(path, 8003)
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "8003") {
		t.Error("custom port")
	}
}
