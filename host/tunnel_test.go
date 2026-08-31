package host

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteTunnelService(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ai-host-tunnel.service")
	err := WriteTunnelService(path, "mygpu", 8002)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, "mygpu") {
		t.Error("alias missing")
	}
	if !strings.Contains(s, "8002:localhost:8002") {
		t.Error("forward missing")
	}
	if strings.Contains(s, "myserver") {
		t.Error("should not contain myserver")
	}
}

func TestWriteTunnelServiceCustomPort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ai-host-tunnel.service")
	WriteTunnelService(path, "mygpu", 8003)
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "8003") {
		t.Error("custom port")
	}
}
