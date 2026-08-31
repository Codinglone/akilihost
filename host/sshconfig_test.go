package host

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnsureSSHConfigCreate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config")
	os.WriteFile(path, []byte("# existing\nHost other\n  HostName 1.1.1.1\n"), 0644)

	created, err := EnsureSSHConfig(path, "mygpu", "1.2.3.4", "ubuntu", "/home/u/.ssh/id_ed25519")
	if err != nil { t.Fatalf("err: %v", err) }
	if !created { t.Fatalf("expected created") }
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, "Host mygpu") { t.Errorf("missing Host mygpu") }
	if !strings.Contains(s, "LocalForward 8002") { t.Errorf("missing LocalForward") }
	if !strings.Contains(s, "Host other") { t.Errorf("should preserve other") }
}

func TestEnsureSSHConfigPreserveGcloudBlock(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config")
	gcloud := "# Google Compute Engine Section\nHost gce\n  HostName 35.1.1.1\n# End of Google Compute Engine Section\n"
	os.WriteFile(path, []byte(gcloud), 0644)
	EnsureSSHConfig(path, "mygpu", "1.2.3.4", "ubuntu", "~/.ssh/id")
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "Google Compute Engine Section") { t.Error("gcloud block lost") }
	if !strings.Contains(string(data), "Host mygpu") { t.Error("mygpu not added") }
}

func TestEnsureSSHConfigUpdateExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config")
	os.WriteFile(path, []byte("Host mygpu\n  HostName oldhost\n  User olduser\n"), 0644)
	created, err := EnsureSSHConfig(path, "mygpu", "newhost", "newuser", "/new/key")
	if err != nil { t.Fatalf("err: %v", err) }
	if created { t.Error("should not be created, updated") }
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "HostName newhost") { t.Error("not updated") }
}

func TestEnsureSSHConfigIdempotent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config")
	os.WriteFile(path, []byte(""), 0644)
	EnsureSSHConfig(path, "mygpu", "1.2.3.4", "ubuntu", "/k")
	data1, _ := os.ReadFile(path)
	EnsureSSHConfig(path, "mygpu", "1.2.3.4", "ubuntu", "/k")
	data2, _ := os.ReadFile(path)
	if string(data1) != string(data2) { t.Error("not idempotent") }
}
