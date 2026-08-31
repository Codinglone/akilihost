package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Codinglone/akilihost/host"
)

func TestConnectCommandExists(t *testing.T) {
	cmd, _, err := rootCmd.Find([]string{"connect"})
	if err != nil || cmd == nil {
		t.Fatalf("connect command not found")
	}
	if cmd.Flags().Lookup("host") == nil {
		t.Error("missing --host")
	}
	if cmd.Flags().Lookup("alias") == nil {
		t.Error("missing --alias")
	}
}

func TestConnectDryRun(t *testing.T) {
	dir := t.TempDir()
	sshPath := filepath.Join(dir, "ssh_config")
	opPath := filepath.Join(dir, "opencode.json")
	tunnelPath := filepath.Join(dir, "tunnel.service")
	_, err := host.EnsureSSHConfig(sshPath, "mygpu", "1.2.3.4", "ubuntu", "/k")
	if err != nil {
		t.Fatalf("ssh: %v", err)
	}
	_, err = host.PatchOpencode(opPath, 8002)
	if err != nil {
		t.Fatalf("opencode: %v", err)
	}
	err = host.WriteTunnelService(tunnelPath, "mygpu", 8002)
	if err != nil {
		t.Fatalf("tunnel: %v", err)
	}
	data, _ := os.ReadFile(sshPath)
	if !strings.Contains(string(data), "mygpu") {
		t.Error("ssh config alias missing")
	}
	data, _ = os.ReadFile(opPath)
	if !strings.Contains(string(data), "8002") {
		t.Error("opencode port missing")
	}
	data, _ = os.ReadFile(tunnelPath)
	if !strings.Contains(string(data), "mygpu") {
		t.Error("tunnel alias missing")
	}
	plan := runConnectDryRun("mygpu", "1.2.3.4", "ubuntu", "/k", 8002)
	if !strings.Contains(plan, "mygpu") {
		t.Errorf("plan missing alias, got %q", plan)
	}
	if !strings.Contains(plan, "1.2.3.4") {
		t.Errorf("plan missing host, got %q", plan)
	}
	if !strings.Contains(plan, "8002") {
		t.Errorf("plan missing port, got %q", plan)
	}
}

func TestConnectValidation(t *testing.T) {
	origPort := connectPort
	origTunnelPort := connectTunnelPort
	origAlias := connectAlias
	origHost := connectHost
	t.Cleanup(func() {
		connectPort = origPort
		connectTunnelPort = origTunnelPort
		connectAlias = origAlias
		connectHost = origHost
	})
	connectPort = 99999
	if err := validateConnectArgs(); err == nil {
		t.Error("expected error for invalid port 99999")
	}
	connectPort = 8002
	connectTunnelPort = 99999
	if err := validateConnectArgs(); err == nil {
		t.Error("expected error for invalid tunnel-port 99999")
	}
	connectTunnelPort = 8002
	connectAlias = "invalid alias"
	if err := validateConnectArgs(); err == nil {
		t.Error("expected error for invalid alias with space")
	}
	connectAlias = "mygpu"
	connectHost = "invalid host"
	if err := validateConnectArgs(); err == nil {
		t.Error("expected error for invalid host with space")
	}
}

func TestConnectFlags(t *testing.T) {
	cmd, _, err := rootCmd.Find([]string{"connect"})
	if err != nil || cmd == nil {
		t.Fatalf("connect not found")
	}
	for _, f := range []string{"host", "alias", "user", "key", "port", "tunnel-port", "yes", "dry-run"} {
		if cmd.Flags().Lookup(f) == nil {
			t.Errorf("missing --%s", f)
		}
	}
	if cmd.Args == nil {
		t.Error("Args validator missing")
	}
	if cmd.Flags().Lookup("port") == nil || cmd.Flags().Lookup("dry-run") == nil {
		t.Error("flags missing")
	}
}

func TestConnectDryRunFlag(t *testing.T) {
	origDryRun := connectDryRun
	origHost := connectHost
	origAlias := connectAlias
	origPort := connectPort
	origTunnelPort := connectTunnelPort
	t.Cleanup(func() {
		connectDryRun = origDryRun
		connectHost = origHost
		connectAlias = origAlias
		connectPort = origPort
		connectTunnelPort = origTunnelPort
	})
	connectDryRun = true
	connectHost = "1.2.3.4"
	connectAlias = "mygpu"
	connectPort = 8002
	connectTunnelPort = 8002
	// Use a fake cobra command with buffer
	cmd, _, err := rootCmd.Find([]string{"connect"})
	if err != nil || cmd == nil {
		t.Fatalf("connect not found")
	}
	var buf strings.Builder
	cmd.SetOut(&buf)
	cmd.SetErr(&buf)
	// call runConnect directly with dry-run should not error and should contain alias
	err = runConnect(cmd, []string{})
	if err != nil {
		t.Fatalf("runConnect dry-run failed: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "mygpu") && !strings.Contains(out, "1.2.3.4") {
		t.Errorf("dry-run output missing alias/host, got %q", out)
	}
}
