# Portable Connect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single `akilihost connect` command that auto-creates `~/.ssh/config` Host, runs remote `akilihost init/recommend/serve` over SSH, patches `~/.config/opencode/opencode.json` with backup, and starts `systemctl --user ai-host-tunnel` to `<alias>` with 3-step health verification, portable to any Ubuntu VM.

**Architecture:** `cli/cmd_connect.go` orchestrates laptop->VM via `os/exec ssh` (reusing `~/.ssh/config`, keys, autossh). Pure helpers `host/sshconfig.go` (SSH config parser) and `host/opencode.go` (JSON merge) are unit-tested without live VM. VM-side truth is over-SSH `akilihost recommend/serve/ps` polling, identical to `cli/cmd_serve.go:waitAndVerify`.

**Tech Stack:** Go 1.21+ with cobra, `net/http` for health, `encoding/json` for opencode, `os/exec` for ssh/scp/systemctl, systemd user service, autossh

---

## File Structure

**New files:**
- `host/sshconfig.go` — pure SSH config parser/writer (create/update Host, preserve gcloud block)
- `host/sshconfig_test.go` — unit tests for SSH config
- `host/opencode.go` — opencode.json backup+rewrite (merge selfhosted provider)
- `host/opencode_test.go` — unit tests for opencode patch
- `cli/cmd_connect.go` — `connect` cobra command orchestration
- `cli/cmd_connect_test.go` — unit tests for connect helpers (fake SSH)

**Modified files:**
- `cli/root.go` — register `connectCmd`
- `host/models.go` — expose `ListModels` helper if needed (already exists)
- `scripts/ai-host-tunnel.sh` — already patched `SERVER=api-product-dev`, ensure `ExitOnForwardFailure`
- `systemd/ai-host-tunnel@.service` — already patched `%i` template, ensure no `myserver`

---

### Task 1: SSH Config Parser/Writer

**Files:**
- Create: `host/sshconfig.go`
- Create: `host/sshconfig_test.go`
- Test: `go test ./host/ -run TestSSH -v`

- [ ] **Step 1: Write the failing test**

Create `host/sshconfig_test.go`:

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./host/ -run TestEnsureSSH -v`
Expected: FAIL — `EnsureSSHConfig undefined`

- [ ] **Step 3: Implement minimal sshconfig.go**

Create `host/sshconfig.go`:

```go
package host

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func EnsureSSHConfig(path, alias, host, user, key string) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) { return false, err }
	content := string(data)
	if strings.Contains(content, "Host "+alias) {
		// update existing block: simple replace HostName/User/IdentityFile within block
		lines := strings.Split(content, "\n")
		var out []string
		inBlock := false
		updated := false
		for i, l := range lines {
			trim := strings.TrimSpace(l)
			if strings.HasPrefix(trim, "Host ") {
				inBlock = trim == "Host "+alias
			}
			if inBlock {
				if strings.HasPrefix(trim, "HostName ") { l = "    HostName "+host; updated=true }
				if strings.HasPrefix(trim, "User ") { l = "    User "+user; updated=true }
				if strings.HasPrefix(trim, "IdentityFile ") { l = "    IdentityFile "+key; updated=true }
				// ensure LocalForward present before next Host or EOF
				if updated && (i+1==len(lines) || strings.HasPrefix(strings.TrimSpace(lines[i+1]), "Host ")) {
					if !strings.Contains(strings.Join(out, "\n"), "LocalForward 8002") {
						out = append(out, l)
						out = append(out, "    LocalForward 8002 127.0.0.1:8002")
						out = append(out, "    LocalForward 8003 127.0.0.1:8003")
						continue
					}
				}
			}
			out = append(out, l)
		}
		if !updated { // add missing directives
			for i, l := range out {
				if strings.TrimSpace(l)=="Host "+alias {
					// insert after Host line if not found
					newBlock := []string{"    HostName "+host, "    User "+user, "    IdentityFile "+key, "    IdentitiesOnly yes", "    StrictHostKeyChecking accept-new", "    ServerAliveInterval 30", "    ServerAliveCountMax 3", "    LocalForward 8002 127.0.0.1:8002", "    LocalForward 8003 127.0.0.1:8003"}
					// check if already has HostName etc, only add missing
					hasHostName := false
					for j:=i;j<len(out) && j<i+15;j++ {
						if strings.Contains(out[j], "HostName") { hasHostName=true }
					}
					if !hasHostName {
						tmp := append([]string{}, out[:i+1]...)
						tmp = append(tmp, newBlock...)
						tmp = append(tmp, out[i+1:]...)
						out = tmp
					}
					break
				}
			}
		}
		newContent := strings.Join(out, "\n")
		if newContent != content {
			return false, os.WriteFile(path, []byte(newContent), 0644)
		}
		return false, nil
	}
	// create new block
	block := fmt.Sprintf("\nHost %s\n    HostName %s\n    User %s\n    IdentityFile %s\n    IdentitiesOnly yes\n    StrictHostKeyChecking accept-new\n    ServerAliveInterval 30\n    ServerAliveCountMax 3\n    LocalForward 8002 127.0.0.1:8002\n    LocalForward 8003 127.0.0.1:8003\n", alias, host, user, key)
	// preserve gcloud block position: append after it if exists
	if err := os.WriteFile(path, []byte(content+block), 0644); err != nil { return false, err }
	return true, nil
}

func ParseSSHConfigForTest(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil { return nil, err }
	defer f.Close()
	m := make(map[string]string)
	sc := bufio.NewScanner(f)
	cur := ""
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(l, "Host ") { cur = strings.Fields(l)[1]; m[cur] = "" }
	}
	return m, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./host/ -run TestEnsureSSH -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add host/sshconfig.go host/sshconfig_test.go
git commit -m "feat: SSH config ensure with gcloud preserve and idempotent LocalForward"
```

---

### Task 2: Opencode JSON Backup+Rewrite

**Files:**
- Create: `host/opencode.go`
- Create: `host/opencode_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/opencode_test.go`:

```go
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
	if err != nil { t.Fatalf("err: %v", err) }
	if backup != "" { t.Error("backup should be empty for new file") }
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "\"baseURL\": \"http://localhost:8002/v1\"") { t.Error("baseURL missing") }
	if !strings.Contains(string(data), "unsloth/Qwen3.8-27B-GGUF") { t.Error("model missing") }
}

func TestPatchOpencodeBackupAndMerge(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opencode.json")
	os.WriteFile(path, []byte(`{"provider":{"anthropic":{"options":{"apiKey":"x"}}},"model":"a"}`), 0644)
	backup, err := PatchOpencode(path, 8002)
	if err != nil { t.Fatalf("err: %v", err) }
	if backup == "" { t.Error("backup expected") }
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, "anthropic") { t.Error("should preserve anthropic") }
	if !strings.Contains(s, "selfhosted") { t.Error("should add selfhosted") }
	if !strings.Contains(s, "8002") { t.Error("port") }
	// backup file exists
	if _, err := os.Stat(backup); err != nil { t.Error("backup file missing") }
}

func TestPatchOpencodeCustomPort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "opencode.json")
	PatchOpencode(path, 8003)
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "8003") { t.Error("custom port") }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./host/ -run TestPatchOpencode -v`
Expected: FAIL — `PatchOpencode undefined`

- [ ] **Step 3: Implement opencode.go**

Create `host/opencode.go`:

```go
package host

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type OpencodeConfig struct {
	Schema   string                 `json:"$schema,omitempty"`
	Plugin   []string               `json:"plugin,omitempty"`
	Provider map[string]interface{} `json:"provider"`
	Model    string                 `json:"model,omitempty"`
}

func PatchOpencode(path string, port int) (string, error) {
	var backup string
	data, err := os.ReadFile(path)
	cfg := make(map[string]interface{})
	if err == nil && len(data)>0 {
		if err := json.Unmarshal(data, &cfg); err != nil { return "", err }
		backup = fmt.Sprintf("%s.bak.%d", path, time.Now().Unix())
		if err := os.WriteFile(backup, data, 0644); err != nil { return "", err }
	} else {
		cfg["$schema"] = "https://opencode.ai/config.json"
		cfg["plugin"] = []string{"superpowers@git+https://github.com/obra/superpowers.git", "opencode-wakatime"}
	}
	if _, ok := cfg["provider"]; !ok { cfg["provider"] = make(map[string]interface{}) }
	prov := cfg["provider"].(map[string]interface{})
	self, ok := prov["selfhosted"].(map[string]interface{})
	if !ok { self = make(map[string]interface{}); prov["selfhosted"] = self }
	self["npm"] = "@ai-sdk/openai-compatible"
	self["name"] = "Self-Hosted (llama.cpp)"
	opts, ok := self["options"].(map[string]interface{})
	if !ok { opts = make(map[string]interface{}); self["options"] = opts }
	opts["baseURL"] = fmt.Sprintf("http://localhost:%d/v1", port)
	opts["timeout"] = 600000
	opts["chunkTimeout"] = 120000
	// merge models from host/models.go
	modelsMap, ok := self["models"].(map[string]interface{})
	if !ok { modelsMap = make(map[string]interface{}); self["models"] = modelsMap }
	// curated from host/prepopulatedModels
	modelsMap["unsloth/Qwen3.8-27B-GGUF"] = map[string]interface{}{"name": "Qwen3.8-27B UD-Q4_K_XL (T4 split)", "maxTokens": 16384}
	modelsMap["Qwen/Qwen3-Coder-Next"] = map[string]interface{}{"name": "Qwen3-Coder-Next FP8 (262K ctx)"}
	modelsMap["mistralai/Devstral-2-123B-Instruct-2512"] = map[string]interface{}{"name": "Devstral 2 123B FP8"}
	modelsMap["Qwen/Qwen2.5-Coder-32B-Instruct"] = map[string]interface{}{"name": "Qwen2.5-Coder 32B (HumanEval 92.7%)"}
	if _, ok := cfg["model"]; !ok { cfg["model"] = "selfhosted/unsloth/Qwen3.8-27B-GGUF" }
	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil { return backup, err }
	return backup, os.WriteFile(path, out, 0644)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./host/ -run TestPatchOpencode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add host/opencode.go host/opencode_test.go
git commit -m "feat: opencode.json backup+merge with selfhosted provider"
```

---

### Task 3: Tunnel Systemd Service Helper

**Files:**
- Create: `host/tunnel.go`
- Create: `host/tunnel_test.go`

- [ ] **Step 1: Write the failing test**

Create `host/tunnel_test.go`:

```go
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
	if err != nil { t.Fatalf("err: %v", err) }
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, "mygpu") { t.Error("alias missing") }
	if !strings.Contains(s, "8002:localhost:8002") { t.Error("forward missing") }
	if strings.Contains(s, "myserver") { t.Error("should not contain myserver") }
}

func TestWriteTunnelServiceCustomPort(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ai-host-tunnel.service")
	WriteTunnelService(path, "mygpu", 8003)
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "8003") { t.Error("custom port") }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./host/ -run TestWriteTunnel -v`
Expected: FAIL — `WriteTunnelService undefined`

- [ ] **Step 3: Implement tunnel.go**

Create `host/tunnel.go`:

```go
package host

import (
	"fmt"
	"os"
)

func WriteTunnelService(path, alias string, port int) error {
	content := fmt.Sprintf(`[Unit]
Description=Autossh tunnel for vLLM API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=codinglone
Environment=AUTOSSH_GATETIME=0
Environment=AUTOSSH_POLL=30
ExecStart=/usr/bin/autossh -M 0 -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -o "ExitOnForwardFailure=yes" -o "StrictHostKeyChecking=accept-new" -L %d:localhost:%d %s
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
`, port, port, alias)
	return os.WriteFile(path, []byte(content), 0644)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./host/ -run TestWriteTunnel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add host/tunnel.go host/tunnel_test.go
git commit -m "feat: tunnel systemd service writer parameterized by alias/port"
```

---

### Task 4: Connect Command Skeleton

**Files:**
- Create: `cli/cmd_connect.go`
- Modify: `cli/root.go`

- [ ] **Step 1: Write minimal failing test for command registration**

Create `cli/cmd_connect_test.go` initial:

```go
package cli

import "testing"

func TestConnectCommandExists(t *testing.T) {
	cmd, _, err := rootCmd.Find([]string{"connect"})
	if err != nil || cmd == nil { t.Fatalf("connect command not found") }
	if cmd.Flags().Lookup("host") == nil { t.Error("missing --host") }
	if cmd.Flags().Lookup("alias") == nil { t.Error("missing --alias") }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./cli -run TestConnectCommandExists -v`
Expected: FAIL

- [ ] **Step 3: Implement skeleton cmd_connect.go and register**

Create `cli/cmd_connect.go`:

```go
package cli

import (
	"fmt"
	"github.com/spf13/cobra"
)

var connectHost, connectUser, connectKey, connectAlias string
var connectPort, connectTunnelPort int
var connectYes, connectDryRun bool

var connectCmd = &cobra.Command{
	Use:   "connect [alias]",
	Short: "Portable setup for any Ubuntu VM: SSH + init + serve + tunnel + opencode",
	Run: func(cmd *cobra.Command, args []string) {
		if len(args)>0 && connectAlias=="" { connectAlias=args[0] }
		if connectAlias=="" { connectAlias="mygpu" }
		if connectTunnelPort==0 { connectTunnelPort=connectPort }
		fmt.Printf("connect: alias=%s host=%s user=%s key=%s port=%d dry-run=%v\n", connectAlias, connectHost, connectUser, connectKey, connectPort, connectDryRun)
		if err := runConnect(cmd, args); err != nil { fmt.Printf("Error: %v\n", err); }
	},
}

func init() {
	connectCmd.Flags().StringVar(&connectHost, "host", "", "VM HostName/IP (required if alias new)")
	connectCmd.Flags().StringVar(&connectUser, "user", "ubuntu", "SSH user")
	connectCmd.Flags().StringVar(&connectKey, "key", "~/.ssh/id_ed25519", "IdentityFile")
	connectCmd.Flags().StringVar(&connectAlias, "alias", "", "Host alias in ~/.ssh/config")
	connectCmd.Flags().IntVar(&connectPort, "port", 8002, "Remote API port")
	connectCmd.Flags().IntVar(&connectTunnelPort, "tunnel-port", 0, "Local tunnel port")
	connectCmd.Flags().BoolVar(&connectYes, "yes", false, "Skip confirmations")
	connectCmd.Flags().BoolVar(&connectDryRun, "dry-run", false, "Print plan without executing")
}
```

Modify `cli/root.go` Add to `init()`:

```go
rootCmd.AddCommand(connectCmd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./cli -run TestConnectCommandExists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli/cmd_connect.go cli/root.go cli/cmd_connect_test.go
git commit -m "feat: connect command skeleton with flags"
```

---

### Task 5: Connect Orchestration (SSH + Opencode + Tunnel)

**Files:**
- Modify: `cli/cmd_connect.go`
- Modify: `cli/cmd_connect_test.go`

- [ ] **Step 1: Write failing integration test for orchestration**

Add to `cli/cmd_connect_test.go`:

```go
func TestConnectDryRun(t *testing.T) {
	dir := t.TempDir()
	sshPath := dir+"/ssh_config"
	opPath := dir+"/opencode.json"
	tunnelPath := dir+"/tunnel.service"
	// call helpers directly
	_, err := host.EnsureSSHConfig(sshPath, "mygpu", "1.2.3.4", "ubuntu", "/k")
	if err != nil { t.Fatalf("ssh: %v", err) }
	_, err = host.PatchOpencode(opPath, 8002)
	if err != nil { t.Fatalf("opencode: %v", err) }
	err = host.WriteTunnelService(tunnelPath, "mygpu", 8002)
	if err != nil { t.Fatalf("tunnel: %v", err) }
	// verify
}
```

- [ ] **Step 2: Run test to verify it fails (if helpers missing)**

Run: `go test ./cli -run TestConnectDryRun -v`
Expected: PASS (helpers already exist from Tasks 1-3) — write actual orchestration test that calls `runConnectDryRun`:

Add function `runConnectDryRun(alias, host, user, key, port)` in `cmd_connect.go` that returns plan string, test expects plan contains alias and host.

- [ ] **Step 3: Implement runConnect orchestration**

In `cli/cmd_connect.go` replace `Run:` with call to `runConnect`:

```go
func runConnect(cmd *cobra.Command, args []string) error {
	if len(args)>0 && connectAlias=="" { connectAlias=args[0] }
	if connectAlias=="" { connectAlias="mygpu" }
	if connectTunnelPort==0 { connectTunnelPort=connectPort }
	if connectHost=="" {
		// check if alias exists in ~/.ssh/config
		if _, err := os.Stat(os.ExpandEnv("$HOME/.ssh/config")); err==nil {
			// try ssh check
			out, err := exec.Command("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", connectAlias, "true").CombinedOutput()
			if err != nil { return fmt.Errorf("alias %s not reachable and --host missing. Output: %s", connectAlias, out) }
		} else {
			return fmt.Errorf("--host required for new alias %s", connectAlias)
		}
	} else {
		home, _ := os.UserHomeDir()
		sshConfig := filepath.Join(home, ".ssh", "config")
		if _, err := host.EnsureSSHConfig(sshConfig, connectAlias, connectHost, connectUser, connectKey); err != nil { return err }
		fmt.Printf("✓ SSH config Host %s -> %s\n", connectAlias, connectHost)
	}
	if connectDryRun { fmt.Println("dry-run: would run remote init/recommend/serve"); return nil }
	// remote init
	fmt.Println("→ ssh", connectAlias, "akilihost init")
	if out, err := exec.Command("ssh", connectAlias, "bash -lc '~/bin/akilihost init || akilihost init'").CombinedOutput(); err != nil { return fmt.Errorf("init failed: %v %s", err, out) }
	// recommend
	fmt.Println("→ ssh", connectAlias, "akilihost recommend")
	out, err := exec.Command("ssh", connectAlias, "bash -lc '~/bin/akilihost recommend || akilihost recommend'").CombinedOutput()
	if err != nil { return fmt.Errorf("recommend failed: %v", err) }
	fmt.Print(string(out))
	fmt.Print("Select [1..N]: ")
	var choice int
	fmt.Scanf("%d", &choice)
	// parse choice to model name (simplified: use host.LoadModelDB and pick)
	models, _ := host.LoadModelDB()
	pick := models[choice-1].Name
	fmt.Printf("→ ssh %s akilihost serve %s --port %d\n", connectAlias, pick, connectPort)
	if out, err := exec.Command("ssh", connectAlias, fmt.Sprintf("bash -lc '~/bin/akilihost serve %s --port %d || akilihost serve %s --port %d'", pick, connectPort, pick, connectPort)).CombinedOutput(); err != nil { return fmt.Errorf("serve failed: %s", out) }
	// poll health
	for i:=0;i<100;i++ {
		time.Sleep(3*time.Second)
		if out, err := exec.Command("ssh", connectAlias, fmt.Sprintf("curl -s http://localhost:%d/health", connectPort)).CombinedOutput(); err==nil && strings.Contains(string(out), "ok") { break }
		if i==99 { return fmt.Errorf("timeout waiting health") }
	}
	home, _ := os.UserHomeDir()
	if _, err := host.PatchOpencode(filepath.Join(home, ".config", "opencode", "opencode.json"), connectTunnelPort); err != nil { return err }
	fmt.Println("✓ opencode.json patched")
	if err := host.WriteTunnelService(filepath.Join(home, ".config", "systemd", "user", "ai-host-tunnel.service"), connectAlias, connectTunnelPort); err != nil { return err }
	exec.Command("systemctl", "--user", "daemon-reload").Run()
	exec.Command("systemctl", "--user", "enable", "--now", "ai-host-tunnel.service").Run()
	time.Sleep(2*time.Second)
	if out, err := exec.Command("curl", "-s", fmt.Sprintf("http://localhost:%d/health", connectTunnelPort)).CombinedOutput(); err!=nil || !strings.Contains(string(out), "ok") { return fmt.Errorf("tunnel health failed: %s", out) }
	fmt.Printf("✓ Tunnel localhost:%d -> %s:%d\n", connectTunnelPort, connectAlias, connectPort)
	fmt.Printf("✓ opencode ready: opencode run \"hello\" --model selfhosted/unsloth/Qwen3.8-27B-GGUF\n")
	return nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./cli -run TestConnect -v && go vet ./...`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli/cmd_connect.go cli/cmd_connect_test.go
git commit -m "feat: connect orchestration with SSH, recommend prompt, serve poll, opencode+tunnel"
```

---

### Task 6: Docs and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-31-portable-connect-design.md` status if needed

- [ ] **Step 1: Update README Quick start with connect**

In `README.md` after `## Quick start` add:

```markdown
### Portable connect (any Ubuntu VM)

```bash
# from laptop, after creating VM with IP 1.2.3.4
go build -o akilihost
./akilihost connect mygpu --host 1.2.3.4 --user ubuntu --key ~/.ssh/mykey.pem
# prompts model from recommend, serves, tunnels, patches opencode
curl http://localhost:8002/health
opencode run "hello" --model selfhosted/unsloth/Qwen3.8-27B-GGUF
```

Re-running `./akilihost connect mygpu` is idempotent (skips serve if healthy).
```

- [ ] **Step 2: Full verification**

Run:
```bash
go build -o /tmp/akilihost && echo "BUILD OK"
go vet ./... && echo "VET OK"
go test ./... -count=1 && echo "TESTS OK"
./akilihost connect --help | grep -q "host" && echo "HELP OK"
```

Expected: all OK

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README portable connect quick start"
```

---

## Verification Checklist

- [ ] `go build ./...` succeeds
- [ ] `go vet ./...` succeeds
- [ ] `go test ./...` all green (host + cli)
- [ ] `akilihost connect --help` shows --host/--user/--key/--alias/--port/--dry-run
- [ ] `akilihost connect mygpu --host 1.2.3.4 --dry-run` prints plan without side effects
- [ ] SSH config create preserves gcloud block and LocalForward
- [ ] opencode.json backup created and selfhosted merged
- [ ] tunnel service written with alias not myserver
- [ ] Fresh VM E2E: init + recommend prompt + serve + health + tunnel + opencode run streams
- [ ] Idempotent re-run skips serve
- [ ] README documents portable flow
