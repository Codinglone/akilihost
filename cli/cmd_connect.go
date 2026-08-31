package cli

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var connectHost, connectUser, connectKey, connectAlias string
var connectPort, connectTunnelPort int
var connectYes, connectDryRun bool

var connectCmd = &cobra.Command{
	Use:   "connect [alias]",
	Short: "Portable setup for any Ubuntu VM: SSH + init + serve + tunnel + opencode",
	Long:  `Portable setup for any Ubuntu VM: ensures ~/.ssh/config Host, runs remote akilihost init, prompts model from akilihost recommend, serves, patches opencode.json and starts tunnel.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runConnect(cmd, args)
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

func buildConnectPlan(alias, host, user, key string, port, tunnelPort int) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("connect plan: alias=%s host=%s user=%s key=%s port=%d tunnelPort=%d\n", alias, host, user, key, port, tunnelPort))
	if host != "" {
		sb.WriteString(fmt.Sprintf("  - ensure SSH config Host %s -> %s (user=%s key=%s)\n", alias, host, user, key))
	} else {
		sb.WriteString(fmt.Sprintf("  - check SSH alias %s reachable (ssh %s true with BatchMode)\n", alias, alias))
	}
	sb.WriteString(fmt.Sprintf("  - remote: ssh %s \"akilihost init\"\n", alias))
	sb.WriteString(fmt.Sprintf("  - remote: ssh %s \"akilihost recommend\" (prompt Select [1..N])\n", alias))
	sb.WriteString(fmt.Sprintf("  - remote: ssh %s \"akilihost serve <model> --port %d\"\n", alias, port))
	sb.WriteString(fmt.Sprintf("  - poll: ssh %s curl localhost:%d/health every 3s x100\n", alias, port))
	sb.WriteString(fmt.Sprintf("  - patch opencode: ~/.config/opencode/opencode.json baseURL http://localhost:%d/v1\n", tunnelPort))
	sb.WriteString(fmt.Sprintf("  - write tunnel: ~/.config/systemd/user/ai-host-tunnel.service alias %s port %d\n", alias, tunnelPort))
	sb.WriteString("  - systemctl --user daemon-reload && systemctl --user enable --now ai-host-tunnel.service\n")
	sb.WriteString(fmt.Sprintf("  - verify: curl http://localhost:%d/health\n", tunnelPort))
	return sb.String()
}

func runConnectDryRun(alias, host, user, key string, port int) string {
	tp := port
	if connectTunnelPort != 0 {
		tp = connectTunnelPort
	}
	return buildConnectPlan(alias, host, user, key, port, tp)
}

func validateConnectArgs() error {
	alias := connectAlias
	if alias == "" {
		alias = "mygpu"
	}
	if strings.ContainsAny(alias, " \t\n\r") || alias == "" {
		return fmt.Errorf("invalid alias %q: must be non-empty without spaces", alias)
	}
	if connectHost != "" && strings.ContainsAny(connectHost, " \t\n\r") {
		return fmt.Errorf("invalid --host %q: must not contain spaces", connectHost)
	}
	if connectUser != "" && strings.ContainsAny(connectUser, " \t\n\r") {
		return fmt.Errorf("invalid --user %q: must not contain spaces", connectUser)
	}
	if connectKey != "" && strings.ContainsAny(connectKey, " \t\n\r") {
		return fmt.Errorf("invalid --key %q: must not contain spaces", connectKey)
	}
	if connectPort < 1 || connectPort > 65535 {
		return fmt.Errorf("invalid --port %d: must be 1-65535", connectPort)
	}
	tp := connectTunnelPort
	if tp == 0 {
		tp = connectPort
	}
	if tp < 1 || tp > 65535 {
		return fmt.Errorf("invalid --tunnel-port %d: must be 1-65535", tp)
	}
	return nil
}

func runConnect(cmd *cobra.Command, args []string) error {
	if len(args) > 0 && connectAlias == "" {
		connectAlias = args[0]
	}
	if connectAlias == "" {
		connectAlias = "mygpu"
	}
	if connectTunnelPort == 0 {
		connectTunnelPort = connectPort
	}

	if err := validateConnectArgs(); err != nil {
		return err
	}

	if connectDryRun {
		plan := buildConnectPlan(connectAlias, connectHost, connectUser, connectKey, connectPort, connectTunnelPort)
		fmt.Fprintln(cmd.OutOrStdout(), plan)
		fmt.Fprintln(cmd.OutOrStdout(), "dry-run: would run remote init/recommend/serve")
		return nil
	}

	// handle --yes flag (skip confirmations) - referenced to avoid unused
	_ = connectYes

	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("get home dir: %w", err)
	}

	if connectHost == "" {
		sshConfigPath := filepath.Join(home, ".ssh", "config")
		if _, err := os.Stat(sshConfigPath); err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("--host required for new alias %s (no ~/.ssh/config found). Example: akilihost connect %s --host 1.2.3.4 --user ubuntu --key ~/.ssh/id_ed25519: %w", connectAlias, connectAlias, err)
			}
			return fmt.Errorf("stat ssh config: %w", err)
		}
		out, err := exec.Command("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", connectAlias, "true").CombinedOutput()
		if err != nil {
			return fmt.Errorf("alias %s not reachable and --host missing (ssh check failed: %s): %w\nHint: verify ~/.ssh/config Host %s and run ssh -v %s or provide --host 1.2.3.4", connectAlias, strings.TrimSpace(string(out)), err, connectAlias, connectAlias)
		}
		fmt.Fprintf(cmd.OutOrStdout(), "✓ SSH alias %s reachable\n", connectAlias)
	} else {
		sshConfig := filepath.Join(home, ".ssh", "config")
		if err := os.MkdirAll(filepath.Dir(sshConfig), 0700); err != nil {
			return fmt.Errorf("create ~/.ssh dir: %w", err)
		}
		created, err := host.EnsureSSHConfig(sshConfig, connectAlias, connectHost, connectUser, connectKey)
		if err != nil {
			return fmt.Errorf("ensure SSH config Host %s -> %s: %w", connectAlias, connectHost, err)
		}
		if created {
			fmt.Fprintf(cmd.OutOrStdout(), "✓ SSH config Host %s -> %s\n", connectAlias, connectHost)
		} else {
			fmt.Fprintf(cmd.OutOrStdout(), "✓ SSH config Host %s updated -> %s\n", connectAlias, connectHost)
		}
	}

	fmt.Fprintf(cmd.OutOrStdout(), "→ ssh %s akilihost init\n", connectAlias)
	if out, err := exec.Command("ssh", connectAlias, "bash -lc '~/bin/akilihost init || akilihost init'").CombinedOutput(); err != nil {
		return fmt.Errorf("remote init failed: %w output: %s", err, string(out))
	}
	fmt.Fprintln(cmd.OutOrStdout(), "✓ remote init done")

	fmt.Fprintf(cmd.OutOrStdout(), "→ ssh %s akilihost recommend\n", connectAlias)
	out, err := exec.Command("ssh", connectAlias, "bash -lc '~/bin/akilihost recommend || akilihost recommend'").CombinedOutput()
	if err != nil {
		return fmt.Errorf("remote recommend failed: %w output: %s", err, string(out))
	}
	fmt.Fprint(cmd.OutOrStdout(), string(out))
	fmt.Fprint(cmd.OutOrStdout(), "Select [1..N]: ")
	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	choice, err := strconv.Atoi(line)
	if err != nil {
		return fmt.Errorf("invalid selection %q: %w", line, err)
	}
	models, _ := host.LoadModelDB()
	if len(models) == 0 {
		return fmt.Errorf("no models available")
	}
	if choice < 1 || choice > len(models) {
		return fmt.Errorf("selection %d out of range [1..%d]", choice, len(models))
	}
	pick := models[choice-1].RepoID
	if pick == "" {
		pick = models[choice-1].Name
	}
	fmt.Fprintf(cmd.OutOrStdout(), "→ selected model %s\n", pick)

	fmt.Fprintf(cmd.OutOrStdout(), "→ ssh %s akilihost serve %s --port %d\n", connectAlias, pick, connectPort)
	serveCmd := fmt.Sprintf("bash -lc '~/bin/akilihost serve %s --port %d || akilihost serve %s --port %d'", pick, connectPort, pick, connectPort)
	if out, err := exec.Command("ssh", connectAlias, serveCmd).CombinedOutput(); err != nil {
		return fmt.Errorf("remote serve failed: %w output: %s", err, string(out))
	}

	fmt.Fprintf(cmd.OutOrStdout(), "→ polling ssh %s health localhost:%d\n", connectAlias, connectPort)
	healthOK := false
	for i := 0; i < 100; i++ {
		time.Sleep(3 * time.Second)
		healthOut, err := exec.Command("ssh", connectAlias, fmt.Sprintf("curl -s http://localhost:%d/health", connectPort)).CombinedOutput()
		if err == nil && strings.Contains(string(healthOut), "ok") {
			healthOK = true
			break
		}
		if err == nil && strings.Contains(string(healthOut), "status") {
			healthOK = true
			break
		}
		if i == 99 {
			return fmt.Errorf("timeout waiting for remote health on %s:%d after 300s", connectAlias, connectPort)
		}
	}
	if !healthOK {
		return fmt.Errorf("remote health check failed")
	}
	fmt.Fprintln(cmd.OutOrStdout(), "✓ remote health ok")

	opencodePath := filepath.Join(home, ".config", "opencode", "opencode.json")
	if err := os.MkdirAll(filepath.Dir(opencodePath), 0755); err != nil {
		return fmt.Errorf("create opencode config dir: %w", err)
	}
	backup, err := host.PatchOpencode(opencodePath, connectTunnelPort)
	if err != nil {
		return fmt.Errorf("patch opencode.json: %w", err)
	}
	if backup != "" {
		fmt.Fprintf(cmd.OutOrStdout(), "✓ opencode.json patched (backup %s) baseURL http://localhost:%d/v1\n", backup, connectTunnelPort)
	} else {
		fmt.Fprintf(cmd.OutOrStdout(), "✓ opencode.json created baseURL http://localhost:%d/v1\n", connectTunnelPort)
	}

	tunnelPath := filepath.Join(home, ".config", "systemd", "user", "ai-host-tunnel.service")
	if err := os.MkdirAll(filepath.Dir(tunnelPath), 0755); err != nil {
		return fmt.Errorf("create systemd user dir: %w", err)
	}
	if err := host.WriteTunnelService(tunnelPath, connectAlias, connectTunnelPort); err != nil {
		return fmt.Errorf("write tunnel service: %w", err)
	}
	fmt.Fprintf(cmd.OutOrStdout(), "✓ tunnel service written %s alias %s port %d\n", tunnelPath, connectAlias, connectTunnelPort)

	fmt.Fprintln(cmd.OutOrStdout(), "→ systemctl --user daemon-reload")
	if out, err := exec.Command("systemctl", "--user", "daemon-reload").CombinedOutput(); err != nil {
		fmt.Fprintf(cmd.ErrOrStderr(), "warning: daemon-reload failed: %s: %v\n", string(out), err)
	}
	fmt.Fprintln(cmd.OutOrStdout(), "→ systemctl --user enable --now ai-host-tunnel.service")
	if out, err := exec.Command("systemctl", "--user", "enable", "--now", "ai-host-tunnel.service").CombinedOutput(); err != nil {
		fmt.Fprintf(cmd.ErrOrStderr(), "warning: enable tunnel failed: %s: %v\n", string(out), err)
	}
	time.Sleep(2 * time.Second)

	fmt.Fprintf(cmd.OutOrStdout(), "→ curl http://localhost:%d/health\n", connectTunnelPort)
	tunnelOut, err := exec.Command("curl", "-s", "--max-time", "5", fmt.Sprintf("http://localhost:%d/health", connectTunnelPort)).CombinedOutput()
	if err != nil || !strings.Contains(string(tunnelOut), "ok") {
		tunnelOut2, err2 := exec.Command("curl", "-s", "--max-time", "5", fmt.Sprintf("http://localhost:%d/v1/models", connectTunnelPort)).CombinedOutput()
		if err2 != nil || (!strings.Contains(string(tunnelOut2), "data") && !strings.Contains(string(tunnelOut), "ok")) {
			return fmt.Errorf("tunnel health failed on localhost:%d: %s: %w", connectTunnelPort, string(tunnelOut), err)
		}
	}
	fmt.Fprintf(cmd.OutOrStdout(), "✓ Tunnel localhost:%d -> %s:%d\n", connectTunnelPort, connectAlias, connectPort)

	hostDisplay := connectHost
	if hostDisplay == "" {
		hostDisplay = connectAlias
	}
	fmt.Fprintf(cmd.OutOrStdout(), "✓ VM %s (%s) healthy: %s on %d\n", connectAlias, hostDisplay, pick, connectPort)
	fmt.Fprintf(cmd.OutOrStdout(), "✓ opencode ready: opencode run \"hello\" --model selfhosted/unsloth/Qwen3.8-27B-GGUF\n")
	return nil
}
