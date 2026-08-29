package cli

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/Codinglone/akilihost/host"
	"github.com/spf13/cobra"
)

var stopCmd = &cobra.Command{
	Use:   "stop <model|port>",
	Short: "Stop a model server (by model name, port, or PID)",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		target := args[0]

		// Try to stop as systemd service first
		serviceName := getSystemdServiceName(target)
		if serviceName != "" {
			stopSystemdService(serviceName)
			return
		}

		// Try to stop as port
		if port, err := strconv.Atoi(target); err == nil {
			stopByPort(port)
			return
		}

		// Try to stop as PID
		if _, err := strconv.Atoi(target); err == nil {
			stopByPID(target)
			return
		}

		// Try to stop by model name (matches part of command line)
		stopByModelName(target)
	},
}

func getSystemdServiceName(target string) string {
	models, err := host.LoadModelDB()
	if err != nil {
		return ""
	}
	targetLower := strings.ToLower(target)
	for _, m := range models {
		nameLower := strings.ToLower(m.Name)
		if strings.Contains(nameLower, targetLower) || strings.Contains(targetLower, nameLower) {
			return host.ServiceName(&m)
		}
	}
	return ""
}

func stopSystemdService(service string) {
	fmt.Printf("Stopping systemd service: %s\n", service)

	// Check if service exists
	cmd := exec.Command("systemctl", "is-active", service, "--quiet")
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Service not found or not active\n")
		return
	}

	// Stop the service
	cmd = exec.Command("sudo", "systemctl", "stop", service)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to stop: %s\n", string(output))
		return
	}

	fmt.Printf("  Service stopped successfully\n")
}

func stopByPort(port int) {
	fmt.Printf("Stopping model on port %d\n", port)

	// Find PIDs with this port in command
	cmd := exec.Command("lsof", "-t", "-i", fmt.Sprintf(":%d", port))
	output, err := cmd.Output()
	if err != nil {
		fmt.Println("  No process found on this port")
		return
	}

	pids := strings.Fields(string(output))
	if len(pids) == 0 {
		fmt.Println("  No process found on this port")
		return
	}

	for _, pidStr := range pids {
		stopByPID(pidStr)
	}
}

func stopByPID(pid string) {
	fmt.Printf("  Killing PID %s\n", pid)
	cmd := exec.Command("sudo", "kill", "-9", pid)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("    Failed: %s\n", string(output))
	} else {
		fmt.Printf("    Killed successfully\n")
	}
}

func stopByModelName(name string) {
	fmt.Printf("Stopping model matching: %s\n", name)

	for _, procPattern := range []string{"vllm serve", "llama-server"} {
		cmd := exec.Command("pgrep", "-af", procPattern)
		output, err := cmd.Output()
		if err != nil {
			continue
		}

		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		for _, line := range lines {
			if strings.Contains(line, name) {
				parts := strings.Fields(line)
				if len(parts) > 0 {
					stopByPID(parts[0])
				}
			}
		}
	}
}
