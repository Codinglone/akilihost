package cli

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/Codinglone/akilihost/host"
	"github.com/spf13/cobra"
)

var psCmd = &cobra.Command{
	Use:   "ps",
	Short: "List running models (port, VRAM, health)",
	Run: func(cmd *cobra.Command, args []string) {
		// Check for systemd services
		checkSystemd := true
		if checkSystemd {
			checkSystemdServices()
		}

		// Also check for running vLLM processes
		checkRunningProcesses()
	},
}

func checkSystemdServices() {
	fmt.Println("Systemd Services:")
	fmt.Println("-----------------")

	models, err := host.LoadModelDB()
	if err != nil {
		fmt.Printf("  Error loading model DB: %v\n", err)
		return
	}

	anyActive := false
	for _, m := range models {
		serviceName := host.ServiceName(&m)
		cmd := exec.Command("systemctl", "is-active", serviceName, "--quiet")
		if err := cmd.Run(); err != nil {
			continue
		}

		anyActive = true
		out, _ := exec.Command("systemctl", "status", serviceName, "--no-pager", "--lines=5").Output()
		lines := strings.Split(string(out), "\n")
		for _, l := range lines {
			if strings.Contains(l, "Active:") || strings.Contains(l, "Main PID") {
				fmt.Println("  ", strings.TrimSpace(l))
			}
		}
		fmt.Printf("  Model: %s\n", m.RepoID)
		fmt.Printf("  Backend: %s\n", m.Backend)

		showCmd, _ := exec.Command("systemctl", "show", serviceName, "--property=ExecStart", "--no-pager").Output()
		cmdStr := strings.TrimSpace(string(showCmd))
		if port := extractPort(cmdStr); port != "" {
			fmt.Printf("  Port: %s\n", port)
			go checkHealth(port)
		}

		showVRAM()
		fmt.Println()
	}

	if !anyActive {
		fmt.Println("  No active model services")
	}
}

func checkRunningProcesses() {
	fmt.Println("Running Processes:")
	fmt.Println("------------------")

	found := false
	for _, pattern := range []string{"vllm serve", "llama-server"} {
		cmd := exec.Command("pgrep", "-af", pattern)
		output, err := cmd.Output()
		if err != nil {
			continue
		}

		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		for _, line := range lines {
			if strings.Contains(line, pattern) && !strings.Contains(line, "grep") {
				found = true
				parts := strings.Fields(line)
				if len(parts) > 1 {
					pid := parts[0]
					fmt.Printf("  PID %s: %s\n", pid, pattern)
					for i, p := range parts {
						if p == "--port" && i+1 < len(parts) {
							fmt.Printf("    Port: %s\n", parts[i+1])
							go checkHealth(parts[i+1])
							break
						}
					}
				}
			}
		}
	}

	if !found {
		fmt.Println("  No model processes found")
	}
}

func checkHealth(port string) {
	resp, err := exec.Command("curl", "-s", "--max-time", "3", fmt.Sprintf("http://localhost:%s/v1/models", port)).Output()
	if err != nil {
		fmt.Printf("    Health: timeout/unreachable\n")
		return
	}

	var data map[string]interface{}
	if err := json.Unmarshal(resp, &data); err != nil {
		fmt.Printf("    Health: error parsing response\n")
		return
	}

	if models, ok := data["data"].([]interface{}); ok && len(models) > 0 {
		if model, ok := models[0].(map[string]interface{}); ok {
			if id, ok := model["id"].(string); ok {
				fmt.Printf("    Health: ok (model: %s)\n", id)
			}
		}
	}
}

func showVRAM() {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}
	line := strings.TrimSpace(string(out))
	parts := strings.Split(line, ",")
	if len(parts) == 2 {
		used := strings.TrimSpace(parts[0])
		total := strings.TrimSpace(parts[1])
		fmt.Printf("  VRAM: %s / %s MiB\n", used, total)
	}
}

func extractPort(cmdStr string) string {
	parts := strings.Fields(cmdStr)
	for i, p := range parts {
		if p == "--port" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}
