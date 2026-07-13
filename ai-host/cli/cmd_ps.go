package cli

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

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
	
	// Check for vllm-qwen service
	cmd := exec.Command("systemctl", "status", "vllm-qwen", "--no-pager", "--quiet")
	if err := cmd.Run(); err == nil {
		// Service exists, get more details
		out, _ := exec.Command("systemctl", " status", "vllm-qwen", "--no-pager", "--lines=5").Output()
		lines := strings.Split(string(out), "\n")
		for _, l := range lines {
			if strings.Contains(l, "Active:") || strings.Contains(l, "Main PID") {
				fmt.Println("  ", strings.TrimSpace(l))
			}
		}
		fmt.Println("  Model: Qwen/Qwen3-Coder-Next")
		fmt.Println("  Port: 8002")
		fmt.Println("  Status: active")
		fmt.Println()
	}

	// Check for vllm-devstral service (if exists)
	cmd = exec.Command("systemctl", "status", "vllm-devstral", "--no-pager", "--quiet")
	if err := cmd.Run(); err == nil {
		out, _ := exec.Command("systemctl", "status", "vllm-devstral", "--no-pager", "--lines=5").Output()
		lines := strings.Split(string(out), "\n")
		for _, l := range lines {
			if strings.Contains(l, "Active:") || strings.Contains(l, "Main PID") {
				fmt.Println("  ", strings.TrimSpace(l))
			}
		}
		fmt.Println("  Model: mistralai/Devstral-2-123B-Instruct-2512")
		fmt.Println("  Port: 8003")
		fmt.Println("  Status: active")
		fmt.Println()
	}
}

func checkRunningProcesses() {
	fmt.Println("Running Processes:")
	fmt.Println("------------------")

	cmd := exec.Command("pgrep", "-af", "vllm serve")
	output, err := cmd.Output()
	if err != nil {
		fmt.Println("  No vLLM processes found")
		return
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 || (len(lines) == 1 && strings.TrimSpace(lines[0]) == "") {
		fmt.Println("  No vLLM processes found")
		return
	}

	for _, line := range lines {
		if strings.Contains(line, "vllm serve") && !strings.Contains(line, "grep") {
			// Extract PID and port
			parts := strings.Fields(line)
			if len(parts) > 1 {
				pid := parts[0]
				fmt.Printf("  PID %s: vLLM server\n", pid)

				// Try to get port from command line
				if len(parts) > 2 {
					for i, p := range parts {
						if p == "--port" && i+1 < len(parts) {
							fmt.Printf("    Port: %s\n", parts[i+1])
							// Check health
							go checkHealth(parts[i+1])
							break
						}
					}
				}
			}
		}
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
