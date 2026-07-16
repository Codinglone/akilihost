package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var serveCmd = &cobra.Command{
	Use:   "serve <model|auto>",
	Short: "Start serving a model (auto-picks best quantization)",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		target := args[0]

		// Detect GPU
		gpu, err := host.DetectGPU()
		if err != nil {
			fmt.Printf("Error detecting GPU: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("GPU: %s\n", gpu.Name)
		fmt.Printf("VRAM: %d MB (%.1f GB)\n", gpu.TotalVRAMMB, float64(gpu.TotalVRAMMB)/1024)
		fmt.Printf("CUDA: %s\n\n", gpu.CUDAVersion)

		// Load models
		models, err := host.LoadModelDB()
		if err != nil {
			fmt.Printf("Error loading model DB: %v\n", err)
			os.Exit(1)
		}

		var modelToServe *host.Model
		var quantization *host.Quantization

		if target == "auto" || target == " recommand" || target == "recommend" {
			// Auto-select best fitting model
			sizer := host.NewModelSizer(gpu)
			results := sizer.FindFit(models, gpu.TotalVRAMMB*85/100) // Use 85% VRAM

			if len(results) == 0 {
				fmt.Println("No models fit on this GPU!")
				os.Exit(1)
			}

			// Sort by best score (highest context, then smallest VRAM)
			fmt.Printf("Recommended model(s) for your GPU:\n")
			for i, r := range results {
				fmt.Printf("%d. %s %s (%.1f GB) - %s\n",
					i+1, r.Model.Name, r.Quantization.Name,
					float64(r.TotalMB)/1024, r.Model.Description)
			}

			// Auto-select first if only one option
			if len(results) == 1 {
				modelToServe = results[0].Model
				quantization = results[0].Quantization
				fmt.Printf("\nAuto-selecting: %s %s\n", modelToServe.Name, quantization.Name)
			} else {
				// For now, just pick the first one (user can specify model manually)
				modelToServe = results[0].Model
				quantization = results[0].Quantization
				fmt.Printf("\nAuto-selecting: %s %s\n", modelToServe.Name, quantization.Name)
			}
		} else {
			// Find model by name or repo ID
			var found bool
			for _, m := range models {
				if strings.EqualFold(m.RepoID, target) || strings.EqualFold(m.Name, target) {
					modelToServe = &m
					found = true
					break
				}
			}

			if !found {
				fmt.Printf("Model '%s' not found in database\n", target)
				os.Exit(1)
			}

			// Select quantization (优先选第一个)
			if len(modelToServe.Quantizations) > 0 {
				quantization = &modelToServe.Quantizations[0]
			}
		}

		if modelToServe == nil || quantization == nil {
			fmt.Println("No model selected")
			os.Exit(1)
		}

		// Show selection and confirm
		fmt.Printf("\nSelected:\n")
		fmt.Printf("  Model: %s\n", modelToServe.Name)
		fmt.Printf("  Repo: %s\n", modelToServe.RepoID)
		fmt.Printf("  Quantization: %s\n", quantization.Name)
		fmt.Printf("  Description: %s\n", quantization.Description)
		fmt.Printf("  Flags: %s\n", strings.Join(quantization.Flags, " "))

		// Determine port based on model name
		port := determinePort(modelToServe.RepoID)

		// Create systemd service
		fmt.Printf("\nCreating systemd service for %s...\n", modelToServe.Name)
		createSystemdService(modelToServe, quantization, port)

		// Start the service
		fmt.Printf("\nStarting service...\n")
		startSystemdService(modelToServe.Name)

		// Wait and verify
		fmt.Printf("\nVerifying...\n")
		waitAndVerify(port)
	},
}

func determinePort(repoID string) int {
	switch {
	case strings.Contains(repoID, "Qwen3-Coder-Next"):
		return 8002
	case strings.Contains(repoID, "Devstral-2"):
		return 8003
	case strings.Contains(repoID, "Qwen2.5-Coder"):
		return 8004
	}
	return 8002
}

func createSystemdService(model *host.Model, quantization *host.Quantization, port int) {
	// Build vLLM command
	args := []string{
		"vllm", "serve", model.RepoID,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(port),
	}
	args = append(args, quantization.Flags...)
	args = append(args, "--enforce-eager")

	if len(quantization.Flags) > 0 && quantization.DType != "" {
		args = append(args, "--generation-config", "vllm")
	}

	if strings.Contains(model.RepoID, "Qwen3-Coder-Next") {
		args = append(args, "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder")
	}

	// Check for GDN kernel issue and add fallback
	args = append(args, "--gdn-prefill-backend", "triton")

	serviceContent := fmt.Sprintf(`[Unit]
Description=vLLM %s Server
Documentation=https://github.com/Codinglone/akilihost
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStart=%s
ExecStop=/bin/kill -TERM $MAINPID

[Install]
WantedBy=multi-user.target
`, model.Name, strings.Join(args, " "))

	// Write service file
	servicePath := "/etc/systemd/system/vllm-" + strings.ReplaceAll(model.Name, " ", "-") + ".service"
	fmt.Printf("  Writing service file: %s\n", servicePath)

	cmd := exec.Command("sudo", "bash", "-c", fmt.Sprintf("cat > %s", servicePath))
	cmd.Stdin = strings.NewReader(serviceContent)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to write service file: %s\n", string(output))
		return
	}

	// Reload systemd
	fmt.Printf("  Reloading systemd daemon...\n")
	cmd = exec.Command("sudo", "systemctl", "daemon-reload")
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed: %s\n", string(output))
	} else {
		fmt.Println("  Done")
	}

	// Enable service
	fmt.Printf("  Enabling service...\n")
	cmd = exec.Command("sudo", "systemctl", "enable", "vllm-"+strings.ReplaceAll(model.Name, " ", "-")+".service")
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed: %s\n", string(output))
	} else {
		fmt.Println("  Done")
	}
}

func startSystemdService(modelName string) {
	serviceName := "vllm-" + strings.ReplaceAll(modelName, " ", "-")

	cmd := exec.Command("sudo", "systemctl", "start", serviceName)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to start: %s\n", string(output))
	} else {
		fmt.Printf("  Service started: %s\n", serviceName)
	}
}

func waitAndVerify(port int) {
	fmt.Printf("  Waiting for server to start...\n")
	for i := 0; i < 10; i++ {
		cmd := exec.Command("curl", "-s", "--max-time", "5", fmt.Sprintf("http://localhost:%d/v1/models", port))
		output, err := cmd.Output()
		if err == nil {
			var data struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if json.Unmarshal(output, &data) == nil && len(data.Data) > 0 {
				fmt.Printf("  Server ready: %s\n", data.Data[0].ID)
				fmt.Printf("  Port: %d\n", port)
				return
			}
		}
		fmt.Printf("  Waiting... (%ds)\n", (i+1)*5)
	}
	fmt.Println("  Timeout - check logs for errors")
}
