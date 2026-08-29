package cli

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var servePort int
var serveGpuMemUtil float64
var serveMaxModelLen int

func init() {
	serveCmd.Flags().IntVar(&servePort, "port", 8002, "API port (auto-increments if busy unless explicitly set)")
	serveCmd.Flags().Float64Var(&serveGpuMemUtil, "gpu-memory-utilization", 0.90, "Max GPU memory fraction (vLLM)")
	serveCmd.Flags().IntVar(&serveMaxModelLen, "max-model-len", 32768, "Max context tokens")
}

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
			results := sizer.FindFit(models, gpu.TotalVRAMMB*85/100, serveMaxModelLen) // Use 85% VRAM

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

			modelToServe = results[0].Model
			quantization = results[0].Quantization
			fmt.Printf("\nAuto-selecting: %s %s\n", modelToServe.Name, quantization.Name)
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

		// Select backend
		backend := host.SelectBackend(modelToServe, quantization, gpu)
		port, err := resolvePort(servePort, cmd.Flags().Changed("port"))
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("\nSelected:\n")
		fmt.Printf("  Model: %s\n", modelToServe.Name)
		fmt.Printf("  Repo: %s\n", modelToServe.RepoID)
		fmt.Printf("  Quantization: %s\n", quantization.Name)
		fmt.Printf("  Backend: %s\n", backend)
		fmt.Printf("  Port: %d\n", port)

		if backend == "llama-cpp" {
			fmt.Printf("\nPreparing llama.cpp backend...\n")
			serveLlamaCpp(modelToServe, quantization, port)
		} else {
			fmt.Printf("\nPreparing vLLM backend...\n")
			serveVllm(modelToServe, quantization, port)
		}

		fmt.Printf("\nVerifying...\n")
		waitAndVerify(port)
	},
}

func resolvePort(explicitPort int, explicit bool) (int, error) {
	if explicit {
		if isPortBusy(explicitPort) {
			return 0, fmt.Errorf("port %d is already in use", explicitPort)
		}
		return explicitPort, nil
	}
	port := explicitPort
	for isPortBusy(port) {
		port++
		if port > explicitPort+100 {
			return 0, fmt.Errorf("no free port in range %d-%d", explicitPort, explicitPort+100)
		}
	}
	return port, nil
}

func isPortBusy(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return true
	}
	listener.Close()
	return false
}

func createVllmService(model *host.Model, quantization *host.Quantization, port int) {
	// Build vLLM command
	args := []string{
		"vllm", "serve", model.RepoID,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(port),
	}
	args = append(args, quantization.Flags...)
	args = append(args, "--enforce-eager")
	args = append(args, "--gpu-memory-utilization", strconv.FormatFloat(serveGpuMemUtil, 'f', -1, 64))
	args = append(args, "--max-model-len", strconv.Itoa(serveMaxModelLen))

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
	serviceName := host.ServiceName(model)
	servicePath := "/etc/systemd/system/" + serviceName + ".service"
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
}

func startVllmService(modelName string) {
	serviceName := "vllm-" + strings.ReplaceAll(modelName, " ", "-")

	cmd := exec.Command("sudo", "systemctl", "start", serviceName)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to start: %s\n", string(output))
	} else {
		fmt.Printf("  Service started: %s\n", serviceName)
	}
}

const verifyTimeoutSeconds = 300

func waitAndVerify(port int) {
	fmt.Printf("  Waiting for server to start...\n")
	attempts := verifyTimeoutSeconds / 5
	for i := 0; i < attempts; i++ {
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
		fmt.Printf("  Waiting... (%ds elapsed)\n", (i+1)*5)
	}
	fmt.Println("  Timeout - check logs for errors")
}

func serveVllm(model *host.Model, quant *host.Quantization, port int) {
	fmt.Printf("  Flags: %s\n", strings.Join(quant.Flags, " "))
	fmt.Printf("\nCreating systemd service for %s...\n", model.Name)
	createVllmService(model, quant, port)
	fmt.Printf("\nStarting service...\n")
	startVllmService(model.Name)
}

func serveLlamaCpp(model *host.Model, quant *host.Quantization, port int) {
	modelDir := host.ModelDir(model)

	// Check if GGUF already downloaded by scanning for files matching the pattern
	var ggufPath string
	if entries, err := os.ReadDir(modelDir); err == nil {
		var files []string
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".gguf") {
				files = append(files, e.Name())
			}
		}
		matched := host.ResolveGGUFFromList(files, quant.FilePattern)
		if len(matched) > 0 {
			ggufPath = filepath.Join(modelDir, matched[0])
			fmt.Printf("  Model already downloaded: %s\n", ggufPath)
		}
	}

	// Download if not found
	if ggufPath == "" {
		fmt.Printf("  Downloading %s %s...\n", model.Name, quant.Name)
		fmt.Printf("  Pattern: %s\n", quant.FilePattern)

		cmd := exec.Command("hf", "download", model.RepoID,
			"--local-dir", modelDir,
			"--include", quant.FilePattern)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("  Download failed: %v\n", err)
			os.Exit(1)
		}

		// Find the downloaded GGUF file(s)
		entries, err := os.ReadDir(modelDir)
		if err != nil {
			fmt.Printf("  Cannot read model dir: %v\n", err)
			os.Exit(1)
		}
		var files []string
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".gguf") {
				files = append(files, e.Name())
			}
		}
		matched := host.ResolveGGUFFromList(files, quant.FilePattern)
		if len(matched) == 0 {
			fmt.Printf("  No GGUF file matching %s found in %s\n", quant.FilePattern, modelDir)
			os.Exit(1)
		}
		ggufPath = filepath.Join(modelDir, matched[0])
		fmt.Printf("  Using: %s\n", ggufPath)
	}

	// Build command and create systemd service
	args := host.BuildLlamaServerCommand(model, quant, ggufPath, port, serveMaxModelLen)
	serviceName := host.ServiceName(model)

	fmt.Printf("  Creating systemd service: %s\n", serviceName)
	createLlamaCppService(serviceName, args)

	fmt.Printf("\n  Starting service...\n")
	startService(serviceName)
}

func createLlamaCppService(serviceName string, args []string) {
	serviceContent := fmt.Sprintf(`[Unit]
Description=akilihost llama-server: %s
Documentation=https://github.com/Codinglone/akilihost
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=on-failure
RestartSec=10
ExecStart=%s
ExecStop=/bin/kill -TERM $MAINPID

[Install]
WantedBy=multi-user.target
`, serviceName, strings.Join(args, " "))

	servicePath := "/etc/systemd/system/" + serviceName + ".service"
	fmt.Printf("  Writing service file: %s\n", servicePath)

	cmd := exec.Command("sudo", "bash", "-c", fmt.Sprintf("cat > %s", servicePath))
	cmd.Stdin = strings.NewReader(serviceContent)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to write service file: %s\n", string(output))
		return
	}

	fmt.Printf("  Reloading systemd daemon...\n")
	cmd = exec.Command("sudo", "systemctl", "daemon-reload")
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed: %s\n", string(output))
	} else {
		fmt.Println("  Done")
	}
}

func startService(serviceName string) {
	cmd := exec.Command("sudo", "systemctl", "start", serviceName)
	if output, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("  Failed to start: %s\n", string(output))
	} else {
		fmt.Printf("  Service started: %s\n", serviceName)
	}
}
