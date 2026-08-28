package cli

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Detect GPU, build llama.cpp, install vLLM venv, setup caches",
	Run: func(cmd *cobra.Command, args []string) {
		gpu, err := host.DetectGPU()
		if err != nil {
			fmt.Printf("Error detecting GPU: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("GPU: %s\n", gpu.Name)
		fmt.Printf("VRAM: %d MB (%.1f GB)\n", gpu.TotalVRAMMB, float64(gpu.TotalVRAMMB)/1024)
		fmt.Printf("System RAM: %d MB (%.1f GB)\n", gpu.SystemRAMMB, float64(gpu.SystemRAMMB)/1024)
		fmt.Printf("CUDA: %s\n\n", gpu.CUDAVersion)

		fmt.Println("[1/3] Building llama.cpp with CUDA support...")
		buildLlamaCpp()

		fmt.Println("\n[2/3] Installing huggingface_hub...")
		installHuggingFaceHub()

		fmt.Println("\n[3/3] Creating model cache directory...")
		createModelDir()

		fmt.Println("\nDone! Ready to serve models.")
		fmt.Println("Run: akilihost serve Qwen3.8-27B")
	},
}

func buildLlamaCpp() {
	llamaDir := "/opt/akilihost/llama.cpp"

	if _, err := os.Stat("/usr/local/bin/llama-server"); err == nil {
		fmt.Println("  llama-server already installed, skipping build")
		return
	}

	fmt.Println("  Installing build dependencies...")
	deps := "build-essential cmake git pciutils libcurl4-openssl-dev"
	cmd := exec.Command("sudo", "bash", "-c", "apt-get update -y && apt-get install -y "+deps)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to install deps: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("  Cloning llama.cpp...")
	cmd = exec.Command("sudo", "bash", "-c",
		fmt.Sprintf("mkdir -p /opt/akilihost && git clone https://github.com/ggml-org/llama.cpp %s", llamaDir))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to clone: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("  Building with CUDA (this takes several minutes)...")
	buildCmd := fmt.Sprintf(
		"cd %s && cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON && "+
			"cmake --build build --config Release -j --clean-first --target llama-server && "+
			"cp build/bin/llama-server /usr/local/bin/",
		llamaDir)
	cmd = exec.Command("sudo", "bash", "-c", buildCmd)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Build failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("  llama-server installed to /usr/local/bin/")
}

func installHuggingFaceHub() {
	cmd := exec.Command("pip3", "install", "-U", "huggingface_hub[cli]")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  pip install failed (trying with --user): %v\n", err)
		cmd = exec.Command("pip3", "install", "--user", "-U", "huggingface_hub[cli]")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("  Failed to install huggingface_hub: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Println("  huggingface_hub installed")
}

func createModelDir() {
	cmd := exec.Command("sudo", "mkdir", "-p", "/opt/akilihost/models")
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed: %v\n", err)
		os.Exit(1)
	}
	cmd = exec.Command("sudo", "chown", "-R",
		fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()), "/opt/akilihost/models")
	if err := cmd.Run(); err != nil {
		fmt.Printf("  Failed to chown: %v\n", err)
	}
	fmt.Println("  Model directory: /opt/akilihost/models/")
}
