package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "ai-host",
	Short: "Self-hosted AI model manager for single GPU",
	Long: `ai-host automates everything painful about running open-weight LLMs on a single GPU:
- CUDA version detection
- vLLM version selection
- venv setup
- Cache management
- VRAM math
- Quantization selection
- Multi-model lifecycle management`,
}

// Execute adds all child commands to the root command
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(psCmd)
	rootCmd.AddCommand(stopCmd)
	rootCmd.AddCommand(recommendCmd)
	rootCmd.AddCommand(tuiCmd)
	rootCmd.AddCommand(daemonCmd)
}
