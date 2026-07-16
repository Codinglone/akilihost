package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Detect GPU, install Python venv + vLLM, setup caches",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("GPU detection, venv setup, and cache configuration...")
	},
}
