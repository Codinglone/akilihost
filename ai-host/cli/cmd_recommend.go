package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var recommendCmd = &cobra.Command{
	Use:   "recommend",
	Short: "Show models that fit your GPU, with benchmarks",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Model recommendations...")
	},
}
