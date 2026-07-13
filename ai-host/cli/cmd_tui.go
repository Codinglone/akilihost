package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var tuiCmd = &cobra.Command{
	Use:   "tui",
	Short: "TUI dashboard for running models",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Starting TUI dashboard...")
	},
}
