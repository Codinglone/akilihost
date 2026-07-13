package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var daemonCmd = &cobra.Command{
	Use:   "daemon",
	Short: "Start background daemon for multi-model management",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Starting daemon on port 9500...")
	},
}
