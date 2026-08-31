package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var connectHost, connectUser, connectKey, connectAlias string
var connectPort, connectTunnelPort int
var connectYes, connectDryRun bool

var connectCmd = &cobra.Command{
	Use:   "connect [alias]",
	Short: "Portable setup for any Ubuntu VM: SSH + init + serve + tunnel + opencode",
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) > 0 && connectAlias == "" {
			connectAlias = args[0]
		}
		if connectAlias == "" {
			connectAlias = "mygpu"
		}
		if connectTunnelPort == 0 {
			connectTunnelPort = connectPort
		}
		fmt.Printf("connect: alias=%s host=%s user=%s key=%s port=%d dry-run=%v\n", connectAlias, connectHost, connectUser, connectKey, connectPort, connectDryRun)
		if err := runConnect(cmd, args); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	},
}

func init() {
	connectCmd.Flags().StringVar(&connectHost, "host", "", "VM HostName/IP (required if alias new)")
	connectCmd.Flags().StringVar(&connectUser, "user", "ubuntu", "SSH user")
	connectCmd.Flags().StringVar(&connectKey, "key", "~/.ssh/id_ed25519", "IdentityFile")
	connectCmd.Flags().StringVar(&connectAlias, "alias", "", "Host alias in ~/.ssh/config")
	connectCmd.Flags().IntVar(&connectPort, "port", 8002, "Remote API port")
	connectCmd.Flags().IntVar(&connectTunnelPort, "tunnel-port", 0, "Local tunnel port")
	connectCmd.Flags().BoolVar(&connectYes, "yes", false, "Skip confirmations")
	connectCmd.Flags().BoolVar(&connectDryRun, "dry-run", false, "Print plan without executing")
}

func runConnect(cmd *cobra.Command, args []string) error {
	return nil
}
