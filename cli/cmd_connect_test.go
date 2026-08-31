package cli

import "testing"

func TestConnectCommandExists(t *testing.T) {
	cmd, _, err := rootCmd.Find([]string{"connect"})
	if err != nil || cmd == nil {
		t.Fatalf("connect command not found")
	}
	if cmd.Flags().Lookup("host") == nil {
		t.Error("missing --host")
	}
	if cmd.Flags().Lookup("alias") == nil {
		t.Error("missing --alias")
	}
}
