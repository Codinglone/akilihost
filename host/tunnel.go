package host

import (
	"fmt"
	"os"
	"strings"
)

func WriteTunnelService(path, alias string, port int) error {
	if path == "" {
		return fmt.Errorf("invalid path: empty")
	}
	if alias == "" || strings.ContainsAny(alias, " \t\n\r") {
		return fmt.Errorf("invalid alias %q: must be non-empty without spaces/newlines", alias)
	}
	if port < 1 || port > 65535 {
		return fmt.Errorf("invalid port %d: must be 1-65535", port)
	}
	content := fmt.Sprintf(`[Unit]
Description=Autossh tunnel for vLLM API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=AUTOSSH_GATETIME=0
Environment=AUTOSSH_POLL=30
ExecStart=/usr/bin/autossh -M 0 -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -o "ExitOnForwardFailure=yes" -o "StrictHostKeyChecking=accept-new" -L %d:localhost:%d %s
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
`, port, port, alias)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return fmt.Errorf("write tunnel service %s: %w", path, err)
	}
	return nil
}
