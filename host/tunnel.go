package host

import (
	"fmt"
	"os"
)

func WriteTunnelService(path, alias string, port int) error {
	content := fmt.Sprintf(`[Unit]
Description=Autossh tunnel for vLLM API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=codinglone
Environment=AUTOSSH_GATETIME=0
Environment=AUTOSSH_POLL=30
ExecStart=/usr/bin/autossh -M 0 -N -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" -o "ExitOnForwardFailure=yes" -o "StrictHostKeyChecking=accept-new" -L %d:localhost:%d %s
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
`, port, port, alias)
	return os.WriteFile(path, []byte(content), 0644)
}
