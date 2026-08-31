package host

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func EnsureSSHConfig(path, alias, host, user, key string) (bool, error) {
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return false, err
	}
	content := string(data)
	if strings.Contains(content, "Host "+alias) {
		lines := strings.Split(content, "\n")
		startIdx := -1
		endIdx := len(lines)
		for i, l := range lines {
			trim := strings.TrimSpace(l)
			if strings.HasPrefix(trim, "Host ") {
				if trim == "Host "+alias && startIdx == -1 {
					startIdx = i
					continue
				}
				if startIdx != -1 {
					endIdx = i
					break
				}
			}
		}
		if startIdx == -1 {
			// fallback, should not happen because Contains check passed
			startIdx = 0
		}
		blockLines := lines[startIdx:endIdx]
		// Build new block
		newBlock := []string{}
		// Preserve Host line exactly as "Host <alias>" normalized
		newBlock = append(newBlock, "Host "+alias)

		foundHostName := false
		foundUser := false
		foundIdentityFile := false
		hasIdentitiesOnly := false
		hasStrict := false
		hasAliveInterval := false
		hasAliveCount := false
		hasForward8002 := false
		hasForward8003 := false

		// Process existing block lines after Host line
		for _, l := range blockLines[1:] {
			trim := strings.TrimSpace(l)
			if trim == "" {
				continue
			}
			switch {
			case strings.HasPrefix(trim, "HostName "):
				if !foundHostName {
					newBlock = append(newBlock, "    HostName "+host)
					foundHostName = true
				}
			case strings.HasPrefix(trim, "User "):
				if !foundUser {
					newBlock = append(newBlock, "    User "+user)
					foundUser = true
				}
			case strings.HasPrefix(trim, "IdentityFile "):
				if !foundIdentityFile {
					newBlock = append(newBlock, "    IdentityFile "+key)
					foundIdentityFile = true
				}
			case strings.HasPrefix(trim, "IdentitiesOnly"):
				if !hasIdentitiesOnly {
					newBlock = append(newBlock, "    IdentitiesOnly yes")
					hasIdentitiesOnly = true
				}
			case strings.HasPrefix(trim, "StrictHostKeyChecking"):
				if !hasStrict {
					newBlock = append(newBlock, "    StrictHostKeyChecking accept-new")
					hasStrict = true
				}
			case strings.HasPrefix(trim, "ServerAliveInterval"):
				if !hasAliveInterval {
					newBlock = append(newBlock, "    ServerAliveInterval 30")
					hasAliveInterval = true
				}
			case strings.HasPrefix(trim, "ServerAliveCountMax"):
				if !hasAliveCount {
					newBlock = append(newBlock, "    ServerAliveCountMax 3")
					hasAliveCount = true
				}
			case strings.HasPrefix(trim, "LocalForward 8002"):
				if !hasForward8002 {
					newBlock = append(newBlock, "    LocalForward 8002 127.0.0.1:8002")
					hasForward8002 = true
				}
			case strings.HasPrefix(trim, "LocalForward 8003"):
				if !hasForward8003 {
					newBlock = append(newBlock, "    LocalForward 8003 127.0.0.1:8003")
					hasForward8003 = true
				}
			case strings.HasPrefix(trim, "LocalForward"):
				// preserve other LocalForward not 8002/8003
				newBlock = append(newBlock, l)
			default:
				newBlock = append(newBlock, l)
			}
		}

		if !foundHostName {
			newBlock = append(newBlock, "    HostName "+host)
		}
		if !foundUser {
			newBlock = append(newBlock, "    User "+user)
		}
		if !foundIdentityFile {
			newBlock = append(newBlock, "    IdentityFile "+key)
		}
		if !hasIdentitiesOnly {
			newBlock = append(newBlock, "    IdentitiesOnly yes")
		}
		if !hasStrict {
			newBlock = append(newBlock, "    StrictHostKeyChecking accept-new")
		}
		if !hasAliveInterval {
			newBlock = append(newBlock, "    ServerAliveInterval 30")
		}
		if !hasAliveCount {
			newBlock = append(newBlock, "    ServerAliveCountMax 3")
		}
		if !hasForward8002 {
			newBlock = append(newBlock, "    LocalForward 8002 127.0.0.1:8002")
		}
		if !hasForward8003 {
			newBlock = append(newBlock, "    LocalForward 8003 127.0.0.1:8003")
		}

		newLines := append([]string{}, lines[:startIdx]...)
		newLines = append(newLines, newBlock...)
		newLines = append(newLines, lines[endIdx:]...)
		newContent := strings.Join(newLines, "\n")
		if strings.HasSuffix(content, "\n") && !strings.HasSuffix(newContent, "\n") {
			newContent += "\n"
		}
		if newContent == content {
			return false, nil
		}
		return false, os.WriteFile(path, []byte(newContent), 0644)
	}
	// create new block
	block := fmt.Sprintf("\nHost %s\n    HostName %s\n    User %s\n    IdentityFile %s\n    IdentitiesOnly yes\n    StrictHostKeyChecking accept-new\n    ServerAliveInterval 30\n    ServerAliveCountMax 3\n    LocalForward 8002 127.0.0.1:8002\n    LocalForward 8003 127.0.0.1:8003\n", alias, host, user, key)
	if err := os.WriteFile(path, []byte(content+block), 0644); err != nil {
		return false, err
	}
	return true, nil
}

func ParseSSHConfigForTest(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m := make(map[string]string)
	sc := bufio.NewScanner(f)
	cur := ""
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(l, "Host ") {
			cur = strings.Fields(l)[1]
			m[cur] = ""
		}
	}
	return m, nil
}
