package host

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

const (
	opencodeTimeout      = 600000
	opencodeChunkTimeout = 120000
)

func PatchOpencode(path string, port int) (string, error) {
	var backup string
	data, err := os.ReadFile(path)
	cfg := make(map[string]interface{})
	if err != nil {
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("read %s: %w", path, err)
		}
		cfg["$schema"] = "https://opencode.ai/config.json"
		cfg["plugin"] = []string{"superpowers@git+https://github.com/obra/superpowers.git", "opencode-wakatime"}
	} else if len(data) > 0 {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return "", fmt.Errorf("unmarshal %s: %w", path, err)
		}
		backup = fmt.Sprintf("%s.bak.%d", path, time.Now().Unix())
		if err := os.WriteFile(backup, data, 0600); err != nil {
			return "", err
		}
		if err := os.Chmod(backup, 0600); err != nil {
			return "", err
		}
	} else {
		cfg["$schema"] = "https://opencode.ai/config.json"
		cfg["plugin"] = []string{"superpowers@git+https://github.com/obra/superpowers.git", "opencode-wakatime"}
	}
	prov, ok := cfg["provider"].(map[string]interface{})
	if !ok {
		prov = make(map[string]interface{})
		cfg["provider"] = prov
	}
	self, ok := prov["selfhosted"].(map[string]interface{})
	if !ok {
		self = make(map[string]interface{})
		prov["selfhosted"] = self
	}
	self["npm"] = "@ai-sdk/openai-compatible"
	self["name"] = "Self-Hosted (llama.cpp)"
	opts, ok := self["options"].(map[string]interface{})
	if !ok {
		opts = make(map[string]interface{})
		self["options"] = opts
	}
	opts["baseURL"] = fmt.Sprintf("http://localhost:%d/v1", port)
	opts["timeout"] = opencodeTimeout
	opts["chunkTimeout"] = opencodeChunkTimeout
	// merge models from host/models.go
	modelsMap, ok := self["models"].(map[string]interface{})
	if !ok {
		modelsMap = make(map[string]interface{})
		self["models"] = modelsMap
	}
	// curated from host/prepopulatedModels
	modelsMap["unsloth/Qwen3.8-27B-GGUF"] = map[string]interface{}{"name": "Qwen3.8-27B UD-Q4_K_XL (T4 split)", "maxTokens": 16384}
	modelsMap["Qwen/Qwen3-Coder-Next"] = map[string]interface{}{"name": "Qwen3-Coder-Next FP8 (262K ctx)"}
	modelsMap["mistralai/Devstral-2-123B-Instruct-2512"] = map[string]interface{}{"name": "Devstral 2 123B FP8"}
	modelsMap["Qwen/Qwen2.5-Coder-32B-Instruct"] = map[string]interface{}{"name": "Qwen2.5-Coder 32B (HumanEval 92.7%)"}
	if _, ok := cfg["model"]; !ok {
		cfg["model"] = "selfhosted/unsloth/Qwen3.8-27B-GGUF"
	}
	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return backup, err
	}
	if err := os.WriteFile(path, out, 0600); err != nil {
		return backup, err
	}
	if err := os.Chmod(path, 0600); err != nil {
		return backup, err
	}
	return backup, nil
}
