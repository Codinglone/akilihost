package host

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type OpencodeConfig struct {
	Schema   string                 `json:"$schema,omitempty"`
	Plugin   []string               `json:"plugin,omitempty"`
	Provider map[string]interface{} `json:"provider"`
	Model    string                 `json:"model,omitempty"`
}

func PatchOpencode(path string, port int) (string, error) {
	var backup string
	data, err := os.ReadFile(path)
	cfg := make(map[string]interface{})
	if err == nil && len(data) > 0 {
		if err := json.Unmarshal(data, &cfg); err != nil {
			return "", err
		}
		backup = fmt.Sprintf("%s.bak.%d", path, time.Now().Unix())
		if err := os.WriteFile(backup, data, 0644); err != nil {
			return "", err
		}
	} else {
		cfg["$schema"] = "https://opencode.ai/config.json"
		cfg["plugin"] = []string{"superpowers@git+https://github.com/obra/superpowers.git", "opencode-wakatime"}
	}
	if _, ok := cfg["provider"]; !ok {
		cfg["provider"] = make(map[string]interface{})
	}
	prov := cfg["provider"].(map[string]interface{})
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
	opts["timeout"] = 600000
	opts["chunkTimeout"] = 120000
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
	return backup, os.WriteFile(path, out, 0644)
}
